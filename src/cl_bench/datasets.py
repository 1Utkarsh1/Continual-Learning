from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets as tv_datasets
from torchvision import transforms

from cl_bench.config import ExperimentConfig, TaskSpec


@dataclass
class TaskLoaders:
    name: str
    dataset: str
    classes: list[int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class SyntheticImageDataset(Dataset):
    """Small deterministic image classification dataset for CI and smoke runs."""

    def __init__(
        self,
        classes: Sequence[int],
        samples_per_class: int,
        image_shape: tuple[int, int, int] = (1, 8, 8),
        noise_std: float = 0.12,
        seed: int = 0,
    ):
        self.classes = [int(label) for label in classes]
        self.data: list[torch.Tensor] = []
        self.targets: list[int] = []
        noise_generator = torch.Generator().manual_seed(seed)

        for class_id in self.classes:
            prototype_generator = torch.Generator().manual_seed(10_003 + class_id * 997)
            prototype = torch.randn(image_shape, generator=prototype_generator) * 0.6
            prototype = prototype + float(class_id) * 0.15
            for _ in range(samples_per_class):
                noise = torch.randn(image_shape, generator=noise_generator) * noise_std
                self.data.append((prototype + noise).float())
                self.targets.append(class_id)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], self.targets[index]


class FeatureCacheDataset(Dataset):
    """Dataset backed by a cached tensor feature file."""

    def __init__(self, cache_path: str | Path):
        payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict) or "features" not in payload or "targets" not in payload:
            raise ValueError(
                "Feature cache must be a torch-saved mapping with 'features' and 'targets'."
            )
        self.data = torch.as_tensor(payload["features"]).float()
        self.targets = torch.as_tensor(payload["targets"]).long().tolist()
        if self.data.size(0) != len(self.targets):
            raise ValueError("Feature cache features and targets have different lengths.")
        classes = payload.get("classes")
        self.classes = list(classes) if classes is not None else sorted(set(self.targets))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], int(self.targets[index])


def build_task_loaders(config: ExperimentConfig) -> tuple[list[TaskLoaders], tuple[int, ...], int]:
    task_loaders: list[TaskLoaders] = []
    all_classes: set[int] = set()

    for task_id, task in enumerate(config.tasks):
        train_dataset, test_dataset, classes = _build_datasets(task, config, task_id)
        all_classes.update(classes)

        train_subset, val_subset = _split_train_validation(
            train_dataset,
            val_fraction=config.val_fraction,
            seed=config.seed + task_id,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + task_id),
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        task_loaders.append(
            TaskLoaders(
                name=task.name,
                dataset=task.dataset,
                classes=classes,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
        )

    if not task_loaders:
        raise ValueError("At least one task must be configured.")

    first_inputs, _ = task_loaders[0].train_loader.dataset[0]
    input_shape = tuple(first_inputs.shape)
    num_classes = max(all_classes) + 1
    return task_loaders, input_shape, num_classes


def _build_datasets(
    task: TaskSpec, config: ExperimentConfig, task_id: int
) -> tuple[Dataset, Dataset, list[int]]:
    dataset_name = task.dataset.lower()
    if dataset_name == "synthetic":
        if task.classes == "all":
            raise ValueError("Synthetic tasks must list explicit class ids.")
        train_samples = task.samples_per_class or 32
        test_samples = task.test_samples_per_class or max(8, train_samples // 2)
        train_dataset = SyntheticImageDataset(
            task.classes,
            samples_per_class=train_samples,
            seed=config.seed + task_id * 100,
        )
        test_dataset = SyntheticImageDataset(
            task.classes,
            samples_per_class=test_samples,
            seed=config.seed + task_id * 100 + 50_000,
        )
        return train_dataset, test_dataset, [int(label) for label in task.classes]

    if dataset_name in {"feature_cache", "cached_features"}:
        if task.train_feature_cache is None or task.test_feature_cache is None:
            raise ValueError(
                "Feature-cache tasks must define train_feature_cache and test_feature_cache."
            )
        train_dataset = FeatureCacheDataset(Path(config.data_dir) / task.train_feature_cache)
        test_dataset = FeatureCacheDataset(Path(config.data_dir) / task.test_feature_cache)
        classes = _resolve_classes(task, train_dataset)
        return (
            _class_subset(train_dataset, classes, task.train_limit, config.seed + task_id),
            _class_subset(test_dataset, classes, task.test_limit, config.seed + task_id + 10_000),
            classes,
        )

    train_dataset = _torchvision_dataset(
        dataset_name,
        Path(config.data_dir),
        train=True,
        augment=config.augment,
    )
    test_dataset = _torchvision_dataset(
        dataset_name,
        Path(config.data_dir),
        train=False,
        augment=False,
    )
    classes = _resolve_classes(task, train_dataset)
    train_subset = _class_subset(train_dataset, classes, task.train_limit, config.seed + task_id)
    test_subset = _class_subset(
        test_dataset, classes, task.test_limit, config.seed + task_id + 10_000
    )
    return train_subset, test_subset, classes


def _torchvision_dataset(dataset_name: str, data_dir: Path, train: bool, augment: bool) -> Dataset:
    transform = _torchvision_transform(dataset_name, train=train, augment=augment)
    if dataset_name == "mnist":
        return tv_datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    if dataset_name == "fashion_mnist":
        return tv_datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
    if dataset_name == "kmnist":
        return tv_datasets.KMNIST(data_dir, train=train, download=True, transform=transform)
    if dataset_name == "cifar10":
        return tv_datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    if dataset_name == "cifar100":
        return tv_datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)
    if dataset_name == "tinyimagenet":
        split = "train" if train else "val"
        root = data_dir / "tiny-imagenet-200" / split
        if not root.exists():
            raise FileNotFoundError(
                "TinyImageNet is not auto-downloaded. Expected ImageFolder layout at "
                f"{root}. Download tiny-imagenet-200 under the configured data_dir."
            )
        return tv_datasets.ImageFolder(root, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _torchvision_transform(dataset_name: str, train: bool, augment: bool) -> transforms.Compose:
    if dataset_name in {"cifar10", "cifar100", "tinyimagenet"}:
        steps: list[object] = []
        if train and augment:
            size = 64 if dataset_name == "tinyimagenet" else 32
            steps.extend(
                [
                    transforms.RandomCrop(size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(num_ops=2, magnitude=9),
                ]
            )
        mean = (0.485, 0.456, 0.406) if dataset_name == "tinyimagenet" else (0.4914, 0.4822, 0.4465)
        std = (0.229, 0.224, 0.225) if dataset_name == "tinyimagenet" else (0.2470, 0.2435, 0.2616)
        steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        if train and augment:
            steps.append(Cutout(size=8 if dataset_name != "tinyimagenet" else 16))
        return transforms.Compose(steps)

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class Cutout:
    """Randomly mask one square region in a tensor image."""

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or self.size <= 0:
            return image
        _, height, width = image.shape
        cutout_height = min(self.size, height)
        cutout_width = min(self.size, width)
        top = int(torch.randint(0, height - cutout_height + 1, (1,)).item())
        left = int(torch.randint(0, width - cutout_width + 1, (1,)).item())
        image = image.clone()
        image[:, top : top + cutout_height, left : left + cutout_width] = 0.0
        return image


def _resolve_classes(task: TaskSpec, dataset: Dataset) -> list[int]:
    if task.classes == "all":
        if not hasattr(dataset, "classes"):
            raise ValueError(f"Dataset {task.dataset} does not expose class metadata.")
        return list(range(len(dataset.classes)))
    return [int(label) for label in task.classes]


def _class_subset(dataset: Dataset, classes: list[int], limit: int | None, seed: int) -> Subset:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [dataset[index][1] for index in range(len(dataset))]
    targets_tensor = torch.as_tensor(targets)
    class_tensor = torch.tensor(classes, dtype=targets_tensor.dtype)
    mask = torch.isin(targets_tensor, class_tensor)
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()

    if limit is not None and limit < len(indices):
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(len(indices), generator=generator)[:limit].tolist()
        indices = [indices[index] for index in selected]

    return Subset(dataset, indices)


def _split_train_validation(
    dataset: Dataset, val_fraction: float, seed: int
) -> tuple[Subset, Subset]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0).")

    total = len(dataset)
    if total < 2 or val_fraction == 0.0:
        return Subset(dataset, list(range(total))), Subset(dataset, list(range(total)))

    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size
    if train_size <= 0:
        train_size, val_size = total - 1, 1

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_subset, val_subset

from __future__ import annotations

import torch

from cl_bench.config import ExperimentConfig, TaskSpec
from cl_bench.datasets import build_task_loaders


def test_synthetic_task_construction_is_deterministic() -> None:
    config = ExperimentConfig(
        name="unit",
        method="baseline",
        seed=11,
        model="linear",
        batch_size=8,
        eval_batch_size=16,
        val_fraction=0.25,
        tasks=[
            TaskSpec(
                name="first",
                dataset="synthetic",
                classes=[0, 1],
                samples_per_class=8,
                test_samples_per_class=4,
            ),
            TaskSpec(
                name="second",
                dataset="synthetic",
                classes=[2, 3],
                samples_per_class=8,
                test_samples_per_class=4,
            ),
        ],
    )

    first_tasks, input_shape, num_classes = build_task_loaders(config)
    second_tasks, second_shape, second_num_classes = build_task_loaders(config)

    assert [task.name for task in first_tasks] == ["first", "second"]
    assert input_shape == (1, 8, 8)
    assert num_classes == 4
    assert second_shape == input_shape
    assert second_num_classes == num_classes

    first_batch = next(iter(first_tasks[0].train_loader))
    second_batch = next(iter(second_tasks[0].train_loader))
    assert first_batch[1].tolist() == second_batch[1].tolist()


def test_cifar_task_construction_uses_rgb_shape_and_limits(monkeypatch) -> None:
    class FakeCIFAR10:
        classes = [str(index) for index in range(10)]

        def __init__(self, root, train: bool, download: bool, transform):
            del root, download
            self.transform = transform
            examples_per_class = 6 if train else 4
            self.targets = [class_id for class_id in range(10) for _ in range(examples_per_class)]
            self.data = [
                torch.full((3, 32, 32), float(class_id) / 10.0)
                for class_id in range(10)
                for _ in range(examples_per_class)
            ]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int):
            return self.data[index], self.targets[index]

    monkeypatch.setattr("cl_bench.datasets.tv_datasets.CIFAR10", FakeCIFAR10)
    config = ExperimentConfig(
        name="cifar_unit",
        method="baseline",
        seed=3,
        model="cifar_convnet",
        batch_size=4,
        eval_batch_size=8,
        val_fraction=0.0,
        augment=False,
        tasks=[
            TaskSpec(
                name="cifar_0_1",
                dataset="cifar10",
                classes=[0, 1],
                train_limit=5,
                test_limit=4,
            )
        ],
    )

    tasks, input_shape, num_classes = build_task_loaders(config)

    assert input_shape == (3, 32, 32)
    assert num_classes == 2
    assert len(tasks[0].train_loader.dataset) == 5
    assert len(tasks[0].test_loader.dataset) == 4

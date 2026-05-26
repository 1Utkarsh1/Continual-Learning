from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare TinyImageNet in torchvision ImageFolder layout."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--url", default=TINY_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "tiny-imagenet-200.zip"
    dataset_dir = data_dir / "tiny-imagenet-200"

    if args.force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    if not dataset_dir.exists():
        if not archive_path.exists():
            print(f"Downloading TinyImageNet from {args.url} to {archive_path}")
            urllib.request.urlretrieve(args.url, archive_path)
        print(f"Extracting {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)

    _prepare_val_split(dataset_dir)
    _validate_layout(dataset_dir)
    print(f"TinyImageNet is ready at {dataset_dir}")
    return 0


def _prepare_val_split(dataset_dir: Path) -> None:
    val_dir = dataset_dir / "val"
    annotations_path = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"
    if not annotations_path.exists():
        if any(path.is_dir() for path in val_dir.iterdir()):
            return
        raise FileNotFoundError(f"Missing TinyImageNet validation annotations: {annotations_path}")

    image_to_class: dict[str, str] = {}
    for line in annotations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_name, class_id, *_ = line.split("\t")
        image_to_class[image_name] = class_id

    for image_name, class_id in image_to_class.items():
        source = images_dir / image_name
        if not source.exists():
            target = val_dir / class_id / "images" / image_name
            if target.exists():
                continue
            raise FileNotFoundError(f"Missing TinyImageNet validation image: {source}")
        target_dir = val_dir / class_id / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target_dir / image_name))

    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()


def _validate_layout(dataset_dir: Path) -> None:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("TinyImageNet must contain train and val directories.")
    train_classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    val_classes = sorted(path.name for path in val_dir.iterdir() if path.is_dir())
    if len(train_classes) != 200 or len(val_classes) != 200:
        raise RuntimeError(
            "TinyImageNet layout is incomplete: "
            f"found {len(train_classes)} train classes and {len(val_classes)} val classes."
        )
    if train_classes != val_classes:
        raise RuntimeError("TinyImageNet train/val class folders do not match.")


if __name__ == "__main__":
    raise SystemExit(main())

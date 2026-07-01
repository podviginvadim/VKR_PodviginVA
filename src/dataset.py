from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

DATASETS = {
    "buildings": {
        "root_name": "spacenet_buildings",
        "num_classes": 2,
        "class_names_ru": ["Фон", "Здания"],
        "colors": np.array([[0, 0, 0], [255, 60, 60]], dtype=np.uint8),
        "type": "binary",
    },
    "roads": {
        "root_name": "spacenet_roads",
        "num_classes": 2,
        "class_names_ru": ["Фон", "Дороги"],
        "colors": np.array([[0, 0, 0], [70, 170, 255]], dtype=np.uint8),
        "type": "binary",
    },
    "loveda": {
        "root_name": "loveda",
        "num_classes": 7,
        "class_names_ru": ["Фон", "Здания", "Дороги", "Вода", "Пустырь", "Лес", "Сельхозземли"],
        "colors": np.array([
            [0, 0, 0], [220, 60, 60], [70, 170, 255], [60, 120, 255],
            [210, 180, 100], [40, 160, 70], [180, 220, 80],
        ], dtype=np.uint8),
        "type": "multiclass",
    },
}

ALIASES = {
    "sn2": "buildings", "sn2_buildings": "buildings", "spacenet_buildings": "buildings",
    "sn3": "roads", "sn3_roads": "roads", "spacenet_roads": "roads",
    "loveda_multiclass": "loveda",
}


def normalize_dataset_name(name: str) -> str:
    key = ALIASES.get(name, name)
    if key not in DATASETS:
        raise ValueError(f"Неизвестный датасет: {name}. Доступно: {list(DATASETS)}")
    return key


def dataset_meta(dataset: str) -> dict:
    return DATASETS[normalize_dataset_name(dataset)]


def num_classes(dataset: str) -> int:
    return int(dataset_meta(dataset)["num_classes"])


def class_info(dataset: str) -> dict:
    meta = dataset_meta(dataset)
    return {
        i: {"name": meta["class_names_ru"][i], "color": tuple(map(int, meta["colors"][i]))}
        for i in range(meta["num_classes"])
    }


def resolve_dataset_root(data_root: str | Path, dataset: str) -> tuple[Path, str]:
    dataset = normalize_dataset_name(dataset)
    root = Path(data_root)
    candidate = root / DATASETS[dataset]["root_name"]
    if candidate.exists():
        return candidate, dataset
    return root, dataset


def _transforms(image_size: int, train: bool) -> A.Compose:
    ops = [A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0)]
    if train:
        ops += [
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ]
    else:
        ops += [A.CenterCrop(height=image_size, width=image_size, p=1.0)]
    ops += [A.Normalize(mean=IMG_MEAN, std=IMG_STD), ToTensorV2()]
    return A.Compose(ops)


class SegmentationDataset(Dataset):
    """Универсальный датасет проекта.

    Ожидаемая структура для каждого набора:
      root/train/images/*.png
      root/train/masks/*.png
      root/val/images/*.png
      root/val/masks/*.png
      root/test/images/*.png
      root/test/masks/*.png
    """

    def __init__(self, root: str | Path, split: str, dataset: str, image_size: int = 256, subset: int | float | None = None):
        self.root = Path(root)
        self.split = split
        self.dataset = normalize_dataset_name(dataset)
        self.meta = DATASETS[self.dataset]
        self.image_size = image_size

        img_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"Не найдена папка изображений: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Не найдена папка масок: {mask_dir}")

        self.images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.images:
            raise RuntimeError(f"В {img_dir} нет изображений")
        self.masks = [mask_dir / f"{p.stem}.png" for p in self.images]
        missing = [p for p in self.masks if not p.exists()]
        if missing:
            raise RuntimeError(f"Не найдены маски: {len(missing)}. Пример: {missing[0]}")

        if split == "train" and subset not in (None, 0, 1.0):
            if isinstance(subset, float) and subset < 1:
                n = max(1, int(len(self.images) * subset))
            else:
                n = min(len(self.images), int(subset))
            rnd = random.Random(42)
            idx = sorted(rnd.sample(range(len(self.images)), n))
            self.images = [self.images[i] for i in idx]
            self.masks = [self.masks[i] for i in idx]

        self.transforms = _transforms(image_size, train=(split == "train"))
        print(f"[{self.dataset} {split}] {len(self.images)} снимков")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {self.images[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Не удалось прочитать маску: {self.masks[idx]}")
        mask = mask.astype(np.int64)
        mask[mask == 255] = 0
        if self.meta["type"] == "binary":
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.clip(mask, 0, self.meta["num_classes"] - 1).astype(np.uint8)

        aug = self.transforms(image=img, mask=mask)
        return {"image": aug["image"].float(), "mask": aug["mask"].long(), "path": str(self.images[idx])}


def get_dataloaders(data_root: str | Path, dataset: str, batch_size: int, image_size: int, workers: int = 0, subset: int | float | None = None) -> Tuple[DataLoader, DataLoader, DataLoader, str, int]:
    root, dataset = resolve_dataset_root(data_root, dataset)
    train_ds = SegmentationDataset(root, "train", dataset, image_size, subset=subset)
    val_ds = SegmentationDataset(root, "val", dataset, image_size)
    test_ds = SegmentationDataset(root, "test", dataset, image_size)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, test_loader, dataset, num_classes(dataset)


def mask_to_rgb(mask: np.ndarray, dataset: str) -> np.ndarray:
    meta = dataset_meta(dataset)
    mask = np.clip(mask, 0, meta["num_classes"] - 1).astype(np.uint8)
    return meta["colors"][mask]

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {path}")
    return img


def read_mask(path: Path, label_format: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Не удалось прочитать маску: {path}")
    mask = mask.astype(np.int16)
    # Поддержка двух частых вариантов:
    # zero_based: 0..6
    # one_based:  1..7, 0 или 255 трактуются как фон/ignore
    if label_format == "one_based":
        mask[mask == 255] = 0
        pos = mask > 0
        mask[pos] -= 1
    elif label_format == "auto":
        vals = set(np.unique(mask).tolist())
        if max(vals) == 7 or vals.issubset(set(range(1, 8)) | {0, 255}):
            mask[mask == 255] = 0
            pos = mask > 0
            mask[pos] -= 1
        else:
            mask[mask == 255] = 0
    else:
        mask[mask == 255] = 0
    return np.clip(mask, 0, 6).astype(np.uint8)


def find_pairs(raw_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    # LoveDA обычно имеет папки images_png/masks_png внутри Train/Val и Urban/Rural.
    for img_dir in raw_root.rglob("images_png"):
        mask_dir = img_dir.parent / "masks_png"
        if not mask_dir.exists():
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            # чаще всего маска имеет то же имя
            candidates = [mask_dir / f"{img_path.stem}.png", mask_dir / img_path.name]
            mask_path = next((c for c in candidates if c.exists()), None)
            if mask_path:
                pairs.append((img_path, mask_path))
    # Универсальный fallback для структуры images/masks
    if not pairs:
        for img_dir in raw_root.rglob("images"):
            mask_dir = img_dir.parent / "masks"
            if not mask_dir.exists():
                continue
            for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
                candidates = [mask_dir / f"{img_path.stem}.png", mask_dir / img_path.name]
                mask_path = next((c for c in candidates if c.exists()), None)
                if mask_path:
                    pairs.append((img_path, mask_path))
    return pairs


def write_pair(img_path: Path, mask_path: Path, out_img: Path, out_mask: Path, label_format: str):
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    img = read_image(img_path)
    mask = read_mask(mask_path, label_format)
    cv2.imwrite(str(out_img), img)
    cv2.imwrite(str(out_mask), mask)


def split_pairs(pairs: list[tuple[Path, Path]], seed: int, train: float, val: float):
    rnd = random.Random(seed)
    pairs = pairs[:]
    rnd.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }


def main():
    p = argparse.ArgumentParser(description="Подготовка LoveDA к формату проекта")
    p.add_argument("--raw-root", required=True, help="Папка с распакованным LoveDA")
    p.add_argument("--out", default="data/loveda", help="Выходная папка")
    p.add_argument("--max-items", type=int, default=None, help="Ограничить число пар для быстрой проверки")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--label-format", choices=["auto", "zero_based", "one_based"], default="auto")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    out = Path(args.out)
    pairs = find_pairs(raw_root)
    if not pairs:
        raise RuntimeError(f"Не найдены пары image/mask в {raw_root}. Ожидается LoveDA с images_png/masks_png.")
    if args.max_items:
        pairs = pairs[:args.max_items]
    print(f"Найдено пар image/mask: {len(pairs)}")
    parts = split_pairs(pairs, args.seed, args.train, args.val)

    if out.exists():
        print(f"Внимание: выходная папка уже существует: {out}. Файлы будут дозаписаны/перезаписаны.")
    for split, items in parts.items():
        print(f"{split}: {len(items)}")
        for i, (img, mask) in enumerate(tqdm(items, desc=f"write {split}")):
            name = f"loveda_{split}_{i:06d}.png"
            write_pair(img, mask, out / split / "images" / name, out / split / "masks" / name, args.label_format)
    print(f"Готово: {out}")


if __name__ == "__main__":
    main()

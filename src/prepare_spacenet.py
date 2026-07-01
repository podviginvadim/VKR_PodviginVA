from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from tqdm import tqdm


def read_rgb_geotiff(path: Path) -> tuple[np.ndarray, rasterio.Affine, tuple[int, int]]:
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]) if src.count >= 3 else np.repeat(src.read(1)[None, ...], 3, axis=0)
        arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            lo, hi = np.percentile(arr, [2, 98])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
            arr = (arr * 255).astype(np.uint8)
        return arr, src.transform, (src.height, src.width)


def img_id(path: Path) -> str | None:
    m = re.search(r"img(\d+)", path.stem)
    return m.group(1) if m else None


def find_geojson(image_path: Path, geojson_dir: Path) -> Path | None:
    ident = img_id(image_path)
    if ident is None:
        return None
    matches = sorted(geojson_dir.glob(f"*img{ident}.geojson")) + sorted(geojson_dir.glob(f"*img{ident}.json"))
    return matches[0] if matches else None


def load_geometries(path: Path | None, buffer_size: float = 0.0) -> list:
    if path is None or not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    geoms = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            obj = shape(geom)
            if buffer_size > 0:
                obj = obj.buffer(buffer_size)
            if not obj.is_empty:
                geoms.append(obj)
        except Exception:
            continue
    return geoms


def rasterize_mask(image_path: Path, geojson_path: Path | None, class_value: int = 1, buffer_size: float = 0.0) -> np.ndarray:
    with rasterio.open(image_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
    geoms = load_geometries(geojson_path, buffer_size=buffer_size)
    if not geoms:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterize(
        [(g, class_value) for g in geoms],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def collect_images(images_dir: Path) -> list[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)


def convert_source(images_dir: Path, geojson_dir: Path, out_dir: Path, prefix: str, max_items: int | None, buffer_size: float):
    images = collect_images(images_dir)
    if max_items:
        images = images[:max_items]
    if not images:
        raise FileNotFoundError(f"Не найдены изображения в {images_dir}")

    img_out = out_dir / "all" / "images"
    mask_out = out_dir / "all" / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    non_empty = 0
    print(f"{prefix}: найдено {len(images)} изображений")
    for i, image_path in enumerate(tqdm(images, desc=f"convert {prefix}")):
        geo_path = find_geojson(image_path, geojson_dir)
        image, _, _ = read_rgb_geotiff(image_path)
        mask = rasterize_mask(image_path, geo_path, class_value=1, buffer_size=buffer_size)
        if mask.max() > 0:
            non_empty += 1
        name = f"{prefix}_{i:06d}.png"
        cv2.imwrite(str(img_out / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_out / name), mask)
    print(f"{prefix}: непустых масок {non_empty}/{len(images)}")


def split_dataset(root: Path, train: float = 0.7, val: float = 0.15, seed: int = 42):
    images = sorted((root / "all" / "images").glob("*.png"))
    if not images:
        raise RuntimeError(f"Нет подготовленных изображений в {root / 'all/images'}")
    rnd = random.Random(seed)
    rnd.shuffle(images)
    n = len(images)
    n_train = int(n * train)
    n_val = int(n * val)
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }
    for split, items in splits.items():
        for sub in ["images", "masks"]:
            shutil.rmtree(root / split / sub, ignore_errors=True)
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for img in items:
            mask = root / "all" / "masks" / img.name
            shutil.copy2(img, root / split / "images" / img.name)
            shutil.copy2(mask, root / split / "masks" / img.name)
        print(f"{root.name}/{split}: {len(items)}")


def verify_masks(root: Path):
    mask_dir = root / "all" / "masks"
    total = non_empty = pixels = obj_pixels = 0
    for mpath in mask_dir.glob("*.png"):
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        total += 1
        obj = int((m > 0).sum())
        pixels += m.size
        obj_pixels += obj
        if obj > 0:
            non_empty += 1
    ratio = obj_pixels / max(pixels, 1)
    print(f"Проверка {root.name}: непустых масок {non_empty}/{total}, доля объекта {ratio:.4%}")


def main():
    p = argparse.ArgumentParser(description="Подготовка SpaceNet 2 buildings и SpaceNet 3 roads как двух независимых бинарных датасетов")
    p.add_argument("--task", choices=["both", "buildings", "roads"], default="both")
    p.add_argument("--sn2-images", help="SpaceNet 2 PS-RGB")
    p.add_argument("--sn2-geojson", help="SpaceNet 2 geojson_buildings")
    p.add_argument("--sn3-images", help="SpaceNet 3 PS-RGB")
    p.add_argument("--sn3-geojson", help="SpaceNet 3 geojson_roads")
    p.add_argument("--out-root", default="data")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--road-buffer", type=float, default=0.0, help="Буфер для линий дорог в единицах CRS; обычно можно оставить 0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_root = Path(args.out_root)
    if args.task in ["both", "buildings"]:
        if not args.sn2_images or not args.sn2_geojson:
            raise SystemExit("Для buildings нужны --sn2-images и --sn2-geojson")
        out = out_root / "spacenet_buildings"
        shutil.rmtree(out / "all", ignore_errors=True)
        convert_source(Path(args.sn2_images), Path(args.sn2_geojson), out, "sn2_buildings", args.max_items, 0.0)
        split_dataset(out, seed=args.seed)
        verify_masks(out)

    if args.task in ["both", "roads"]:
        if not args.sn3_images or not args.sn3_geojson:
            raise SystemExit("Для roads нужны --sn3-images и --sn3-geojson")
        out = out_root / "spacenet_roads"
        shutil.rmtree(out / "all", ignore_errors=True)
        convert_source(Path(args.sn3_images), Path(args.sn3_geojson), out, "sn3_roads", args.max_items, args.road_buffer)
        split_dataset(out, seed=args.seed)
        verify_masks(out)


if __name__ == "__main__":
    main()

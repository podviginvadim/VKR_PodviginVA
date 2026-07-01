from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from inference import load_checkpoint, predict, read_image
from models import build_model

COLORS = {
    0: (0, 0, 0),
    1: (255, 60, 60),   # buildings, RGB
    2: (70, 170, 255),  # roads, RGB
}


def load_binary_model(model_name: str, checkpoint: str, image_size: int, device):
    model = build_model(model_name, 2, image_size).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    print(f"Загружен чекпоинт: {checkpoint}")
    return model


def combine_masks(mask_buildings: np.ndarray, mask_roads: np.ndarray, road_priority: bool = True) -> np.ndarray:
    if mask_buildings.shape != mask_roads.shape:
        mask_roads = cv2.resize(mask_roads, (mask_buildings.shape[1], mask_buildings.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined = np.zeros(mask_buildings.shape, dtype=np.uint8)
    combined[mask_buildings == 1] = 1
    if road_priority:
        combined[mask_roads == 1] = 2
    else:
        combined[(mask_roads == 1) & (combined == 0)] = 2
    return combined


def combined_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb


def save_outputs(image: np.ndarray, buildings: np.ndarray, roads: np.ndarray, combined: np.ndarray, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    color = combined_to_rgb(combined)
    overlay = cv2.addWeighted(image, 0.65, color, 0.35, 0)
    cv2.imwrite(str(out_dir / f"{stem}_buildings_mask.png"), buildings)
    cv2.imwrite(str(out_dir / f"{stem}_roads_mask.png"), roads)
    cv2.imwrite(str(out_dir / f"{stem}_combined_mask.png"), combined)
    cv2.imwrite(str(out_dir / f"{stem}_combined_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Сохранено: {out_dir / f'{stem}_overlay.png'}")


def collect_images(image: str | None, image_dir: str | None) -> list[Path]:
    if image:
        return [Path(image)]
    if image_dir:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
    raise SystemExit("Укажи --image или --image-dir")


def main():
    p = argparse.ArgumentParser(description="Ансамбль только для SpaceNet: buildings + roads = карта критической инфраструктуры")
    p.add_argument("--buildings-checkpoint", required=True)
    p.add_argument("--roads-checkpoint", required=True)
    p.add_argument("--buildings-model", default="segformer_b0", choices=["unet", "segformer_b0", "segformer_b1", "segformer_b2"])
    p.add_argument("--roads-model", default="segformer_b0", choices=["unet", "segformer_b0", "segformer_b1", "segformer_b2"])
    p.add_argument("--image", default=None)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--output", default="results/critical_infrastructure")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--sliding-window", action="store_true")
    p.add_argument("--no-road-priority", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buildings_model = load_binary_model(args.buildings_model, args.buildings_checkpoint, args.image_size, device)
    roads_model = load_binary_model(args.roads_model, args.roads_checkpoint, args.image_size, device)

    for path in collect_images(args.image, args.image_dir):
        image = read_image(path)
        mask_buildings = predict(buildings_model, image, device, 2, args.image_size, args.sliding_window)
        mask_roads = predict(roads_model, image, device, 2, args.image_size, args.sliding_window)
        combined = combine_masks(mask_buildings, mask_roads, road_priority=not args.no_road_priority)
        save_outputs(image, mask_buildings, mask_roads, combined, Path(args.output), path.stem)
        print("Классы итоговой маски:", dict(zip(*np.unique(combined, return_counts=True))))


if __name__ == "__main__":
    main()

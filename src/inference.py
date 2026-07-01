from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import IMG_MEAN, IMG_STD, mask_to_rgb, num_classes
from models import build_model


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(image: np.ndarray, size: int) -> torch.Tensor:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(IMG_MEAN, dtype=np.float32)) / np.array(IMG_STD, dtype=np.float32)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def load_checkpoint(model, checkpoint: str, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


@torch.no_grad()
def predict(model, image: np.ndarray, device, n_classes: int, size: int = 256, sliding_window: bool = False) -> np.ndarray:
    H, W = image.shape[:2]
    model.eval()
    if not sliding_window or (H <= size and W <= size):
        x = preprocess(image, size).to(device)
        pred = model(x).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

    stride = max(1, size // 2)
    prob_acc = np.zeros((n_classes, H, W), dtype=np.float32)
    count_acc = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(H - size + 1, 1), stride)) or [0]
    xs = list(range(0, max(W - size + 1, 1), stride)) or [0]
    if ys[-1] != max(H - size, 0):
        ys.append(max(H - size, 0))
    if xs[-1] != max(W - size, 0):
        xs.append(max(W - size, 0))

    for y in ys:
        for x in xs:
            tile = image[y:y + size, x:x + size]
            h, w = tile.shape[:2]
            logits = model(preprocess(tile, size).to(device))
            probs = logits.softmax(1).squeeze(0).cpu().numpy()
            probs = probs[:, :h, :w]
            prob_acc[:, y:y + h, x:x + w] += probs
            count_acc[y:y + h, x:x + w] += 1
    return (prob_acc / np.maximum(count_acc, 1e-8)).argmax(0).astype(np.uint8)


def save_result(image: np.ndarray, mask: np.ndarray, dataset: str, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    color = mask_to_rgb(mask, dataset)
    overlay = cv2.addWeighted(image, 0.65, color, 0.35, 0)
    cv2.imwrite(str(output), mask)
    cv2.imwrite(str(output.with_name(output.stem + "_color.png")), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output.with_name(output.stem + "_overlay.png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Сохранено: {output}")


def main():
    p = argparse.ArgumentParser(description="Инференс одной модели: buildings, roads или loveda")
    p.add_argument("--dataset", required=True, choices=["buildings", "roads", "loveda"])
    p.add_argument("--model", default="unet", choices=["unet", "segformer_b0", "segformer_b1", "segformer_b2"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="results/prediction.png")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--sliding-window", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = num_classes(args.dataset)
    model = build_model(args.model, n_classes, args.image_size).to(device)
    load_checkpoint(model, args.checkpoint, device)
    image = read_image(Path(args.image))
    mask = predict(model, image, device, n_classes, args.image_size, args.sliding_window)
    save_result(image, mask, args.dataset, Path(args.output))
    unique, counts = np.unique(mask, return_counts=True)
    print(dict(zip(unique.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()

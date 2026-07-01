from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import class_info, get_dataloaders, num_classes
from models import build_model


def confusion_matrix(pred: np.ndarray, target: np.ndarray, n_classes: int) -> np.ndarray:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = (target >= 0) & (target < n_classes)
    return np.bincount(n_classes * target[mask].astype(int) + pred[mask].astype(int), minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics_from_cm(cm: np.ndarray, dataset: str) -> dict:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc = tp.sum() / (cm.sum() + 1e-8)
    info = class_info(dataset)
    out = {"pixel_acc": float(acc), "mIoU": float(iou.mean()), "mF1": float(f1.mean())}
    for c in range(len(iou)):
        out[f"IoU/{info[c]['name']}"] = float(iou[c])
        out[f"F1/{info[c]['name']}"] = float(f1[c])
    return out


def print_metrics(metrics: dict) -> None:
    print(f"Pixel Acc: {metrics['pixel_acc']:.4f}")
    print(f"mIoU:      {metrics['mIoU']:.4f}")
    print(f"mF1:       {metrics['mF1']:.4f}")
    for k, v in metrics.items():
        if k.startswith("IoU/") or k.startswith("F1/"):
            print(f"{k:22s}: {v:.4f}")


@torch.no_grad()
def evaluate_model(model, loader, dataset: str, device, n_classes: int) -> dict:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for batch in tqdm(loader, desc="eval"):
        x = batch["image"].to(device)
        y = batch["mask"].numpy()
        pred = model(x).argmax(1).cpu().numpy()
        cm += confusion_matrix(pred, y, n_classes)
    return metrics_from_cm(cm, dataset)


def load_checkpoint(model, checkpoint: str, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def main():
    p = argparse.ArgumentParser(description="Оценка модели на test split")
    p.add_argument("--dataset", required=True, choices=["buildings", "roads", "loveda"])
    p.add_argument("--model", default="unet", choices=["unet", "segformer_b0", "segformer_b1", "segformer_b2"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--output", default=None, help="Опционально сохранить метрики в JSON")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, dataset, n_classes = get_dataloaders(args.data_root, args.dataset, batch_size=1, image_size=args.image_size, workers=0)
    model = build_model(args.model, n_classes, args.image_size).to(device)
    ckpt = args.checkpoint or f"models/{dataset}/{args.model}/best_model.pth"
    load_checkpoint(model, ckpt, device)
    metrics = evaluate_model(model, test_loader, dataset, device, n_classes)
    print_metrics(metrics)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

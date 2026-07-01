from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset import class_info, get_dataloaders, num_classes
from evaluate import confusion_matrix, metrics_from_cm, print_metrics
from models import build_model


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, n_classes: int, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.dice = DiceLoss(n_classes)

    def forward(self, logits, target):
        return self.alpha * self.ce(logits, target) + (1 - self.alpha) * self.dice(logits, target)


@torch.no_grad()
def validate(model, loader, dataset: str, device, n_classes: int) -> dict:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["image"].to(device)
        y = batch["mask"].numpy()
        pred = model(x).argmax(1).cpu().numpy()
        cm += confusion_matrix(pred, y, n_classes)
    return metrics_from_cm(cm, dataset)


def save_checkpoint(path: Path, model, args: argparse.Namespace, metrics: dict, dataset: str, n_classes: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "dataset": dataset,
        "num_classes": n_classes,
    }, path)


def main():
    p = argparse.ArgumentParser(description="Обучение U-Net / SegFormer для SpaceNet2, SpaceNet3 или LoveDA")
    p.add_argument("--dataset", required=True, choices=["buildings", "roads", "loveda"])
    p.add_argument("--model", default="unet", choices=["unet", "segformer_b0", "segformer_b1", "segformer_b2"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--subset", type=float, default=None, help="Если >=1 — число train-снимков; если <1 — доля train")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == "cpu" and args.model == "segformer_b2":
        print("Внимание: SegFormer-B2 на CPU очень медленный и может быть завершен системой из-за RAM.")

    lr = args.lr if args.lr is not None else (6e-5 if args.model.startswith("segformer") else 1e-3)
    train_loader, val_loader, _, dataset, n_classes = get_dataloaders(
        args.data_root, args.dataset, args.batch_size, args.image_size, args.workers, subset=args.subset
    )
    model = build_model(args.model, n_classes, args.image_size).to(device)
    print(f"Модель: {args.model}, классов: {n_classes}, параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print(f"Загружен чекпоинт: {args.resume}")

    criterion = CombinedLoss(n_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=lr * 0.05)

    ckpt_dir = Path("models") / dataset / args.model
    best_path = ckpt_dir / "best_model.pth"
    last_path = ckpt_dir / "last_model.pth"
    best_miou = -1.0
    history = []

    print("=" * 60)
    print(f"Датасет: {dataset}")
    print(f"Классы: {[class_info(dataset)[i]['name'] for i in range(n_classes)]}")
    print(f"Модель: {args.model}")
    print(f"Эпох: {args.epochs}, batch: {args.batch_size}, image: {args.image_size}, lr: {lr}")
    print("=" * 60)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = validate(model, val_loader, dataset, device, n_classes)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(f"Эпоха {epoch + 1}: loss={train_loss:.4f}, mIoU={metrics['mIoU']:.4f}, time={time.time() - t0:.0f}s")

        save_checkpoint(last_path, model, args, metrics, dataset, n_classes)
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            save_checkpoint(best_path, model, args, metrics, dataset, n_classes)
            print(f"  ↑ новый лучший mIoU: {best_miou:.4f}")
            print_metrics(metrics)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Готово. Лучшая модель: {best_path}")


if __name__ == "__main__":
    main()

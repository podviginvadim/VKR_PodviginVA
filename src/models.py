from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, num_classes: int = 2, base: int = 32):
        super().__init__()
        self.e1 = DoubleConv(3, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.head(d1)


class SegFormerWrapper(nn.Module):
    HF_NAMES = {
        "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    }

    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        hf_name = self.HF_NAMES[model_name]
        id2label = {i: str(i) for i in range(num_classes)}
        label2id = {v: k for k, v in id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hf_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x).logits
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(name: str, num_classes: int = 2, image_size: int = 256) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(num_classes=num_classes)
    if name in SegFormerWrapper.HF_NAMES:
        return SegFormerWrapper(name, num_classes=num_classes)
    raise ValueError("--model должен быть: unet, segformer_b0, segformer_b1 или segformer_b2")

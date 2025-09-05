from __future__ import annotations
import math
from typing import List
import torch
from torch import nn, Tensor

def max_depth_from_hw(H: int, W: int) -> int:
    return int(math.floor(math.log2(min(H, W))))

class DoubleConv(nn.Module):
    net: nn.Sequential
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Down(nn.Module):
    net: nn.Sequential
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            DoubleConv(in_ch, out_ch),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class BottleNeck(nn.Module):
    net: nn.Sequential
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            DoubleConv(in_ch, out_ch),
            DoubleConv(out_ch, out_ch),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

def center_crop(enc_feat: Tensor, target_h: int, target_w: int) -> Tensor:
    _, _, h, w = enc_feat.shape
    dh = (h - target_h) // 2
    dw = (w - target_w) // 2
    return enc_feat[:, :, dh:dh + target_h, dw:dw + target_w]

class Up(nn.Module):
    up: nn.ConvTranspose2d
    conv: DoubleConv
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch=out_ch * 2, out_ch=out_ch)
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            target_h = min(x.shape[-2], skip.shape[-2])
            target_w = min(x.shape[-1], skip.shape[-1])
            x = center_crop(x, target_h, target_w)
            skip = center_crop(skip, target_h, target_w)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    stem: DoubleConv
    downs: nn.ModuleList
    bottleneck: BottleNeck
    ups: nn.ModuleList
    head: nn.Conv2d
    def __init__(self, num_classes: int, depth: int, in_channels: int = 3, width: int = 64) -> None:
        super().__init__()
        assert depth >= 1
        widths: List[int] = [width * (2 ** i) for i in range(depth)]
        self.stem = DoubleConv(in_ch=in_channels, out_ch=widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.bottleneck = BottleNeck(widths[-1], widths[-1])
        self.ups = nn.ModuleList([Up(widths[i], widths[i - 1]) for i in range(len(widths) - 1, 0, -1)])
        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1, bias=True)
    def forward(self, x: Tensor) -> Tensor:
        skips: List[Tensor] = []
        x = self.stem(x)

        for down in self.downs:
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)
        logits: Tensor = self.head(x)
        return logits

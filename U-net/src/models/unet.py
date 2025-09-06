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
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
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
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]

            pad_x = [0, 0, 0, 0] 
            pad_skip = [0, 0, 0, 0]

            if diff_w > 0:
                pad_x[0] = diff_w // 2
                pad_x[1] = diff_w - pad_x[0]
            elif diff_w < 0:
                dw = -diff_w
                pad_skip[0] = dw // 2
                pad_skip[1] = dw - pad_skip[0]

            if diff_h > 0:
                pad_x[2] = diff_h // 2
                pad_x[3] = diff_h - pad_x[2]
            elif diff_h < 0:
                dh = -diff_h
                pad_skip[2] = dh // 2
                pad_skip[3] = dh - pad_skip[2]

            if any(p > 0 for p in pad_x):
                x = torch.nn.functional.pad(x, pad_x)
            if any(p > 0 for p in pad_skip):
                skip = torch.nn.functional.pad(skip, pad_skip)

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

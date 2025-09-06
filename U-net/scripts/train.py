from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader, Dataset

ROOT: Path = Path(__file__).resolve().parents[1]
SRC: Path = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import CamVidDataset  
from config import TRAIN_PATH, MASK_PATH, COLORS_PATH
from models.unet import Unet, max_depth_from_hw

import torch
import torch.nn as nn
import torch.optim as opti
import pandas as pd

PathLike = Union[str, Path]


def create_camvid_dataloader(
    image_dir: PathLike = TRAIN_PATH,
    mask_dir: PathLike = MASK_PATH,
    colors_csv: PathLike = COLORS_PATH,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    ignore_index: Optional[int] = None,
) -> DataLoader[tuple]:
    """
    Create a DataLoader for a CamVid-style dataset using the existing CamVidDataset.

    - image_dir: path to images folder (e.g., U-net/data/CamVid/train)
    - mask_dir:  path to masks folder  (e.g., U-net/data/CamVid/train_labels)
    - colors_csv: CSV with columns r,g,b (row order defines class ids)

    Returns a torch DataLoader yielding (image, mask) batches where:
      image: float32 tensor [B, 3, H, W] in [0,1]
      mask:  long   tensor [B, H, W] with class ids
    """

    def _resolve(p: PathLike) -> Path:
        pth: Path = Path(p)
        return pth if pth.is_absolute() else (ROOT / pth)

    image_dir_p: Path = _resolve(image_dir)
    mask_dir_p: Path = _resolve(mask_dir)
    colors_csv_p: Path = _resolve(colors_csv)

    ds: Dataset = CamVidDataset(
        image_dir=image_dir_p,
        mask_dir=mask_dir_p,
        colors_csv=colors_csv_p,
        ignore_index=ignore_index,
    )
    loader: DataLoader[tuple] = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader

def check_model_validity() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_camvid_dataloader(
        batch_size=1, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )

    try:
        x_cpu, y_cpu = next(iter(loader))
    except StopIteration:
        raise RuntimeError("Dataset appears empty. Check TRAIN_PATH/MASK_PATH.")

    H, W = y_cpu.shape[-2], y_cpu.shape[-1]
    crop_h = min(256, H)
    crop_w = min(256, W)
    off_h = max(0, (H - crop_h) // 2)
    off_w = max(0, (W - crop_w) // 2)
    x_cpu = x_cpu[:, :, off_h:off_h + crop_h, off_w:off_w + crop_w]
    y_cpu = y_cpu[:, off_h:off_h + crop_h, off_w:off_w + crop_w]

    colors_csv_p = (ROOT / COLORS_PATH) if not Path(COLORS_PATH).is_absolute() else Path(COLORS_PATH)
    if not colors_csv_p.exists():
        raise FileNotFoundError(f"Colors CSV not found at: {colors_csv_p}")
    num_classes = int(pd.read_csv(colors_csv_p).shape[0])

    depth_cap = 4
    depth = max(1, min(depth_cap, max_depth_from_hw(crop_h, crop_w)))
    model = Unet(num_classes=num_classes, depth=depth, in_channels=3, width=32).to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = opti.Adam(model.parameters(), lr=1e-3)

    x = x_cpu.to(device)
    y = y_cpu.to(device)

    model.train()
    steps = 200
    print(f"[Sanity] device={device}, img={tuple(x.shape)}, classes={num_classes}, depth={depth}")
    last_loss = None
    for it in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        if it % 20 == 0 or it == 1:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                pix_acc = (preds == y).float().mean().item()
            print(f"step {it:03d} | loss {last_loss:.4f} | pix_acc {pix_acc:.3f}")

        if last_loss is not None and last_loss < 0.01:
            print(f"Early stop at step {it}: loss={last_loss:.4f}")
            break

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
        pix_acc = (preds == y).float().mean().item()
    print(f"Final: loss {last_loss:.4f} | pixel_acc {pix_acc:.3f}")

print(Unet)

if __name__ == "__main__":
    check_model_validity()

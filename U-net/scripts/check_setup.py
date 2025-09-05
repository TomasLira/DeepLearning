from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader


def add_src_to_path() -> None:
    this_dir = Path(__file__).resolve().parent
    u_net_root = this_dir.parent
    src_path = u_net_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


add_src_to_path()

from data import CamVidDataset 
from models.unet import Unet, center_crop


def center_crop_2d(mask: Tensor, target_h: int, target_w: int) -> Tensor:
    """Center-crop a 2D or 3D tensor to (target_h, target_w).

    Accepts tensors shaped [H, W] or [N, H, W].
    """
    if mask.dim() == 2:
        h, w = mask.shape
        dh = (h - target_h) // 2
        dw = (w - target_w) // 2
        return mask[dh : dh + target_h, dw : dw + target_w]
    elif mask.dim() == 3:
        n, h, w = mask.shape
        dh = (h - target_h) // 2
        dw = (w - target_w) // 2
        return mask[:, dh : dh + target_h, dw : dw + target_w]
    else:
        raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")


def infer_num_classes(colors_csv: Path) -> int:
    import pandas as pd

    df = pd.read_csv(colors_csv)
    return len(df)


def load_split(root: Path, split: str, ignore_index: int | None = None) -> CamVidDataset:
    images = root / split
    labels = root / f"{split}_labels"
    colors_csv = root / "class_dict.csv"
    assert images.exists() and labels.exists(), f"Missing split folders for {split} under {root}"
    ds = CamVidDataset(
        image_dir=images,
        mask_dir=labels,
        colors_csv=colors_csv,
        ignore_index=ignore_index,
    )
    return ds


def check_batch_shapes(
    images: Tensor, masks: Tensor, num_classes: int, depth: int = 4, in_channels: int = 3, width: int = 64
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run a forward pass and compute a dummy loss to validate pipeline.

    Returns (logits, cropped_masks, loss)
    """
    model = Unet(num_classes=num_classes, depth=depth, in_channels=in_channels, width=width)
    model.eval()

    def debug_forward(model: Unet, x: Tensor) -> Tensor:
        """
        Run a verbose forward pass through the U-Net, printing feature shapes
        at each stage to verify channels and spatial sizes.
        """
        prints = []
        def p(msg: str) -> None:
            prints.append(msg)

        b, c, h, w = x.shape
        p(f"input: b={b} c={c} h={h} w={w}")

        # Stem
        x = model.stem(x)
        b, c, h, w = x.shape
        p(f"stem:  b={b} c={c} h={h} w={w}")

        # Encoder (downs)
        skips: list[Tensor] = []
        for i, down in enumerate(model.downs):
            skips.append(x)
            before = x.shape
            x = down(x)
            after = x.shape
            p(f"down[{i}]: in={tuple(before)} out={tuple(after)} | skip={tuple(skips[-1].shape)}")

        # Bottleneck
        before = x.shape
        x = model.bottleneck(x)
        after = x.shape
        p(f"bottleneck: in={tuple(before)} out={tuple(after)}")

        # Decoder (ups)
        for i, up in enumerate(model.ups):
            skip = skips.pop()
            x_pre = x
            # 1) transposed conv upsample
            x = up.up(x)
            p(f"up[{i}].up: x={tuple(x_pre.shape)} -> {tuple(x.shape)}; skip={tuple(skip.shape)}")
            # 2) crop both to match smallest spatial dims (same as model.Up.forward)
            if x.shape[-2:] != skip.shape[-2:]:
                target_h = min(x.shape[-2], skip.shape[-2])
                target_w = min(x.shape[-1], skip.shape[-1])
                x = center_crop(x, target_h, target_w)
                skip = center_crop(skip, target_h, target_w)
                p(f"up[{i}].crop: x={tuple(x.shape)}; skip={tuple(skip.shape)} -> target=({target_h},{target_w})")
            # 3) concat and conv
            x_cat = torch.cat([skip, x], dim=1)
            pre_conv = x_cat.shape
            x = up.conv(x_cat)
            p(f"up[{i}].conv: in={tuple(pre_conv)} out={tuple(x.shape)}")

        # Head logits
        pre_head = x.shape
        logits: Tensor = model.head(x)
        p(f"head: in={tuple(pre_head)} out={tuple(logits.shape)}")

        # Emit prints at the end to keep order tidy
        for line in prints:
            print(line)
        return logits

    with torch.inference_mode():
        logits: Tensor = debug_forward(model, images)

    _, _, H_out, W_out = logits.shape
    masks_c = center_crop_2d(masks, H_out, W_out)

    assert images.dtype == torch.float32, f"Images dtype {images.dtype} != float32"
    assert masks.dtype == torch.long, f"Masks dtype {masks.dtype} != long"
    assert logits.shape[0] == images.shape[0], "Batch size mismatch"
    assert logits.shape[1] == num_classes, f"num_classes mismatch: {logits.shape[1]} != {num_classes}"

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, masks_c)
    return logits, masks_c, loss


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Check U-Net feature and pixel dimensions")
    parser.add_argument("--dummy", action="store_true", help="Use dummy random input instead of dataset")
    parser.add_argument("--height", type=int, default=256, help="Dummy/image height")
    parser.add_argument("--width", type=int, default=256, help="Dummy/image width")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--classes", type=int, default=None, help="Override num classes")
    parser.add_argument("--depth", type=int, default=4, help="U-Net depth")
    parser.add_argument("--channels", type=int, default=3, help="Input channels")
    parser.add_argument("--width_mult", type=int, default=64, help="Base width multiplier")
    args = parser.parse_args()

    if args.dummy:
        num_classes = args.classes or 12
        images = torch.randn(args.batch, args.channels, args.height, args.width)
        masks = torch.zeros(args.batch, args.height, args.width, dtype=torch.long)
        print(f"Using dummy input: images={tuple(images.shape)} masks={tuple(masks.shape)} classes={num_classes}")
    else:
        root = Path(__file__).resolve().parents[1] / "data" / "CamVid"
        if not root.exists():
            print(f"CamVid root not found at {root}; falling back to --dummy run.")
            num_classes = args.classes or 12
            images = torch.randn(args.batch, args.channels, args.height, args.width)
            masks = torch.zeros(args.batch, args.height, args.width, dtype=torch.long)
            print(f"Using dummy input: images={tuple(images.shape)} masks={tuple(masks.shape)} classes={num_classes}")
        else:
            train_ds = load_split(root, "train")
            val_ds = load_split(root, "val")
            test_ds = load_split(root, "test")
            print(f"Train samples: {len(train_ds)}; Val: {len(val_ds)}; Test: {len(test_ds)}")
            num_classes = args.classes or infer_num_classes(root / "class_dict.csv")
            print(f"Detected classes: {num_classes}")
            loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False)
            images, masks = next(iter(loader))
            print(f"Batch shapes -> images: {tuple(images.shape)}, masks: {tuple(masks.shape)}")

    logits, masks_c, loss = check_batch_shapes(
        images,
        masks,
        num_classes=num_classes,
        depth=args.depth,
        in_channels=args.channels,
        width=args.width_mult,
    )
    print(f"Logits shape: {tuple(logits.shape)}; Cropped masks: {tuple(masks_c.shape)}")
    print(f"Dummy CE loss: {loss.item():.4f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

Color = Tuple[int, int, int]                 
ColorToId = Dict[Color, int]                
PathLike = Union[str, Path]

class MaskToClassIds:
    """
    Convert an RGB color-encoded segmentation mask to a class-id tensor.

    Input (to __call__):
        mask_img : PIL.Image.Image
            RGB or convertible-to-RGB mask. Each pixel is an (R,G,B) color.

    Output:
        torch.Tensor
            Long tensor of shape [H, W] with class ids (dtype = torch.long).

    Errors:
        Raises ValueError if unknown colors are found and `ignore_index` is None.
    """

    def __init__(self, colors_id: ColorToId, ignore_index: Optional[int] = None) -> None:
        self.colors2id: ColorToId = colors_id
        self.ignore_index: Optional[int] = ignore_index

    def __call__(self, mask_img: Image.Image) -> Tensor:
        rgb_np: np.ndarray = np.array(mask_img.convert("RGB"), dtype=np.uint8)   
        H, W, _ = rgb_np.shape

        flat: np.ndarray = rgb_np.reshape(-1, 3)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)  

        ids_unique = np.empty(len(uniq), dtype=np.int32)
        unknown: List[Color] = []
        for i, (r, g, b) in enumerate(uniq):
            key: Color = (int(r), int(g), int(b))
            if key in self.colors2id:
                ids_unique[i] = int(self.colors2id[key])
            else:
                if self.ignore_index is None:
                    unknown.append(key)
                else:
                    ids_unique[i] = int(self.ignore_index)

        if unknown:
            raise ValueError(f"Unknown mask colors not in class_dict")

        class_ids_np: np.ndarray = ids_unique[inv].reshape(H, W).astype(np.int64)  
        class_ids_t: Tensor = torch.from_numpy(class_ids_np) 
        return class_ids_t



class CamVidDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    CamVid dataset producing:
        image : torch.Tensor  -> shape [3, H, W], dtype=float32 in [0,1]
        mask  : torch.Tensor  -> shape [H, W],   dtype=torch.long (class ids)

    Constructor inputs:
        image_dir  : directory with *.png RGB images
        mask_dir   : directory with *.png RGB masks (color-encoded)
        colors_csv : CSV with columns r, g, b (row order defines class ids)
        ignore_index : optional class id for unknown colors (e.g., 255)
    """

    def __init__(
        self,
        image_dir: PathLike,
        mask_dir: PathLike,
        colors_csv: PathLike,
        ignore_index: Optional[int] = None,
    ) -> None:
        self.image_root: List[Path] = sorted(Path(image_dir).glob("*.png"))
        self.mask_root: List[Path]  = sorted(Path(mask_dir).glob("*.png"))
        assert len(self.image_root) == len(self.mask_root), "Image/mask counts differ."

        self.image_transform = transforms.ToTensor()

        df = pd.read_csv(colors_csv)
        self.color2id: ColorToId = {
            (int(r), int(g), int(b)): class_id
            for class_id, (r, g, b) in enumerate(zip(df.r, df.g, df.b))
        }

        self.mask_transform = MaskToClassIds(self.color2id, ignore_index=ignore_index)

    def __len__(self) -> int:
        return len(self.image_root)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_pil: Image.Image = Image.open(self.image_root[idx]).convert("RGB")
        msk_pil: Image.Image = Image.open(self.mask_root[idx]).convert("RGB")

        img_t: Tensor = self.image_transform(img_pil)   
        msk_t: Tensor = self.mask_transform(msk_pil)   

        return img_t, msk_t

# data/mrsi_y_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict


class MRSiYDataset(Dataset):
    """
    Liefert pro Sample:
        inp_noisy : maskiertes Noisy-Bild  (2,H,W)
        inp_lr    : LowRank (2,H,W) – ggf. (teil-)maskiert, je nach Transform-Flags
        target    : Original Noisy        (2,H,W)
        mask      : Bool-Maske pro Noisy-Kanal (2,H,W) → Loss nur dort auswerten
    """
    def __init__(
        self,
        noisy_data    : np.ndarray,
        lowrank_data  : np.ndarray,
        image_axes    : Tuple[int,int],
        fixed_indices : Optional[Dict[int,int]] = None,
        transform=None,
        num_samples   : int = 10000,
    ):
        assert noisy_data.shape == lowrank_data.shape, \
            "Noisy- und LowRank-Arrays müssen identische Shape besitzen!"
        self.noisy   = noisy_data
        self.lr      = lowrank_data
        self.image_axes = image_axes
        self.fixed      = fixed_indices or {}
        self.transform  = transform
        self.num_samples= num_samples
        self.other_axes = [ax for ax in range(self.noisy.ndim)
                           if ax not in image_axes]

    def __len__(self): 
        return self.num_samples

    def __getitem__(self, idx):
        # --- Zufällige/Fixed Indices wählen ----------------------------------
        sel = {ax: (self.fixed[ax] if ax in self.fixed
                    else np.random.randint(self.noisy.shape[ax]))
               for ax in self.other_axes}

        slicer = [slice(None) if ax in self.image_axes else sel[ax]
                  for ax in range(self.noisy.ndim)]

        arr_noisy = self.noisy[tuple(slicer)]
        arr_lr    = self.lr   [tuple(slicer)]

        # --- Complex → 2-Kanäle ----------------------------------------------
        def to_channels(arr):
            return np.stack(
                [arr.real.astype(np.float32),
                 arr.imag.astype(np.float32)],
                axis=0
            )

        img_noisy = to_channels(arr_noisy)  # (2,H,W)
        img_lr    = to_channels(arr_lr)     # (2,H,W)

        # Reihenfolge wie in Transform erwartet: [LR, NO]
        img_comb  = np.concatenate([img_lr, img_noisy], axis=0)  # (4,H,W)

        _, H, W = img_comb.shape

        if self.transform is not None:
            inp_comb, tgt_comb, mask = self.transform(img_comb)  # mask: (2,H,W) (Noisy)
            inp_lr    = inp_comb[:2]   # (2,H,W)
            inp_noisy = inp_comb[2:]   # (2,H,W)
            tgt       = tgt_comb[2:]   # (2,H,W)  – Original Noisy als Target
        else:
            inp_lr    = img_lr
            inp_noisy = img_noisy
            tgt       = img_noisy
            mask      = np.ones((2, H, W), dtype=bool)

        return (
            torch.from_numpy(inp_noisy),
            torch.from_numpy(inp_lr),
            torch.from_numpy(tgt),
            torch.from_numpy(mask.astype(np.float32)),  # (2,H,W)
        )





# data/mrsi_y_dataset.py  (neu)
import numpy as np, torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List

class MRSiYDataset(Dataset):
    """
    Liefert pro Sample:
        inp_noisy  : gemasktes Noisy-Bild  (C,H,W)
        inp_lr     : unmaskiertes LowRank  (C,H,W)
        target     : Original Noisy        (C,H,W)
        mask       : Bool-Maske            (1,H,W)
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

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        # --- Zufällige/Fixed Indices wählen -------------------------------
        sel = {ax: (self.fixed[ax] if ax in self.fixed
                    else np.random.randint(self.noisy.shape[ax]))
               for ax in self.other_axes}

        slicer = [slice(None) if ax in self.image_axes else sel[ax]
                  for ax in range(self.noisy.ndim)]

        arr_noisy = self.noisy [tuple(slicer)]
        arr_lr    = self.lr    [tuple(slicer)]

        # --- Complex → 2-Kanäle ------------------------------------------
        def to_channels(arr):
            return np.stack([arr.real.astype(np.float32),
                             arr.imag.astype(np.float32)], axis=0)
        img_noisy = to_channels(arr_noisy)   # (2,H,W)
        img_lr    = to_channels(arr_lr)      # (2,H,W)

        # --- Noise2Void-Mask nur auf Noisy --------------------------------
        if self.transform:
            inp_noisy, tgt, mask = self.transform(img_noisy)
        else:
            inp_noisy, tgt = img_noisy, img_noisy
            mask = np.ones((*img_noisy.shape[1:],), bool)

        # Low-Rank bleibt unmaskiert
        inp_lr = img_lr

        return (torch.from_numpy(inp_noisy),
                torch.from_numpy(inp_lr),
                torch.from_numpy(tgt),
                torch.from_numpy(mask.astype(np.float32)[None]))




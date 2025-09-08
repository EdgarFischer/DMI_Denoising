# data/mrsi_2d_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict


class MRSiNDataset(Dataset):
    """
    Single-Branch 2D MRSI Dataset (kein LowRank).

    Rückgabe pro Sample:
        inp   : maskiertes Input-Bild (2, H, W)  [Real, Imag]
        tgt   : Original-Bild      (2, H, W)
        mask  : Bool-Maske pro Kanal (2, H, W)
                True = Stelle maskiert → Loss dort.
    """
    def __init__(
        self,
        data: np.ndarray,
        image_axes: Tuple[int, int],
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,
        num_samples: int = 10000,
        phase_prob: float = 0,          # <--- neu: Wahrscheinlichkeit für Phase-Jitter
    ):
        self.data        = data
        self.image_axes  = image_axes
        self.fixed       = fixed_indices or {}
        self.transform   = transform
        self.num_samples = num_samples
        self.phase_prob  = float(phase_prob)      # <--- Wichtig
        self.other_axes  = [ax for ax in range(self.data.ndim)
                            if ax not in image_axes]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --- Auswahl der Slice-Indizes -----------------------------------
        sel = {
            ax: (self.fixed[ax] if ax in self.fixed
                 else np.random.randint(self.data.shape[ax]))
            for ax in self.other_axes
        }
        slicer = [
            slice(None) if ax in self.image_axes else sel[ax]
            for ax in range(self.data.ndim)
        ]
        arr2d = self.data[tuple(slicer)]

        # --- Globale Phase (optional) --------------------------------------
        if self.phase_prob > 0 and np.random.rand() < self.phase_prob:
            theta = np.random.rand() * 2 * np.pi
            arr2d = arr2d * np.exp(1j * theta)

        # --- Complex → 2 Kanäle ------------------------------------------
        real = np.real(arr2d).astype(np.float32)
        imag = np.imag(arr2d).astype(np.float32)
        img  = np.stack([real, imag], axis=0)  # (2,H,W)
        _, H, W = img.shape

        # --- Transform anwenden ------------------------------------------
        if self.transform is not None:
            inp, tgt, mask = self.transform(img)  # mask: (2,H,W)
            # Falls dein Transform noch (H,W) oder (1,H,W) liefert, erweitere hier:
            if mask.ndim == 2:  # (H,W) → (2,H,W)
                mask = np.broadcast_to(mask[None], (2, H, W)).copy()
            elif mask.ndim == 3 and mask.shape[0] == 1:
                mask = np.broadcast_to(mask, (2, H, W)).copy()
        else:
            inp  = img
            tgt  = img
            mask = np.ones((2, H, W), dtype=bool)

        # --- direkt mask (2,H,W) zurückgeben, keine zusätzliche None-Achse ---
        return (
            torch.from_numpy(inp) ,                   # (2,H,W)
            torch.from_numpy(tgt) ,                   # (2,H,W)
            torch.from_numpy(mask.astype(np.float32)) # (2,H,W)
        )

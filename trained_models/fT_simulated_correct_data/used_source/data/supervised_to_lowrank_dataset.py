# data/supervised_to_lowrank_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Callable


class SupervisedToLowRankDataset(Dataset):
    """
    Supervised Training auf Low-Rank-Targets (analog MRSiYDataset-Pattern).

    Erwartet:
        noisy_data   : np.ndarray (complex oder real), shape beliebig
        lowrank_data : np.ndarray gleiche Shape wie noisy_data

    image_axes     : Achsen (H,W) o.ä., die als 2D-Input extrahiert werden.
                     Alle anderen Achsen werden bei __getitem__ zufällig oder
                     via fixed_indices indiziert (wie in MRSiYDataset).

    transform(optional):
        Callable(inp_noisy: np.ndarray, target_lr: np.ndarray) -> (inp_noisy, target_lr)
        Beide Arrays sind 2-Kanal (real,imag) Arrays (2,H,W) wenn complex;
        bei realen Inputs -> zweiter Kanal = 0.

    Rückgabe:
        inp_noisy : torch.FloatTensor (2,H,W)
        target_lr : torch.FloatTensor (2,H,W)
    """
    def __init__(
        self,
        noisy_data: np.ndarray,
        lowrank_data: np.ndarray,
        image_axes: Tuple[int, int],
        fixed_indices: Optional[Dict[int, int]] = None,
        transform: Optional[Callable] = None,
        num_samples: int = 10000,
    ):
        assert noisy_data.shape == lowrank_data.shape, \
            "Noisy- und Low-Rank-Daten brauchen identische Shape!"
        self.noisy = noisy_data
        self.lr = lowrank_data
        self.image_axes = image_axes
        self.fixed = fixed_indices or {}
        self.transform = transform
        self.num_samples = num_samples
        self.other_axes = [ax for ax in range(self.noisy.ndim)
                           if ax not in image_axes]

    def __len__(self):
        return self.num_samples

    def _to_channels(self, arr: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(arr):
            real = arr.real.astype(np.float32)
            imag = arr.imag.astype(np.float32)
        else:
            real = arr.astype(np.float32)
            imag = np.zeros_like(real, dtype=np.float32)
        return np.stack([real, imag], axis=0)  # (2,H,W)

    def __getitem__(self, idx):
        # Zufalls-/feste Indizes für Nicht-Image-Achsen
        sel = {
            ax: (self.fixed[ax] if ax in self.fixed
                 else np.random.randint(self.noisy.shape[ax]))
            for ax in self.other_axes
        }
        slicer = [slice(None) if ax in self.image_axes else sel[ax]
                  for ax in range(self.noisy.ndim)]

        arr_noisy = self.noisy[tuple(slicer)]
        arr_lr = self.lr[tuple(slicer)]

        inp_noisy = self._to_channels(arr_noisy)  # (2,H,W)
        target_lr = self._to_channels(arr_lr)     # (2,H,W)

        if self.transform is not None:
            inp_noisy, target_lr = self.transform(inp_noisy, target_lr)

        return torch.from_numpy(inp_noisy), torch.from_numpy(target_lr)

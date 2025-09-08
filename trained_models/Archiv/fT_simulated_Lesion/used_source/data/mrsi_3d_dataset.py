# data/mrsi_3d_dataset.py  (NEU)
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict

class MRSiN3DDataset(Dataset):
    """
    Liefert ein 3-D-Volume exakt in der Reihenfolge, die `volume_axes`
    vorgibt – plus 2 Kanäle (Real, Imag) → Layout (C, D1, D2, D3).
    """

    def __init__(
        self,
        data: np.ndarray,                 # z.B. (X,Y,Z,t,T,D_runs)
        volume_axes: Tuple[int, int, int],# z.B. (2,3,4)  →  (Z,t,T)
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,                   # StratifiedVoxelSelection …
        num_samples: int = 10000,
    ):
        self.data         = data
        self.vol_axes     = volume_axes
        self.fixed        = fixed_indices or {}
        self.transform    = transform
        self.num_samples  = num_samples

        # Alle Nicht-Volumen-Achsen
        self.other_axes = [ax for ax in range(self.data.ndim)
                           if ax not in self.vol_axes]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Zufällige / feste Indizes für alle anderen Achsen wählen
        indices = {ax: (self.fixed[ax] if ax in self.fixed
                        else np.random.randint(self.data.shape[ax]))
                   for ax in self.other_axes}

        # 2) Slicer bauen: slice(None) für volume_axes, festen Index sonst
        slicer = [slice(None) if ax in self.vol_axes else indices[ax]
                  for ax in range(self.data.ndim)]

        # 3) 3-D-Block holen: Reihenfolge entspricht volume_axes
        vol = self.data[tuple(slicer)]                 # shape = (D1,D2,D3)

        # 4) Real/Imag stapeln  →  (2, D1, D2, D3)
        img = np.stack([vol.real, vol.imag], axis=0).astype(np.float32)

        # 5) Optionale Noise2Void-Maske
        if self.transform:
            inp, tgt, mask = self.transform(img)      # mask shape (D1,D2,D3)
        else:
            inp = tgt = img
            mask = np.ones(img.shape[1:], dtype=bool)

        # 6) Rückgabe-Tensoren
        return (
            torch.from_numpy(inp),                         # (2, D1, D2, D3)
            torch.from_numpy(tgt),                         # (2, D1, D2, D3)
            torch.from_numpy(mask.astype(np.float32))[None]# (1, D1, D2, D3)
        )



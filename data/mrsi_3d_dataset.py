# data/mrsi_3d_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict

# data/mrsi_3d_dataset.py
class MRSiN3DDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,            # (X, Y, Z, t, T, D)
        volume_axes: Tuple[int, int, int],  # z.B. (0, 1, 2)
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,              # z.B. UniformVoxelSelection()
        num_samples: int = 10_000,
    ):
        self.data = data
        self.vol_axes = volume_axes
        self.fixed = fixed_indices or {}
        self.transform = transform
        self.num_samples = num_samples

        # Alle Achsen, die nicht im Volumen liegen
        self.other_axes = [ax for ax in range(self.data.ndim)
                           if ax not in self.vol_axes]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Zufällig / fix indexieren
        indices = {
            ax: (self.fixed[ax] if ax in self.fixed
                 else np.random.randint(self.data.shape[ax]))
            for ax in self.other_axes
        }

        # 2) Slicer erstellen
        slicer = [
            slice(None) if ax in self.vol_axes else indices[ax]
            for ax in range(self.data.ndim)
        ]

        # 3) 3-D-Block holen: (X,Y,Z) o. ä.
        vol = self.data[tuple(slicer)]

        # 4) Real/Imag stapeln → (2, X, Y, Z)
        real = np.real(vol).astype(np.float32)
        imag = np.imag(vol).astype(np.float32)
        img = np.stack([real, imag], axis=0)

        # 5) Achsen für PyTorch umordnen  (C, Z, Y, X)
        img = np.transpose(img, (0, 3, 2, 1))   # (2, Z, Y, X)

        # 6) Optional 3-D-Noise2Void-Transform
        if self.transform:
            inp, tgt, mask = self.transform(img)   # mask (Z, Y, X)
        else:
            inp = tgt = img
            mask = np.ones(img.shape[1:], dtype=bool)  # (Z, Y, X)

        # 7) Rückgabe-Tensoren
        return (
            torch.from_numpy(inp),                                # (2, D, H, W)
            torch.from_numpy(tgt),                                # (2, D, H, W)
            torch.from_numpy(mask.astype(np.float32))[None],      # (1, D, H, W)
        )


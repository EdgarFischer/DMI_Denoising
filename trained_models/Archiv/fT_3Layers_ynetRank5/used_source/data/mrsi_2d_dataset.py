# data/mrsi_nd_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict

class MRSiNDataset(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 image_axes: Tuple[int,int],
                 fixed_indices: Optional[Dict[int,int]] = None,
                 transform=None,
                 num_samples: int = 10000):
        """
        data: np.ndarray mit Shape z.B. (X,Y,Z,t,T,D)
        image_axes: die beiden Achsen, die Dein 2D-Bild definieren (z.B. (0,1),(3,4) etc.)
        fixed_indices: Dict[Achse,Index], Achsen, die Du immer auf einen festen Wert setzen willst
        transform: z.B. UniformPixelSelection() für Noise2Void
        num_samples: Länge des Datasets (Anzahl zufälliger Beispiele pro Epoche)
        """
        self.data = data
        self.image_axes = image_axes
        self.fixed = fixed_indices or {}
        self.transform = transform
        self.num_samples = num_samples

        # Alle Achsen, die nicht in image_axes sind, müssen indexiert werden
        self.other_axes = [ax for ax in range(self.data.ndim)
                           if ax not in image_axes]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Für jede „andere“ Achse festen oder zufälligen Index wählen
        indices = {}
        for ax in self.other_axes:
            if ax in self.fixed:
                indices[ax] = self.fixed[ax]
            else:
                indices[ax] = np.random.randint(self.data.shape[ax])

        # 2) Slicer bauen: slice(None) für Bild-Achsen, fester Index sonst
        slicer = []
        for ax in range(self.data.ndim):
            if ax in self.image_axes:
                slicer.append(slice(None))
            else:
                slicer.append(indices[ax])

        # 3) 2D-Array extrahieren
        arr2d = self.data[tuple(slicer)]    # complex64, shape (H, W)

        # → Real- und Imag-Anteil als 2 Kanäle
        real = np.real(arr2d).astype(np.float32)
        imag = np.imag(arr2d).astype(np.float32)
        img  = np.stack([real, imag], axis=0)  # shape (2, H, W)

        # 4) Transform (Noise2Void) auf multi-channel anwenden
        if self.transform:
            inp, tgt, mask = self.transform(img)  # jetzt img mit shape (2, H, W)
        else:
            inp, tgt = img, img
            mask = np.ones((img.shape[1], img.shape[2]), dtype=bool)  # (H, W)

        # 5) Tensoren: Input/Target (2×H×W), Maske (1×H×W)
        return (
            torch.from_numpy(inp),                      # (2, H, W)
            torch.from_numpy(tgt),                      # (2, H, W)
            torch.from_numpy(mask.astype(np.float32))[None],  # (1, H, W)
        )



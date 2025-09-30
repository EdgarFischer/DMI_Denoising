# data/mrsi_3d_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict


class MRSi3DDataset(Dataset):
    """
    Single-Branch 3D MRSI-Dataset für (z,f,T)-Noise2Void.

    • data        : komplexes Array (X, Y, Z, F, T)
    • image_axes  : die drei Achsen (Z,F,T) – default (2,3,4)
    • fixed_indices: optionale {Achse: Index} – z.B. {0: x0, 1: y0} für festen (x,y)
    • transform   : Instanz von StratifiedPixelSelection3D
    • phase_prob  : Wahrscheinlichkeit für globalen Phasen-Jitter pro Sample
    """
    def __init__(
        self,
        data: np.ndarray,
        image_axes: Tuple[int, int, int] = (2, 3, 4),
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,
        num_samples: int = 10000,
        phase_prob: float = 0.0,
    ):
        self.data        = data
        self.image_axes  = image_axes
        self.fixed       = fixed_indices or {}
        self.transform   = transform
        self.num_samples = int(num_samples)
        self.phase_prob  = float(phase_prob)

        self.other_axes  = [ax for ax in range(self.data.ndim)
                            if ax not in image_axes]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --- (x,y) o.ä. wählen -------------------------------------------
        sel = {
            ax: (self.fixed[ax] if ax in self.fixed
                 else np.random.randint(self.data.shape[ax]))
            for ax in self.other_axes
        }
        slicer = [
            slice(None) if ax in self.image_axes else sel[ax]
            for ax in range(self.data.ndim)
        ]
        arr3d = self.data[tuple(slicer)]  # (Z,F,T) komplex

        # --- optionale globale Phase -------------------------------------
        if self.phase_prob > 0 and np.random.rand() < self.phase_prob:
            theta = np.random.rand() * 2 * np.pi
            arr3d = arr3d * np.exp(1j * theta)

        # --- Complex → (2,Z,F,T) -----------------------------------------
        real = np.real(arr3d).astype(np.float32)
        imag = np.imag(arr3d).astype(np.float32)
        img  = np.stack([real, imag], axis=0)  # (2,Z,F,T)

        # --- Transform ---------------------------------------------------
        if self.transform is not None:
            inp, tgt, mask = self.transform(img)  # mask: (2 or 4?, Z,F,T)
        else:
            inp  = img
            tgt  = img
            mask = np.ones_like(img[:2], dtype=bool)  # (2,Z,F,T)

        return (
            torch.from_numpy(inp),                   # (C,Z,F,T)
            torch.from_numpy(tgt),                   # (C,Z,F,T)
            torch.from_numpy(mask.astype(np.float32))# (2,Z,F,T)  (float für Loss)
        )

# data/transforms_3d.py
import numpy as np
from typing import List

# ----------------------------------------------------------------------
# NEU: 1-D-Maskierung entlang der F-Achse, repliziert über (X,Y)
# ----------------------------------------------------------------------
class StratifiedFreqSelection3D:
    """
    Noise2Void-Maske für (x,y,f)-Netz:

    • Eingabe-Shape (C, X, Y, F)
    • Es werden *einzelne Frequenzen* maskiert
    • Die gewählte f-Position gilt für **alle (x,y)**
    • Stratifikation: F in gleich breite Tiles aufteilen
    """

    def __init__(self, num_masked_freq: int = 32):
        self.N = int(num_masked_freq)

    # ---------------------------
    def __call__(self, img: np.ndarray):
        if img.ndim != 4:
            raise ValueError("Erwarte 4D-Array (C,X,Y,F)")

        C, X, Y, F = img.shape
        tgt  = img.copy()
        inp  = img.copy()
        mask = np.zeros((C if C == 2 else 2, X, Y, F), dtype=bool)

        # -------- stratifiziert F wählen ----------
        tile = max(1, F // self.N)
        freqs: List[int] = []
        for start in range(0, F, tile):
            if len(freqs) >= self.N:
                break
            end = min(start + tile, F)
            freqs.append(np.random.randint(start, end))
        # falls zu wenig, random auffüllen
        while len(freqs) < self.N:
            freqs.append(np.random.randint(0, F))

        # --------------- maskieren -----------------
        for f in freqs:
            # Nachbar-Freq auswählen (≠ f)
            while True:
                ff = np.random.randint(0, F)
                if ff != f:
                    break
            # swap für alle (x,y) auf einmal
            inp[..., f] = img[..., ff]
            mask[..., f] = True

        return inp, tgt, mask


import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List


class MRSi3DDataset(Dataset):
    """Flexible 3-way 3‑D MRSI Dataset for Noise2Void.

    Key points
    ----------
    * Accepts **≥5‑D** arrays: `(X, Y, Z, F, T[, D])`.
    * Chooses per sample one spatial axis together with F & T → `(x,f,T)`, `(y,f,T)` or `(z,f,T)`.
    * Returns `(C, Z, F, T)` with **identical Z‑length** across the whole batch so the
      `DataLoader` can stack.
    * **Default rule now:** pad *smaller* spatial slices with **zeros** to the
      **largest spatial size** (`max(X, Y, Z)`).  *No cropping* is performed.
    * If `axis_mode="fixed"`, the original slice length is preserved (no padding).
    """

    _AXIS_TRIPLETS: List[Tuple[int, int, int]] = [(0, 3, 4), (1, 3, 4), (2, 3, 4)]

    def __init__(
        self,
        data: np.ndarray,
        image_axes: Tuple[int, int, int] = (2, 3, 4),
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,
        num_samples: int = 10000,
        phase_prob: float = 0.0,
        axis_mode: str = "random",  # 'random' | 'fixed' | 'xyz_only'
        target_len: Optional[int] = None,  # spatial len after padding; None → max(X,Y,Z)
    ):
        if data.ndim < 5:
            raise ValueError("data needs at least 5 dims (X,Y,Z,F,T[,...])")

        self.data = data
        self.default_axes = tuple(image_axes)
        self.fixed = fixed_indices or {}
        self.transform = transform
        self.num_samples = int(num_samples)
        self.phase_prob = float(phase_prob)
        self.axis_mode = axis_mode.lower()

        if self.axis_mode not in {"random", "fixed", "xyz_only"}:
            raise ValueError("axis_mode must be 'random', 'fixed' or 'xyz_only'")

        # Determine target spatial length ---------------------------------------
        if self.axis_mode == "fixed":
            # keep original length for that axis ⇒ no padding / cropping
            self.target_len = None
        else:
            if target_len is None:
                self.target_len = int(max(data.shape[0:3]))  # pad to largest spatial dim
            else:
                self.target_len = int(target_len)

    # ------------------------------------------------------------------
    def _pick_axes(self):
        if self.axis_mode == "fixed":
            return self.default_axes
        if self.axis_mode == "xyz_only":
            return self._AXIS_TRIPLETS[np.random.randint(2)]  # (0,3,4) or (1,3,4)
        return self._AXIS_TRIPLETS[np.random.randint(3)]

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    def __getitem__(self, idx):  # pylint: disable=unused-argument
        image_axes = self._pick_axes()

        other_axes = [ax for ax in range(self.data.ndim) if ax not in image_axes]
        sel = {
            ax: (self.fixed[ax] if ax in self.fixed else np.random.randint(self.data.shape[ax]))
            for ax in other_axes
        }
        slicer = [slice(None) if ax in image_axes else sel[ax] for ax in range(self.data.ndim)]

        vol = self.data[tuple(slicer)]  # ndim==3, axes order = image_axes order

        # map current order → (0,1,2)
        kept = [ax for ax, sl in enumerate(slicer) if isinstance(sl, slice)]
        current_order = tuple(kept.index(ax) for ax in image_axes)
        if current_order != (0, 1, 2):
            vol = np.moveaxis(vol, current_order, (0, 1, 2))
        arr3d = vol  # (spatial, F, T)

        # Pad smaller spatial slices with zeros to target_len (if set) ----------
        if self.target_len is not None and arr3d.shape[0] < self.target_len:
            pad_total = self.target_len - arr3d.shape[0]
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            arr3d = np.pad(arr3d, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
        # If > target_len: keep original; no cropping so batch shapes still align

        # Optional global phase --------------------------------------------------
        if self.phase_prob > 0 and np.random.rand() < self.phase_prob:
            theta = np.random.rand() * 2 * np.pi
            arr3d = arr3d * np.exp(1j * theta)

        # Complex → channels -----------------------------------------------------
        img = np.stack([arr3d.real.astype(np.float32), arr3d.imag.astype(np.float32)], axis=0)

        # Transform --------------------------------------------------------------
        if self.transform is not None:
            inp, tgt, mask = self.transform(img)
        else:
            inp = tgt = img
            mask = np.ones_like(img[:2], dtype=bool)

        return (
            torch.from_numpy(inp),
            torch.from_numpy(tgt),
            torch.from_numpy(mask.astype(np.float32)),
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


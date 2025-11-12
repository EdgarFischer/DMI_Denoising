# data/mrsi_y_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict

class MRSiYDataset(Dataset):
    """
    Zweipfad-2D-MRSI-Dataset (Noisy + LowRank), analog zu MRSiNDataset.

    Rückgabe pro Sample:
        inp_noisy : maskiertes Noisy-Bild  (2, H, W)   [Real, Imag]
        inp_lr    : (ggf. transformiertes) LowRank     (2, H, W)
        tgt       : Original Noisy                     (2, H, W)
        mask      : Bool-/Float-Maske pro Noisy-Kanal  (2, H, W)
                    True/1.0 = Stelle maskiert → Loss dort.
    """
    def __init__(
        self,
        noisy_data   : np.ndarray,
        lowrank_data : np.ndarray,
        image_axes   : Tuple[int, int],
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,
        num_samples  : int = 10000,
        phase_prob   : float = 0.0,   # wie in MRSiNDataset
    ):
        assert noisy_data.shape == lowrank_data.shape, \
            "Noisy- und LowRank-Arrays müssen identische Shapes besitzen!"
        self.noisy       = noisy_data
        self.lr          = lowrank_data
        self.image_axes  = image_axes
        self.fixed       = fixed_indices or {}
        self.transform   = transform
        self.num_samples = int(num_samples)
        self.phase_prob  = float(phase_prob)

        self.other_axes  = [ax for ax in range(self.noisy.ndim)
                            if ax not in image_axes]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --- Auswahl der Slice-Indizes (fix oder zufällig) -------------------
        sel = {
            ax: (self.fixed[ax] if ax in self.fixed
                 else np.random.randint(self.noisy.shape[ax]))
            for ax in self.other_axes
        }
        slicer = [
            slice(None) if ax in self.image_axes else sel[ax]
            for ax in range(self.noisy.ndim)
        ]

        arr_noisy = self.noisy[tuple(slicer)]
        arr_lr    = self.lr   [tuple(slicer)]

        # --- Optionale gemeinsame globale Phase (wie MRSiNDataset) ----------
        if self.phase_prob > 0 and np.random.rand() < self.phase_prob:
            theta = np.random.rand() * 2 * np.pi
            ph = np.exp(1j * theta).astype(arr_noisy.dtype)
            arr_noisy = arr_noisy * ph
            arr_lr    = arr_lr    * ph  # gleiche Phase für Konsistenz

        # --- Complex → 2 Kanäle ---------------------------------------------
        def to_2ch(arr: np.ndarray) -> np.ndarray:
            return np.stack(
                (arr.real.astype(np.float32),
                 arr.imag.astype(np.float32)),
                axis=0
            )

        img_noisy = to_2ch(arr_noisy)  # (2,H,W)
        img_lr    = to_2ch(arr_lr)     # (2,H,W)

        # Für Transform: wie bisher 4-Kanal-Stack [LR, NOISY]
        img_comb = np.concatenate([img_lr, img_noisy], axis=0)  # (4,H,W)
        _, H, W  = img_comb.shape

        if self.transform is not None:
            inp_comb, tgt_comb, mask = self.transform(img_comb)  # erwartet (4,H,W)

            # Split zurück in Pfade
            inp_lr    = inp_comb[:2]   # (2,H,W)
            inp_noisy = inp_comb[2:]   # (2,H,W)
            tgt       = tgt_comb[2:]   # (2,H,W) – Ziel ist Original-Noisy

            # Masken-Handhabung (robust):
            # - Falls (H,W) oder (1,H,W): auf (2,H,W) für Noisy broadcasten
            # - Falls (2,H,W): direkt nutzen
            # - Falls (4,H,W): Noisy-Teil nehmen (Kanäle 2..3)
            if isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    mask = np.broadcast_to(mask[None], (2, H, W)).copy()
                elif mask.ndim == 3:
                    if mask.shape[0] == 1:
                        mask = np.broadcast_to(mask, (2, H, W)).copy()
                    elif mask.shape[0] == 2:
                        pass  # ok
                    elif mask.shape[0] == 4:
                        mask = mask[2:]  # nur Noisy-Kanäle
                    else:
                        # Fallback: volle Maske für Noisy
                        mask = np.ones((2, H, W), dtype=bool)
                else:
                    mask = np.ones((2, H, W), dtype=bool)
            else:
                # falls Transform z.B. None zurückgibt
                mask = np.ones((2, H, W), dtype=bool)

        else:
            # Kein Transform: Identität + volle Maske
            inp_lr    = img_lr
            inp_noisy = img_noisy
            tgt       = img_noisy
            mask      = np.ones((2, H, W), dtype=bool)

        return (
            torch.from_numpy(inp_noisy),                   # (2,H,W)
            torch.from_numpy(inp_lr),                      # (2,H,W)
            torch.from_numpy(tgt),                         # (2,H,W)
            torch.from_numpy(mask.astype(np.float32)),     # (2,H,W)
        )






# data/mrsi_y_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict

class MRSiYDataset(Dataset):
    """
    Zweipfad-2D-MRSI-Dataset (Noisy + LowRank), analog zu MRSiNDataset.

    Rückgabe pro Sample:
        inp_noisy : MASKIERTES Noisy (2,H,W)        [Real, Imag]
        inp_lr    : MASKIERTES LowRank (2,H,W)
        tgt_noisy : UNMASKIERTES Noisy (2,H,W)
        mask      : Blind-Spot-Maske für Noisy (2,H,W), True/1.0 = Stelle maskiert
        tgt_lr    : UNMASKIERTES LowRank (2,H,W)  <-- NEU
    """
    def __init__(
        self,
        noisy_data   : np.ndarray,
        lowrank_data : np.ndarray,
        image_axes   : Tuple[int, int],
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,
        num_samples  : int = 10000,
        phase_prob   : float = 0.0,
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

        # --- Optionale gemeinsame globale Phase ------------------------------
        if self.phase_prob > 0 and np.random.rand() < self.phase_prob:
            theta = np.random.rand() * 2 * np.pi
            ph = np.exp(1j * theta).astype(arr_noisy.dtype)
            arr_noisy = arr_noisy * ph
            arr_lr    = arr_lr    * ph  # gleiche Phase

        # --- Complex → 2 Kanäle ---------------------------------------------
        def to_2ch(arr: np.ndarray) -> np.ndarray:
            return np.stack(
                (arr.real.astype(np.float32),
                 arr.imag.astype(np.float32)),
                axis=0
            )

        img_noisy = to_2ch(arr_noisy)  # (2,H,W)  UNMASKIERT
        img_lr    = to_2ch(arr_lr)     # (2,H,W)  UNMASKIERT

        # UNMASKIERTE Targets vor Transform sichern
        tgt_noisy_2ch = img_noisy.copy()
        tgt_lr_2ch    = img_lr.copy()

        # Für Transform: 4-Kanal-Stack [LR, NOISY]
        img_comb = np.concatenate([img_lr, img_noisy], axis=0)  # (4,H,W)
        _, H, W  = img_comb.shape

        if self.transform is not None:
            inp_comb, tgt_comb, mask = self.transform(img_comb)  # erwartet (4,H,W)

            # Split zurück in Pfade (Inputs sind MASKIERT/geswapped)
            inp_lr    = inp_comb[:2]   # (2,H,W)
            inp_noisy = inp_comb[2:]   # (2,H,W)

            # Dein Transform liefert als tgt_comb üblicherweise Originale zurück;
            # wir verwenden aber explizit die vor dem Transform gesicherten Targets:
            tgt_noisy = tgt_noisy_2ch

            # Masken-Aufbereitung (auf Noisy-Kanäle normiert)
            if isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    mask = np.broadcast_to(mask[None], (2, H, W)).copy()
                elif mask.ndim == 3:
                    if mask.shape[0] == 1:
                        mask = np.broadcast_to(mask, (2, H, W)).copy()
                    elif mask.shape[0] == 2:
                        pass  # ok
                    elif mask.shape[0] == 4:
                        mask = mask[2:]  # Noisy-Kanäle
                    else:
                        mask = np.ones((2, H, W), dtype=bool)
                else:
                    mask = np.ones((2, H, W), dtype=bool)
            else:
                mask = np.ones((2, H, W), dtype=bool)

        else:
            # Kein Transform: Identität + volle Maske
            inp_lr    = img_lr
            inp_noisy = img_noisy
            tgt_noisy = img_noisy
            mask      = np.ones((2, H, W), dtype=bool)

        return (
            torch.from_numpy(inp_noisy),                   # (2,H,W)  masked noisy (input)
            torch.from_numpy(inp_lr),                      # (2,H,W)  masked lowrank (input)
            torch.from_numpy(tgt_noisy),                   # (2,H,W)  UNmasked noisy (target)
            torch.from_numpy(mask.astype(np.float32)),     # (2,H,W)  blind-spot mask (noisy)
            torch.from_numpy(tgt_lr_2ch),                  # (2,H,W)  UNmasked lowrank (target)  <-- NEU
        )






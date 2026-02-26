import numpy as np
from typing import List, Tuple


class StratifiedPixelSelection3D:
    """
    Noise2Void-ähnliche Maskierung für 3D-Daten (C, Z, F, T).

    • Es wird eine 1D-Maske entlang der Zeitachse T erstellt
    • Die maskierten Zeitpunkte werden auf alle (Z,F) repliziert
    • Für jeden maskierten Zeitpunkt t* wird ein Nachbarzeitpunkt t' gewählt
    • Nachbarzeitpunkt t' ist für alle (Z,F) identisch
    """

    @staticmethod
    def _sample_time_neighbor(j, T, rad):
        j0, j1 = max(j - rad, 0), min(j + rad + 1, T)
        while True:
            cc = np.random.randint(j0, j1)
            if cc != j:
                return cc

    def __init__(
        self,
        num_masked_pixels: int = 1,
        window_size: int = 3,
        random_mask_low_rank: bool = False,
        random_mask_noisy: bool = True,
        swap_mode: str = "time",
    ):
        assert window_size % 2 == 1 and window_size >= 3
        self.N = int(num_masked_pixels)
        self.win = int(window_size)
        self.rad = self.win // 2

        self.random_mask_low_rank = bool(random_mask_low_rank)
        self.random_mask_noisy = bool(random_mask_noisy)
        self.swap_mode = swap_mode  # bleibt erhalten, aber hier immer "time"

    def __call__(self, img: np.ndarray):
        """
        Eingabe:  img (C, Z, F, T)
        Rückgabe: inp, tgt, mask_noisy
        """
        if img.ndim != 4:
            raise ValueError(f"Erwarte 4D‐Array (C,Z,F,T), bekam {img.shape}")

        C, Z, F, T = img.shape
        tgt = img.copy()
        inp = img.copy()
        mask_noisy = np.zeros((C if C == 2 else 2, Z, F, T), dtype=bool)

        # ---- 1D Zeitmasken-Indizes stratifiziert wählen ----
        tile = max(1, T // self.N)
        coords: List[int] = []
        for start in range(0, T, tile):
            if len(coords) >= self.N:
                break
            end = min(start + tile, T)
            coords.append(np.random.randint(start, end))
        while len(coords) < self.N:
            coords.append(np.random.randint(0, T))

        # ---- Maskieren & Swappen (gleiche Zeitpunkte für alle Z,F) ----
        for j_t in coords:
            cc = self._sample_time_neighbor(j_t, T, self.rad)

            if C == 4:
                if self.random_mask_low_rank and self.random_mask_noisy:
                    real_sel = np.random.rand() < 0.5
                    if real_sel:
                        inp[0, :, :, j_t] = img[0, :, :, cc]
                        inp[2, :, :, j_t] = img[2, :, :, cc]
                        mask_noisy[0, :, :, j_t] = True
                    else:
                        inp[1, :, :, j_t] = img[1, :, :, cc]
                        inp[3, :, :, j_t] = img[3, :, :, cc]
                        mask_noisy[1, :, :, j_t] = True

                elif self.random_mask_noisy and not self.random_mask_low_rank:
                    nz_ch_rel = np.random.randint(2)
                    inp[2 + nz_ch_rel, :, :, j_t] = img[2 + nz_ch_rel, :, :, cc]
                    mask_noisy[nz_ch_rel, :, :, j_t] = True
                    inp[0, :, :, j_t] = img[0, :, :, cc]
                    inp[1, :, :, j_t] = img[1, :, :, cc]

                elif self.random_mask_low_rank and not self.random_mask_noisy:
                    lr_ch = np.random.randint(2)
                    inp[lr_ch, :, :, j_t] = img[lr_ch, :, :, cc]
                    inp[2, :, :, j_t] = img[2, :, :, cc]
                    inp[3, :, :, j_t] = img[3, :, :, cc]
                    mask_noisy[:, :, :, j_t] = True

                else:
                    inp[:, :, :, j_t] = img[:, :, :, cc]
                    mask_noisy[:, :, :, j_t] = True

            else:  # C == 2
                if self.random_mask_noisy:
                    ch = np.random.randint(2)
                    inp[ch, :, :, j_t] = img[ch, :, :, cc]
                    mask_noisy[ch, :, :, j_t] = True
                else:
                    inp[:, :, :, j_t] = img[:, :, :, cc]
                    mask_noisy[:, :, :, j_t] = True

        return inp, tgt, mask_noisy













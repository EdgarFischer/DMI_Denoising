# data/transforms.py
import numpy as np

class StratifiedPixelSelection:
    """
    Noise2Void-ähnliche Maskierung für 2D-Daten mit 1D-Zeitmaske.
    - Eingabe: img mit Shape (C, F, T)   (F = "2. Dimension", T = Zeit)
    - Es werden N Zeitpunkte j_t gewählt (stratifiziert über T)
    - Für jeden j_t wird ein Nachbar-Zeitpunkt cc in einem Fenster gewählt
    - Swap/Maske werden dann für ALLE F bei diesem j_t angewandt (Replikation über F)

    Unterstützt C=4  (LR_real, LR_imag, NO_real, NO_imag) und C=2 (real, imag).
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
        num_masked_pixels: int = 8,   # Anzahl maskierter Zeitpunkte
        window_size: int = 3,         # ungerade; Radius = win//2
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

    def __call__(self, img: np.ndarray):
        # img: (C, F, T)
        if img.ndim != 3:
            raise ValueError(f"Erwarte 3D-Array (C,F,T), bekam {img.shape}.")

        C, F, T = img.shape
        if C not in (2, 4):
            raise ValueError(f"Unterstütze nur C=2 oder C=4, bekam C={C}.")

        tgt = img.copy()
        inp = img.copy()
        mask_noisy = np.zeros((2, F, T), dtype=bool)  # 2 Masken: real/imag

        # ---- 1D Zeitmasken-Indizes stratifiziert wählen ----
        tile = max(1, T // self.N) if self.N > 0 else T
        coords = []
        for start in range(0, T, tile):
            if len(coords) >= self.N:
                break
            end = min(start + tile, T)
            coords.append(np.random.randint(start, end))
        while len(coords) < self.N:
            coords.append(np.random.randint(0, T))

        # ---- Maskieren & Swappen: für alle F bei jedem j_t gleicher cc ----
        for j_t in coords[:self.N]:
            cc = self._sample_time_neighbor(j_t, T, self.rad)

            if C == 4:
                if self.random_mask_low_rank and self.random_mask_noisy:
                    # Entweder Real- oder Imag-Paar maskieren
                    if np.random.rand() < 0.5:
                        inp[0, :, j_t] = img[0, :, cc]  # LR_real
                        inp[2, :, j_t] = img[2, :, cc]  # NO_real
                        mask_noisy[0, :, j_t] = True
                    else:
                        inp[1, :, j_t] = img[1, :, cc]  # LR_imag
                        inp[3, :, j_t] = img[3, :, cc]  # NO_imag
                        mask_noisy[1, :, j_t] = True

                elif self.random_mask_noisy and not self.random_mask_low_rank:
                    nz = np.random.randint(2)  # 0=real, 1=imag
                    inp[2 + nz, :, j_t] = img[2 + nz, :, cc]
                    mask_noisy[nz, :, j_t] = True
                    inp[0, :, j_t] = img[0, :, cc]
                    inp[1, :, j_t] = img[1, :, cc]

                elif self.random_mask_low_rank and not self.random_mask_noisy:
                    lr = np.random.randint(2)  # 0=LR_real, 1=LR_imag
                    inp[lr, :, j_t] = img[lr, :, cc]
                    inp[2, :, j_t] = img[2, :, cc]
                    inp[3, :, j_t] = img[3, :, cc]
                    mask_noisy[:, :, j_t] = True

                else:
                    inp[:, :, j_t] = img[:, :, cc]
                    mask_noisy[:, :, j_t] = True

            else:  # C == 2
                if self.random_mask_noisy:
                    ch = np.random.randint(2)  # 0=real, 1=imag
                    inp[ch, :, j_t] = img[ch, :, cc]
                    mask_noisy[ch, :, j_t] = True
                else:
                    inp[:, :, j_t] = img[:, :, cc]
                    mask_noisy[:, :, j_t] = True

        return inp, tgt, mask_noisy












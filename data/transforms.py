# data/transforms.py
import numpy as np

# class StratifiedPixelSelection:
#     """
#     Noise2Void-ähnliche Maskierung für 2D-Daten.

#     Erwartet Eingaben mit Kanälen:
#       C=4: [LR_real, LR_imag, NO_real, NO_imag]
#       C=2: [real, imag]

#     Parameter:
#       num_masked_pixels, window_size, random_mask_low_rank, random_mask_noisy: wie gehabt
#       swap_mode : {"both", "time", "freq"}  -> Richtung des Tausch-Nachbarn
#                   "time" = nur entlang der Zeit (2. Index, j-Richtung)
#                   "freq" = nur entlang der Frequenz (1. Index, i-Richtung)
#                   "both" = wie bisher: 2D-Fenster
#     """
#     def __init__(
#         self,
#         num_masked_pixels: int = 64,
#         window_size: int = 3,
#         random_mask_low_rank: bool = False,
#         random_mask_noisy: bool = True,
#         swap_mode: str = "both",
#     ):
#         assert window_size % 2 == 1, "window_size muss ungerade sein"
#         assert window_size >= 3, "window_size muss ≥3 sein"
#         assert swap_mode in {"both", "time", "freq"}, "swap_mode muss 'both', 'time' oder 'freq' sein"

#         self.N   = int(num_masked_pixels)
#         self.win = int(window_size)
#         self.rad = self.win // 2

#         self.random_mask_low_rank = bool(random_mask_low_rank)
#         self.random_mask_noisy    = bool(random_mask_noisy)
#         self.swap_mode            = swap_mode

#     # ---------------- Helper fürs 2D-Fenster ----------------
#     @staticmethod
#     def _sample_other_pixel(i, j, i0, i1, j0, j1):
#         """Zufälliger Pixel im Fenster, aber NICHT (i,j)."""
#         area = (i1 - i0) * (j1 - j0)
#         if area <= 1:
#             raise ValueError("Fenster enthält nur den Zentrumspixel.")
#         while True:
#             rr = np.random.randint(i0, i1)
#             cc = np.random.randint(j0, j1)
#             if rr != i or cc != j:
#                 return rr, cc

#     # -------------- Helper nur Zeitrichtung -----------------
#     @staticmethod
#     def _sample_time_neighbor(i, j, W, rad):
#         j0, j1 = max(j - rad, 0), min(j + rad + 1, W)
#         if (j1 - j0) <= 1:
#             # kein anderer Zeitindex verfügbar → gib trotzdem j zurück (fallback)
#             return i, j
#         while True:
#             cc = np.random.randint(j0, j1)
#             if cc != j:
#                 return i, cc

#     # -------------- Helper nur Frequenzrichtung --------------
#     @staticmethod
#     def _sample_freq_neighbor(i, j, H, rad):
#         i0, i1 = max(i - rad, 0), min(i + rad + 1, H)
#         if (i1 - i0) <= 1:
#             return i, j
#         while True:
#             rr = np.random.randint(i0, i1)
#             if rr != i:
#                 return rr, j

#     def __call__(self, img: np.ndarray):
#         if img.ndim != 3:
#             raise ValueError(f"Erwarte 3D-Array (C,H,W), bekam {img.shape}.")

#         C, H, W = img.shape
#         tgt  = img.copy()
#         inp  = img.copy()

#         if C == 4:
#             mask_noisy = np.zeros((2, H, W), dtype=bool)
#         elif C == 2:
#             mask_noisy = np.zeros((2, H, W), dtype=bool)
#         else:
#             raise ValueError(f"Unterstütze nur C=2 oder C=4, bekam C={C}.")

#         # --- Stratified Pixel-Auswahl wie gehabt ---------------------------
#         n_rows = max(1, int(np.sqrt(self.N * H / W)))
#         n_cols = int(np.ceil(self.N / n_rows))
#         tile_h = H / n_rows
#         tile_w = W / n_cols

#         coords = []
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if len(coords) >= self.N: break
#                 y0, y1 = int(i * tile_h), min(int((i+1)*tile_h), H)
#                 x0, x1 = int(j * tile_w), min(int((j+1)*tile_w), W)
#                 if y1 > y0 and x1 > x0:
#                     coords.append((np.random.randint(y0, y1),
#                                    np.random.randint(x0, x1)))
#             if len(coords) >= self.N: break

#         if len(coords) < self.N:
#             all_coords = [(ii, jj) for ii in range(H) for jj in range(W)]
#             np.random.shuffle(all_coords)
#             for (ii, jj) in all_coords:
#                 if len(coords) >= self.N: break
#                 if (ii, jj) not in coords:
#                     coords.append((ii, jj))

#         # --- Maskieren -----------------------------------------------------
#         for (i, j) in coords[:self.N]:

#             if self.swap_mode == "both":
#                 i0, i1 = max(i - self.rad, 0), min(i + self.rad + 1, H)
#                 j0, j1 = max(j - self.rad, 0), min(j + self.rad + 1, W)
#                 rr, cc = self._sample_other_pixel(i, j, i0, i1, j0, j1)

#             elif self.swap_mode == "time":
#                 rr, cc = self._sample_time_neighbor(i, j, W, self.rad)

#             else:  # "freq"
#                 rr, cc = self._sample_freq_neighbor(i, j, H, self.rad)

#             # ---- Kanalmaskierung wie gehabt -------------------------------
#             if C == 4:
#                 if self.random_mask_low_rank and self.random_mask_noisy:
#                     real_sel = (np.random.rand() < 0.5)
#                     if real_sel:
#                         inp[0, i, j] = img[0, rr, cc]  # LR_real
#                         inp[2, i, j] = img[2, rr, cc]  # NO_real
#                         mask_noisy[0, i, j] = True
#                     else:
#                         inp[1, i, j] = img[1, rr, cc]  # LR_imag
#                         inp[3, i, j] = img[3, rr, cc]  # NO_imag
#                         mask_noisy[1, i, j] = True

#                 elif self.random_mask_noisy and not self.random_mask_low_rank:
#                     nz_ch_rel = np.random.randint(2)  # 0=real,1=imag
#                     inp[2 + nz_ch_rel, i, j] = img[2 + nz_ch_rel, rr, cc]
#                     mask_noisy[nz_ch_rel, i, j] = True
#                     inp[0, i, j] = img[0, rr, cc]
#                     inp[1, i, j] = img[1, rr, cc]

#                 elif self.random_mask_low_rank and not self.random_mask_noisy:
#                     lr_ch = np.random.randint(2)
#                     inp[lr_ch, i, j] = img[lr_ch, rr, cc]
#                     inp[2, i, j] = img[2, rr, cc]
#                     inp[3, i, j] = img[3, rr, cc]
#                     mask_noisy[:, i, j] = True

#                 else:
#                     inp[:, i, j] = img[:, rr, cc]
#                     mask_noisy[:, i, j] = True

#             else:  # C == 2
#                 if self.random_mask_noisy:
#                     ch = np.random.randint(2)
#                     inp[ch, i, j] = img[ch, rr, cc]
#                     mask_noisy[ch, i, j] = True
#                 else:
#                     inp[:, i, j] = img[:, rr, cc]
#                     mask_noisy[:, i, j] = True

#         return inp, tgt, mask_noisy

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












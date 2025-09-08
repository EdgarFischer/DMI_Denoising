# data/transforms_3d.py
import numpy as np
from typing import List, Tuple


class StratifiedPixelSelection3D:
    """
    Noise2Void-ähnliche Maskierung für 3D-Daten (C, Z, F, T).

    • Masken werden in der (F,T)-Ebene erzeugt (wie 2D)  
    • Die ausgewählten (f,T)-Koordinaten werden auf **alle z** kopiert
    • Nachbar-Pixel (swap) ist in jeder z-Schicht derselbe Offset
    • Kanal-Logik identisch zu deiner 2D-Version
    """

    # ---------- Helper aus der 2D-Klasse -------------------
    @staticmethod
    def _sample_other_pixel(i, j, i0, i1, j0, j1):
        while True:
            rr = np.random.randint(i0, i1)
            cc = np.random.randint(j0, j1)
            if rr != i or cc != j:
                return rr, cc

    @staticmethod
    def _sample_time_neighbor(i, j, W, rad):
        j0, j1 = max(j - rad, 0), min(j + rad + 1, W)
        while True:
            cc = np.random.randint(j0, j1)
            if cc != j:
                return i, cc

    @staticmethod
    def _sample_freq_neighbor(i, j, H, rad):
        i0, i1 = max(i - rad, 0), min(i + rad + 1, H)
        while True:
            rr = np.random.randint(i0, i1)
            if rr != i:
                return rr, j

    # ---------- Konstruktor --------------------------------
    def __init__(
        self,
        num_masked_pixels: int = 64,
        window_size: int = 3,
        random_mask_low_rank: bool = False,
        random_mask_noisy: bool = True,
        swap_mode: str = "both",
    ):
        assert window_size % 2 == 1 and window_size >= 3
        assert swap_mode in {"both", "time", "freq"}

        self.N   = int(num_masked_pixels)
        self.win = int(window_size)
        self.rad = self.win // 2

        self.random_mask_low_rank = bool(random_mask_low_rank)
        self.random_mask_noisy    = bool(random_mask_noisy)
        self.swap_mode            = swap_mode

    # ---------- Hauptaufruf --------------------------------
    def __call__(self, img: np.ndarray):
        """
        Eingabe:  img (C, Z, F, T)  – Komplex-Kanäle bereits separat (C=2 oder 4)
        Rückgabe: inp, tgt, mask_noisy  (Shapes identisch zu img)
        """
        if img.ndim != 4:
            raise ValueError(f"Erwarte 4D‐Array (C,Z,F,T), bekam {img.shape}")

        C, Z, F, T = img.shape
        tgt  = img.copy()
        inp  = img.copy()
        mask_noisy = np.zeros((C if C == 2 else 2, Z, F, T), dtype=bool)  # nur 2 Maskenkanäle

        # --------- (f,T)-Koordinaten stratifiziert ziehen ----------
        n_rows = max(1, int(np.sqrt(self.N * F / T)))
        n_cols = int(np.ceil(self.N / n_rows))
        tile_h = F / n_rows
        tile_w = T / n_cols

        coords: List[Tuple[int, int]] = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(coords) >= self.N:
                    break
                y0, y1 = int(i * tile_h), min(int((i + 1) * tile_h), F)
                x0, x1 = int(j * tile_w), min(int((j + 1) * tile_w), T)
                if y1 > y0 and x1 > x0:
                    coords.append((
                        np.random.randint(y0, y1),
                        np.random.randint(x0, x1)
                    ))
            if len(coords) >= self.N:
                break

        # Fallback, falls Stratifikation nicht genug Pixel lieferte
        if len(coords) < self.N:
            all_coords = [(ii, jj) for ii in range(F) for jj in range(T)]
            np.random.shuffle(all_coords)
            coords += all_coords[: self.N - len(coords)]

        # ------------- Maskieren & Tauschen ------------------------
        for (i_f, j_t) in coords:

            # Nachbar im (f,T)‐Fenster bestimmen (global, für alle Z)
            if self.swap_mode == "both":
                i0, i1 = max(i_f - self.rad, 0), min(i_f + self.rad + 1, F)
                j0, j1 = max(j_t - self.rad, 0), min(j_t + self.rad + 1, T)
                rr, cc = self._sample_other_pixel(i_f, j_t, i0, i1, j0, j1)
            elif self.swap_mode == "time":
                rr, cc = self._sample_time_neighbor(i_f, j_t, T, self.rad)
            else:  # "freq"
                rr, cc = self._sample_freq_neighbor(i_f, j_t, F, self.rad)

            # ---- für **alle Z‐Schichten** denselben Offset anwenden ----
            for z in range(Z):

                if C == 4:
                    if self.random_mask_low_rank and self.random_mask_noisy:
                        real_sel = (np.random.rand() < 0.5)
                        if real_sel:
                            inp[0, z, i_f, j_t] = img[0, z, rr, cc]
                            inp[2, z, i_f, j_t] = img[2, z, rr, cc]
                            mask_noisy[0, z, i_f, j_t] = True
                        else:
                            inp[1, z, i_f, j_t] = img[1, z, rr, cc]
                            inp[3, z, i_f, j_t] = img[3, z, rr, cc]
                            mask_noisy[1, z, i_f, j_t] = True

                    elif self.random_mask_noisy and not self.random_mask_low_rank:
                        nz_ch_rel = np.random.randint(2)
                        inp[2 + nz_ch_rel, z, i_f, j_t] = img[2 + nz_ch_rel, z, rr, cc]
                        mask_noisy[nz_ch_rel, z, i_f, j_t] = True
                        inp[0, z, i_f, j_t] = img[0, z, rr, cc]
                        inp[1, z, i_f, j_t] = img[1, z, rr, cc]

                    elif self.random_mask_low_rank and not self.random_mask_noisy:
                        lr_ch = np.random.randint(2)
                        inp[lr_ch, z, i_f, j_t] = img[lr_ch, z, rr, cc]
                        inp[2, z, i_f, j_t] = img[2, z, rr, cc]
                        inp[3, z, i_f, j_t] = img[3, z, rr, cc]
                        mask_noisy[:, z, i_f, j_t] = True
                    else:
                        inp[:, z, i_f, j_t] = img[:, z, rr, cc]
                        mask_noisy[:, z, i_f, j_t] = True

                else:  # C == 2
                    if self.random_mask_noisy:
                        ch = np.random.randint(2)
                        inp[ch, z, i_f, j_t] = img[ch, z, rr, cc]
                        mask_noisy[ch, z, i_f, j_t] = True
                    else:
                        inp[:, z, i_f, j_t] = img[:, z, rr, cc]
                        mask_noisy[:, z, i_f, j_t] = True

        return inp, tgt, mask_noisy
    
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












# src/denoising/data/transforms.py
import numpy as np


class StratifiedPixelSelection2D:
    """
    Noise2Void-ähnliche Maskierung für 2D-Daten im (F,T)-Fenster.
    Erwartet img shape (C,F,T) mit C==2: [real, imag].

    Parameter:
      num_masked_pixels: Anzahl maskierter Pixel (F,T-Koordinaten)
      window_size: ungerade >=3, Nachbarschaftsfenster
      swap_mode: {"both","time","freq"}
        - both: 2D-Fenster
        - time: nur entlang T (Zeit)
        - freq: nur entlang F (Frequenz)
      random_mask_noisy:
        - True: nur einen Kanal (real ODER imag) ersetzen
        - False: beide Kanäle ersetzen
    """

    def __init__(
        self,
        num_masked_pixels: int = 64,
        window_size: int = 3,
        random_mask_noisy: bool = True,
        swap_mode: str = "both",
    ):
        assert window_size % 2 == 1, "window_size muss ungerade sein"
        assert window_size >= 3, "window_size muss ≥3 sein"
        assert swap_mode in {"both", "time", "freq"}, "swap_mode muss 'both', 'time' oder 'freq' sein"

        self.N = int(num_masked_pixels)
        self.win = int(window_size)
        self.rad = self.win // 2
        self.random_mask_noisy = bool(random_mask_noisy)
        self.swap_mode = swap_mode

    @staticmethod
    def _sample_other_pixel(i, j, i0, i1, j0, j1):
        area = (i1 - i0) * (j1 - j0)
        if area <= 1:
            raise ValueError("Fenster enthält nur den Zentrumspixel.")
        while True:
            rr = np.random.randint(i0, i1)
            cc = np.random.randint(j0, j1)
            if rr != i or cc != j:
                return rr, cc

    @staticmethod
    def _sample_time_neighbor(i, j, T, rad):
        j0, j1 = max(j - rad, 0), min(j + rad + 1, T)
        if (j1 - j0) <= 1:
            return i, j
        while True:
            cc = np.random.randint(j0, j1)
            if cc != j:
                return i, cc

    @staticmethod
    def _sample_freq_neighbor(i, j, F, rad):
        i0, i1 = max(i - rad, 0), min(i + rad + 1, F)
        if (i1 - i0) <= 1:
            return i, j
        while True:
            rr = np.random.randint(i0, i1)
            if rr != i:
                return rr, j

    def __call__(self, img: np.ndarray):
        if img.ndim != 3:
            raise ValueError(f"Erwarte (C,F,T), bekam {img.shape}.")
        C, F, T = img.shape
        if C != 2:
            raise ValueError(f"Erwarte C==2 (real,imag), bekam C={C}.")

        tgt = img.copy()
        inp = img.copy()
        mask_noisy = np.zeros((2, F, T), dtype=bool)  # 2 Masken: real/imag

        # ---- Stratified Auswahl über (F,T) ----
        n_rows = max(1, int(np.sqrt(self.N * F / max(T, 1))))
        n_cols = int(np.ceil(self.N / n_rows))
        tile_h = F / n_rows
        tile_w = T / n_cols

        coords = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(coords) >= self.N:
                    break
                f0, f1 = int(i * tile_h), min(int((i + 1) * tile_h), F)
                t0, t1 = int(j * tile_w), min(int((j + 1) * tile_w), T)
                if f1 > f0 and t1 > t0:
                    coords.append((np.random.randint(f0, f1), np.random.randint(t0, t1)))
            if len(coords) >= self.N:
                break

        # Fallback fill
        if len(coords) < self.N:
            all_coords = [(ii, jj) for ii in range(F) for jj in range(T)]
            np.random.shuffle(all_coords)
            for (ii, jj) in all_coords:
                if len(coords) >= self.N:
                    break
                if (ii, jj) not in coords:
                    coords.append((ii, jj))

        # ---- Maskieren ----
        for (i, j) in coords[: self.N]:
            if self.swap_mode == "both":
                i0, i1 = max(i - self.rad, 0), min(i + self.rad + 1, F)
                j0, j1 = max(j - self.rad, 0), min(j + self.rad + 1, T)
                rr, cc = self._sample_other_pixel(i, j, i0, i1, j0, j1)
            elif self.swap_mode == "time":
                rr, cc = self._sample_time_neighbor(i, j, T, self.rad)
            else:  # "freq"
                rr, cc = self._sample_freq_neighbor(i, j, F, self.rad)

            if self.random_mask_noisy:
                ch = np.random.randint(2)  # 0=real, 1=imag
                inp[ch, i, j] = img[ch, rr, cc]
                mask_noisy[ch, i, j] = True
            else:
                inp[:, i, j] = img[:, rr, cc]
                mask_noisy[:, i, j] = True

        return inp, tgt, mask_noisy


class StratifiedPixelSelectionTime1D:
    """
    1D-Zeitmaske:
    - Eingabe: img shape (C,F,T) mit C==2.
    - Es werden N Zeitpunkte j_t gewählt (stratifiziert über T).
    - Pro j_t wird ein Nachbar cc im Zeitfenster gewählt.
    - Swap/Maske werden für ALLE F bei diesem j_t angewandt (Replikation über F).
    """

    @staticmethod
    def _sample_time_neighbor(j, T, rad):
        j0, j1 = max(j - rad, 0), min(j + rad + 1, T)
        if (j1 - j0) <= 1:
            # kein anderer Zeitindex verfügbar
            return j
        while True:
            cc = np.random.randint(j0, j1)
            if cc != j:
                return cc

    def __init__(
        self,
        num_masked_pixels: int = 8,
        window_size: int = 3,
        random_mask_noisy: bool = False,
    ):
        assert window_size % 2 == 1 and window_size >= 3
        self.N = int(num_masked_pixels)
        self.win = int(window_size)
        self.rad = self.win // 2
        self.random_mask_noisy = bool(random_mask_noisy)

    def __call__(self, img: np.ndarray):
        if img.ndim != 3:
            raise ValueError(f"Erwarte (C,F,T), bekam {img.shape}.")
        C, F, T = img.shape
        if C != 2:
            raise ValueError(f"Erwarte C==2 (real,imag), bekam C={C}.")

        tgt = img.copy()
        inp = img.copy()
        mask_noisy = np.zeros((2, F, T), dtype=bool)

        # ---- stratifizierte Zeitpunkte wählen ----
        if self.N <= 0:
            return inp, tgt, mask_noisy

        tile = max(1, T // self.N)
        coords = []
        for start in range(0, T, tile):
            if len(coords) >= self.N:
                break
            end = min(start + tile, T)
            coords.append(np.random.randint(start, end))
        while len(coords) < self.N:
            coords.append(np.random.randint(0, T))

        # ---- maskieren und replizieren über F ----
        for j_t in coords[: self.N]:
            cc = self._sample_time_neighbor(j_t, T, self.rad)

            if self.random_mask_noisy:
                ch = np.random.randint(2)  # 0=real, 1=imag
                inp[ch, :, j_t] = img[ch, :, cc]
                mask_noisy[ch, :, j_t] = True
            else:
                inp[:, :, j_t] = img[:, :, cc]
                mask_noisy[:, :, j_t] = True

        return inp, tgt, mask_noisy
# data/transforms.py
import numpy as np

class UniformPixelSelection:
    def __init__(self, num_masked_pixels: int = 64, window_size: int = 3):
        """
        num_masked_pixels: wie viele Pixel pro Bild gemas­kert werden
        window_size: Größe des lokalen Nachbar­fensters (z.B. 3 → 3×3)
        """
        assert window_size % 2 == 1, "window_size muss ungerade sein"
        self.N = num_masked_pixels
        self.win = window_size
        self.rad = window_size // 2

    def __call__(self, img: np.ndarray):
        """
        img: 2D float32-Array, shape (H, W)
        Returns:
          inp: maskiertes Bild
          tgt: Original-Bild
          mask: Boolean-Maske (True = gemasktes Pixel)
        """
        H, W = img.shape
        tgt = img.copy()
        inp = img.copy()
        mask = np.zeros((H, W), dtype=bool)

        # 1) zufällig N Pixel auswählen
        coords = np.random.choice(H * W, size=self.N, replace=False)
        for idx in coords:
            i, j = divmod(int(idx), W)

            # 2) Fenster-Koordinaten begrenzen
            i0 = max(i - self.rad, 0)
            i1 = min(i + self.rad + 1, H)
            j0 = max(j - self.rad, 0)
            j1 = min(j + self.rad + 1, W)

            # 3) Uniform aus Nachbar-Fenster auswählen
            rr = np.random.randint(i0, i1)
            cc = np.random.randint(j0, j1)

            inp[i, j] = img[rr, cc]
            mask[i, j] = True

        return inp, tgt, mask
    
# data/transforms.py

import numpy as np

class StratifiedPixelSelection:
    """
    Wählt N Pixel möglichst gleichmäßig über das Bild verteilt,
    ersetzt jeden durch einen zufälligen Nachbarwert (≠ eigener Wert)
    und liefert Input, Target und Maske für Noise-2-Void-Training.
    """

    def __init__(self, num_masked_pixels: int = 64, window_size: int = 3):
        assert window_size % 2 == 1 and window_size > 1, \
            "window_size muss ungerade und >1 sein"
        self.N   = num_masked_pixels
        self.win = window_size
        self.rad = window_size // 2     # Radius

    def __call__(self, img: np.ndarray):
        """
        Parameters
        ----------
        img : np.ndarray, shape (C, H, W)
            z. B. C=2 für Real+Imag

        Returns
        -------
        inp  : np.ndarray, (C, H, W)  – maskiertes Bild
        tgt  : np.ndarray, (C, H, W)  – Originalbild
        mask : np.ndarray, (H, W) bool – True an maskierten Pixeln
        """
        C, H, W = img.shape
        tgt  = img.copy()
        inp  = img.copy()
        mask = np.zeros((H, W), dtype=bool)

        # ─ 1) Bild in n_rows × n_cols Zellen unterteilen ─
        n_rows = max(1, int(np.sqrt(self.N * H / W)))
        n_cols = int(np.ceil(self.N / n_rows))
        tile_h = H / n_rows
        tile_w = W / n_cols

        coords = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(coords) >= self.N:
                    break
                y0, y1 = int(i * tile_h), min(int((i + 1) * tile_h), H)
                x0, x1 = int(j * tile_w), min(int((j + 1) * tile_w), W)
                if y1 > y0 and x1 > x0:
                    rr = np.random.randint(y0, y1)
                    cc = np.random.randint(x0, x1)
                    coords.append((rr, cc))
            if len(coords) >= self.N:
                break

        # ─ 2) ggf. zufällig auffüllen bis N erreicht ─
        if len(coords) < self.N:
            all_coords = [(i, j) for i in range(H) for j in range(W)]
            np.random.shuffle(all_coords)
            coords.extend(all_coords[: self.N - len(coords)])

        # ─ 3) Maskieren ─
        for (i, j) in coords[: self.N]:
            i0, i1 = max(i - self.rad, 0), min(i + self.rad + 1, H)
            j0, j1 = max(j - self.rad, 0), min(j + self.rad + 1, W)

            # ⇒ Zufälliger Nachbar, aber nie das Pixel selbst
            while True:
                rr = np.random.randint(i0, i1)
                cc = np.random.randint(j0, j1)
                if rr != i or cc != j:        # Bedingung erfüllt → raus
                    break

            inp[:, i, j] = img[:, rr, cc]     # Ersatzwert setzen
            mask[i, j]   = True

        return inp.astype(np.float32), tgt.astype(np.float32), mask






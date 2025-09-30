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
    def __init__(self, num_masked_pixels: int = 64, window_size: int = 3):
        assert window_size % 2 == 1, "window_size muss ungerade sein"
        self.N   = num_masked_pixels
        self.win = window_size
        self.rad = window_size // 2

    def __call__(self, img: np.ndarray):
        """
        img: np.ndarray mit Shape (C, H, W) – z.B. C=2 für Real+Imag
        returns:
          inp: maskiertes Bild, Shape (C, H, W)
          tgt: Original-Bild,   Shape (C, H, W)
          mask: Boolean-Maske,  Shape (H, W)
        """
        C, H, W = img.shape
        tgt  = img.copy()
        inp  = img.copy()
        mask = np.zeros((H, W), dtype=bool)

        # 1) Anzahl Zeilen/Spalten von Zellen bestimmen
        n_rows = max(1, int(np.sqrt(self.N * H / W)))
        n_cols = int(np.ceil(self.N / n_rows))
        tile_h = H / n_rows
        tile_w = W / n_cols

        coords = []
        # 2) Pro Zelle genau einen Pixel wählen
        for i in range(n_rows):
            for j in range(n_cols):
                if len(coords) >= self.N:
                    break
                y0, y1 = int(i * tile_h), min(int((i+1)*tile_h), H)
                x0, x1 = int(j * tile_w), min(int((j+1)*tile_w), W)
                if y1 > y0 and x1 > x0:
                    rr = np.random.randint(y0, y1)
                    cc = np.random.randint(x0, x1)
                    coords.append((rr, cc))
            if len(coords) >= self.N:
                break

        # 3) Bei Bedarf zufällig auffüllen
        if len(coords) < self.N:
            all_coords = [(i, j) for i in range(H) for j in range(W)]
            np.random.shuffle(all_coords)
            for (ii, jj) in all_coords:
                if len(coords) >= self.N:
                    break
                if (ii, jj) not in coords:
                    coords.append((ii, jj))

        # 4) Maskierung mit lokalem Fenster, auf alle Kanäle anwenden
        for (i, j) in coords[:self.N]:
            i0, i1 = max(i - self.rad, 0), min(i + self.rad + 1, H)
            j0, j1 = max(j - self.rad, 0), min(j + self.rad + 1, W)
            rr = np.random.randint(i0, i1)
            cc = np.random.randint(j0, j1)
            # ersetze in allen Kanälen
            for c in range(C):
                inp[c, i, j] = img[c, rr, cc]
            mask[i, j] = True

        return inp, tgt, mask





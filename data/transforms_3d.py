# data/transforms_3d.py
import numpy as np

class StratifiedVoxelSelection:
    r"""
    Stratifiziertes Noise2Void-Maskieren in 3-D.

    • Wählt genau `num_masked_voxels` Voxel so, dass das gesamte Volumen
      gleichmäßig abgedeckt wird (Raster-Sampling).
    • Ersetzt jeden maskierten Voxel durch einen zufällig gewählten Voxel
      aus einem kubischen Nachbarfenster der Kantenlänge `window_size`.
    • Funktioniert kanal-agnostisch, d. h. dieselbe Maske wird für alle C-Kanäle
      (Real + Imag) angewendet.

    Args
    ----
    num_masked_voxels : int
        Wie viele Voxel pro Volumen maskiert werden.
    window_size : int
        Kantenlänge des lokalen Ersatzfensters (muss ungerade sein).

    Returns
    -------
    inp  : np.ndarray,  shape (C, D, H, W) – maskiertes Volumen
    tgt  : np.ndarray,  shape (C, D, H, W) – Original-Volumen
    mask : np.ndarray,  shape (D, H, W)     – Bool-Maske (True = maskiert)
    """
    def __init__(self, num_masked_voxels: int = 256, window_size: int = 3):
        assert window_size % 2 == 1, "window_size muss ungerade sein"
        self.N   = num_masked_voxels
        self.win = window_size
        self.rad = window_size // 2

    # ------------------------------------------------------------------
    # Hilfsroutine: finde ganzzahlige Gitter-Teilungen n_d × n_h × n_w,
    # deren Produkt ≥ N ist (ein Voxel pro Zelle) und die in (D,H,W) passen.
    # ------------------------------------------------------------------
    @staticmethod
    def _grid_dimensions(D, H, W, N):
        g = int(np.ceil(N ** (1/3)))          # Start mit Würfelgitter
        n_d = min(D, g)
        n_h = min(H, g)
        n_w = min(W, g)

        # Schrittweise vergrößern, bis Produkt ≥ N
        while n_d * n_h * n_w < N:
            # Erhöhe jeweils die Achse mit dem größten restlichen Spielraum
            if n_d < D and (n_d <= n_h and n_d <= n_w):
                n_d += 1
            elif n_h < H and (n_h <= n_d and n_h <= n_w):
                n_h += 1
            elif n_w < W:
                n_w += 1
            else:          # Alles voll – sollte praktisch nie passieren
                break
        return n_d, n_h, n_w

    # ------------------------------------------------------------------
    def __call__(self, img: np.ndarray):
        # img erwartet (C, D, H, W)
        C, D, H, W = img.shape
        tgt  = img.copy()
        inp  = img.copy()
        mask = np.zeros((D, H, W), dtype=bool)

        # 1) Gittergröße bestimmen
        n_d, n_h, n_w = self._grid_dimensions(D, H, W, self.N)
        tile_d = D / n_d
        tile_h = H / n_h
        tile_w = W / n_w

        coords = []
        # 2) Genau einen Voxel pro Zelle wählen (stratifiziert)
        for zi in range(n_d):
            for yi in range(n_h):
                for xi in range(n_w):
                    if len(coords) >= self.N:
                        break
                    z0, z1 = int(zi * tile_d), min(int((zi + 1) * tile_d), D)
                    y0, y1 = int(yi * tile_h), min(int((yi + 1) * tile_h), H)
                    x0, x1 = int(xi * tile_w), min(int((xi + 1) * tile_w), W)
                    if z1 > z0 and y1 > y0 and x1 > x0:
                        zz = np.random.randint(z0, z1)
                        yy = np.random.randint(y0, y1)
                        xx = np.random.randint(x0, x1)
                        coords.append((zz, yy, xx))
            if len(coords) >= self.N:
                break

        # 3) Falls < N Koordinaten (Sehr kleine Volumina) → zufällig auffüllen
        if len(coords) < self.N:
            all_coords = [(z, y, x) for z in range(D)
                                     for y in range(H)
                                     for x in range(W)]
            np.random.shuffle(all_coords)
            for (zz, yy, xx) in all_coords:
                if len(coords) >= self.N:
                    break
                if (zz, yy, xx) not in coords:
                    coords.append((zz, yy, xx))

        # 4) Maskierung & Ersatz aus lokalem Fenster
        for (z, y, x) in coords[:self.N]:
            z0, z1 = max(z - self.rad, 0), min(z + self.rad + 1, D)
            y0, y1 = max(y - self.rad, 0), min(y + self.rad + 1, H)
            x0, x1 = max(x - self.rad, 0), min(x + self.rad + 1, W)

            zz = np.random.randint(z0, z1)
            yy = np.random.randint(y0, y1)
            xx = np.random.randint(x0, x1)

            # gleiche Ersetzung für alle Kanäle
            for c in range(C):
                inp[c, z, y, x] = img[c, zz, yy, xx]
            mask[z, y, x] = True

        return inp, tgt, mask





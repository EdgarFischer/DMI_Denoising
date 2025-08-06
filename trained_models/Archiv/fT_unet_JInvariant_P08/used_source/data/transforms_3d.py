# data/transforms_3d.py
import numpy as np

class StratifiedVoxelSelection:
    r"""
    Stratifiziertes Noise2Void-Maskieren in 3-D ohne Selbstersatz.

    • Genau N maskierte Voxel, gleichmäßig im Volumen verteilt.
    • Jeder maskierte Voxel wird durch einen anderen (!) Voxel
      aus einem kubischen Nachbarfenster ersetzt (Fensterkante = window_size).
    • Maske identisch für alle Kanäle (Real + Imag).

    Args
    ----
    num_masked_voxels : int
        Anzahl zu maskierender Voxel.
    window_size : int (ungerade)
        Kantenlänge des Ersatzfensters.
    """
    def __init__(self, num_masked_voxels: int = 256, window_size: int = 3):
        assert window_size % 2 == 1, "window_size muss ungerade sein"
        self.N   = num_masked_voxels
        self.win = window_size
        self.rad = window_size // 2

    # ------------------------------------------------------------------
    @staticmethod
    def _grid_dimensions(D, H, W, N):
        """Finde n_d × n_h × n_w ≥ N, die in (D,H,W) passen."""
        g = int(np.ceil(N ** (1/3)))
        n_d, n_h, n_w = min(D, g), min(H, g), min(W, g)
        while n_d * n_h * n_w < N:
            if n_d < D and (n_d <= n_h and n_d <= n_w):
                n_d += 1
            elif n_h < H and (n_h <= n_d and n_h <= n_w):
                n_h += 1
            elif n_w < W:
                n_w += 1
            else:
                break
        return n_d, n_h, n_w

    # ------------------------------------------------------------------
    def __call__(self, img: np.ndarray):
        # img: (C, D, H, W)
        C, D, H, W = img.shape
        tgt  = img.copy()
        inp  = img.copy()
        mask = np.zeros((D, H, W), dtype=bool)

        # 1) Stratifiziertes Raster
        n_d, n_h, n_w = self._grid_dimensions(D, H, W, self.N)
        tile_d, tile_h, tile_w = D / n_d, H / n_h, W / n_w

        coords = []
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

        # 2) Falls zu wenige Koordinaten → zufällig auffüllen
        if len(coords) < self.N:
            all_coords = [(z, y, x) for z in range(D)
                                       for y in range(H)
                                       for x in range(W)]
            np.random.shuffle(all_coords)
            coords.extend(all_coords[: self.N - len(coords)])

        # 3) Maskieren & Ersatz aus Fenster (immer ≠ Zentrum)
        for (z, y, x) in coords[: self.N]:
            z0, z1 = max(z - self.rad, 0), min(z + self.rad + 1, D)
            y0, y1 = max(y - self.rad, 0), min(y + self.rad + 1, H)
            x0, x1 = max(x - self.rad, 0), min(x + self.rad + 1, W)

            # Kandidaten außer Zentrum sammeln
            candidates = [
                (zz, yy, xx)
                for zz in range(z0, z1)
                for yy in range(y0, y1)
                for xx in range(x0, x1)
                if not (zz == z and yy == y and xx == x)
            ]
            # fallback, falls Fenster nur das Zentrum enthält (sehr kleiner Volumensrand)
            if len(candidates) == 0:
                continue

            zz, yy, xx = candidates[np.random.randint(len(candidates))]

            inp[:, z, y, x] = img[:, zz, yy, xx]
            mask[z, y, x]   = True

        return inp, tgt, mask






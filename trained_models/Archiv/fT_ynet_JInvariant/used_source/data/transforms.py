# data/transforms.py
import numpy as np


class StratifiedPixelSelection:
    """
    Noise2Void-ähnliche Maskierung für 2D-Daten.

    Erwartet Eingaben mit Kanälen:
      C=4: [LR_real, LR_imag, NO_real, NO_imag]
      C=2: [real, imag] (einzelner Branch; wird wie 'noisy' behandelt)

    Parameter:
      random_mask_low_rank : bool
          True  -> pro Maskenpixel genau EIN LR-Kanal (real/imag) maskieren.
          False -> beide LR-Kanäle maskieren.
      random_mask_noisy : bool
          True  -> pro Maskenpixel genau EIN Noisy-Kanal (real/imag) maskieren.
          False -> beide Noisy-Kanäle maskieren.

      Falls BEIDE True → konsistente Wahl: entweder real ODER imag in beiden Branches.

      WICHTIG: Der zu ersetzende Pixel wird NIE durch sich selbst ersetzt.
    """
    def __init__(
        self,
        num_masked_pixels: int = 64,
        window_size: int = 3,
        random_mask_low_rank: bool = False,
        random_mask_noisy: bool = True,
    ):
        assert window_size % 2 == 1, "window_size muss ungerade sein"
        assert window_size >= 3, "window_size muss ≥3 sein, sonst gibt es keine Nachbarn"
        self.N   = int(num_masked_pixels)
        self.win = int(window_size)
        self.rad = self.win // 2

        self.random_mask_low_rank = bool(random_mask_low_rank)
        self.random_mask_noisy    = bool(random_mask_noisy)

    @staticmethod
    def _sample_other_pixel(i, j, i0, i1, j0, j1):
        """Ziehe zufällig einen Pixel im Fenster, aber NICHT (i, j)."""
        area = (i1 - i0) * (j1 - j0)
        if area <= 1:
            raise ValueError(
                "Das Fenster enthält nur den Zentrumspixel. "
                "Erhöhe window_size, damit ein anderer Pixel gewählt werden kann."
            )
        while True:
            rr = np.random.randint(i0, i1)
            cc = np.random.randint(j0, j1)
            if rr != i or cc != j:
                return rr, cc

    def __call__(self, img: np.ndarray):
        """
        img: np.ndarray, Shape (C,H,W).
        Returns:
            inp  : maskiertes Bild (C,H,W)
            tgt  : Original-Bild   (C,H,W)
            mask : Bool-Maske für *Noisy*-Kanäle, Shape
                   (2,H,W) falls C==4,
                   (2,H,W) falls C==2 (einzelner Branch; beide Kanäle als "noisy")
        """
        if img.ndim != 3:
            raise ValueError(f"Erwarte 3D-Array (C,H,W), bekam {img.shape}.")

        C, H, W = img.shape
        tgt  = img.copy()
        inp  = img.copy()

        if C == 4:
            mask_noisy = np.zeros((2, H, W), dtype=bool)  # nur Noisy zählt für Loss
        elif C == 2:
            mask_noisy = np.zeros((2, H, W), dtype=bool)  # beide Kanäle = "noisy"
        else:
            raise ValueError(f"Unterstütze nur C=2 oder C=4, bekam C={C}.")

        # --- Stratified Koordinaten bestimmen --------------------------------
        n_rows = max(1, int(np.sqrt(self.N * H / W)))
        n_cols = int(np.ceil(self.N / n_rows))
        tile_h = H / n_rows
        tile_w = W / n_cols

        coords = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(coords) >= self.N:
                    break
                y0, y1 = int(i * tile_h), min(int((i+1)*tile_h), H)
                x0, x1 = int(j * tile_w), min(int((j+1)*tile_w), W)
                if y1 > y0 and x1 > x0:
                    coords.append((np.random.randint(y0, y1),
                                   np.random.randint(x0, x1)))
            if len(coords) >= self.N:
                break

        if len(coords) < self.N:
            all_coords = [(ii, jj) for ii in range(H) for jj in range(W)]
            np.random.shuffle(all_coords)
            for (ii, jj) in all_coords:
                if len(coords) >= self.N:
                    break
                if (ii, jj) not in coords:
                    coords.append((ii, jj))

        # --- Maskieren -------------------------------------------------------
        for (i, j) in coords[:self.N]:
            i0, i1 = max(i - self.rad, 0), min(i + self.rad + 1, H)
            j0, j1 = max(j - self.rad, 0), min(j + self.rad + 1, W)

            # NEU: Exkludiere (i, j)
            rr, cc = self._sample_other_pixel(i, j, i0, i1, j0, j1)

            if C == 4:
                # --- Y-Net Fall ------------------------------------------------
                if self.random_mask_low_rank and self.random_mask_noisy:
                    # SYNC: gleiche Wahl für beide Branches
                    real_sel = (np.random.rand() < 0.5)
                    if real_sel:
                        inp[0, i, j] = img[0, rr, cc]  # LR_real
                        inp[2, i, j] = img[2, rr, cc]  # NO_real
                        mask_noisy[0, i, j] = True
                    else:
                        inp[1, i, j] = img[1, rr, cc]  # LR_imag
                        inp[3, i, j] = img[3, rr, cc]  # NO_imag
                        mask_noisy[1, i, j] = True

                elif self.random_mask_noisy and not self.random_mask_low_rank:
                    # Noisy: zufälliger Kanal; Low-Rank: BEIDE maskieren
                    nz_ch_rel = np.random.randint(2)  # 0=real,1=imag
                    inp[2 + nz_ch_rel, i, j] = img[2 + nz_ch_rel, rr, cc]
                    mask_noisy[nz_ch_rel, i, j] = True
                    # LR beide
                    inp[0, i, j] = img[0, rr, cc]
                    inp[1, i, j] = img[1, rr, cc]

                elif self.random_mask_low_rank and not self.random_mask_noisy:
                    # Low-Rank: zufällig; Noisy: BEIDE maskieren
                    lr_ch = np.random.randint(2)
                    inp[lr_ch, i, j] = img[lr_ch, rr, cc]
                    inp[2, i, j] = img[2, rr, cc]
                    inp[3, i, j] = img[3, rr, cc]
                    mask_noisy[:, i, j] = True

                else:
                    # Beide False -> alle maskieren
                    inp[:, i, j] = img[:, rr, cc]
                    mask_noisy[:, i, j] = True

            else:
                # --- C==2: Einzel-Branch (als noisy interpretiert) -------------
                if self.random_mask_noisy:
                    ch = np.random.randint(2)
                    inp[ch, i, j] = img[ch, rr, cc]
                    mask_noisy[ch, i, j] = True
                else:
                    inp[:, i, j] = img[:, rr, cc]
                    mask_noisy[:, i, j] = True

        return inp, tgt, mask_noisy










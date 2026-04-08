import numpy as np


class StratifiedAxisMasking:
    """
    Flexible stratified masking over 1, 2, or 3 selected LOCAL axes.

    Expected input:
        img shape = (2, ...)

    where:
        axis 0 = [real, imag]
        axes 1.. = structured dimensions seen by the transform

    The Dataset passes masked_axes_local, which are LOCAL axes in img,
    including the leading real/imag axis convention:
        - axis 0 is reserved for real/imag and must NOT be masked
        - allowed masked axes are therefore >= 1

    Behavior:
        - if len(masked_axes_local) == 1:
              build a 1D stratified mask along that axis
              and replicate to all other axes
        - if len(masked_axes_local) == 2:
              build a 2D stratified mask on those two axes
              and replicate to all other axes
        - if len(masked_axes_local) == 3:
              build a true 3D stratified mask on those three axes
              and replicate to all other axes (if any remain)
    """

    def __init__(
        self,
        num_masked_pixels: int = 8,
        window_size: int = 3,
        random_mask_noisy: bool = False,
    ):
        if window_size % 2 != 1 or window_size < 3:
            raise ValueError("window_size must be odd and >= 3")

        self.N = int(num_masked_pixels)
        self.win = int(window_size)
        self.rad = self.win // 2
        self.random_mask_noisy = bool(random_mask_noisy)

    @staticmethod
    def _sample_other_index_1d(i, size, rad):
        i0, i1 = max(i - rad, 0), min(i + rad + 1, size)
        if (i1 - i0) <= 1:
            return i
        while True:
            ii = np.random.randint(i0, i1)
            if ii != i:
                return ii

    @staticmethod
    def _sample_other_index_2d(i, j, size0, size1, rad):
        i0, i1 = max(i - rad, 0), min(i + rad + 1, size0)
        j0, j1 = max(j - rad, 0), min(j + rad + 1, size1)
        area = (i1 - i0) * (j1 - j0)
        if area <= 1:
            return i, j
        while True:
            ii = np.random.randint(i0, i1)
            jj = np.random.randint(j0, j1)
            if ii != i or jj != j:
                return ii, jj

    @staticmethod
    def _sample_other_index_3d(i, j, k, size0, size1, size2, rad):
        i0, i1 = max(i - rad, 0), min(i + rad + 1, size0)
        j0, j1 = max(j - rad, 0), min(j + rad + 1, size1)
        k0, k1 = max(k - rad, 0), min(k + rad + 1, size2)

        vol = (i1 - i0) * (j1 - j0) * (k1 - k0)
        if vol <= 1:
            return i, j, k

        while True:
            ii = np.random.randint(i0, i1)
            jj = np.random.randint(j0, j1)
            kk = np.random.randint(k0, k1)
            if ii != i or jj != j or kk != k:
                return ii, jj, kk

    @staticmethod
    def _broadcast_mask(mask_base, full_shape, masked_axes_local):
        """
        mask_base shape corresponds only to masked axes.
        Broadcast to full structured image shape: (2, ...)
        """
        reshape_shape = [1] * len(full_shape)
        for k, ax in enumerate(masked_axes_local):
            reshape_shape[ax] = mask_base.shape[k]

        mask_view = mask_base.reshape(reshape_shape)
        full_mask = np.broadcast_to(mask_view, full_shape).copy()
        return full_mask

    def _make_stratified_coords_1d(self, size):
        if self.N <= 0:
            return []

        tile = max(1, size // self.N)
        coords = []

        for start in range(0, size, tile):
            if len(coords) >= self.N:
                break
            end = min(start + tile, size)
            coords.append(np.random.randint(start, end))

        while len(coords) < self.N:
            coords.append(np.random.randint(0, size))

        return coords[:self.N]

    def _make_stratified_coords_2d(self, size0, size1):
        if self.N <= 0:
            return []

        n_rows = max(1, int(np.sqrt(self.N * size0 / max(size1, 1))))
        n_cols = int(np.ceil(self.N / n_rows))

        tile_h = size0 / n_rows
        tile_w = size1 / n_cols

        coords = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(coords) >= self.N:
                    break
                a0, a1 = int(i * tile_h), min(int((i + 1) * tile_h), size0)
                b0, b1 = int(j * tile_w), min(int((j + 1) * tile_w), size1)
                if a1 > a0 and b1 > b0:
                    coords.append((np.random.randint(a0, a1), np.random.randint(b0, b1)))
            if len(coords) >= self.N:
                break

        if len(coords) < self.N:
            all_coords = [(i, j) for i in range(size0) for j in range(size1)]
            np.random.shuffle(all_coords)
            for ij in all_coords:
                if len(coords) >= self.N:
                    break
                if ij not in coords:
                    coords.append(ij)

        return coords[:self.N]

    def _make_stratified_coords_3d(self, size0, size1, size2):
        if self.N <= 0:
            return []

        # Choose grid counts so that n0 * n1 * n2 ~ N
        scale = (self.N / max(size0 * size1 * size2, 1)) ** (1 / 3)
        n0 = max(1, int(round(size0 * scale)))
        n1 = max(1, int(round(size1 * scale)))
        n2 = max(1, int(np.ceil(self.N / max(n0 * n1, 1))))

        # If product is still too small, grow n2 first, then n1, then n0
        while n0 * n1 * n2 < self.N:
            if size2 >= size1 and size2 >= size0:
                n2 += 1
            elif size1 >= size0:
                n1 += 1
            else:
                n0 += 1

        tile0 = size0 / n0
        tile1 = size1 / n1
        tile2 = size2 / n2

        coords = []
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    if len(coords) >= self.N:
                        break

                    a0, a1 = int(i * tile0), min(int((i + 1) * tile0), size0)
                    b0, b1 = int(j * tile1), min(int((j + 1) * tile1), size1)
                    c0, c1 = int(k * tile2), min(int((k + 1) * tile2), size2)

                    if a1 > a0 and b1 > b0 and c1 > c0:
                        coords.append((
                            np.random.randint(a0, a1),
                            np.random.randint(b0, b1),
                            np.random.randint(c0, c1),
                        ))
                if len(coords) >= self.N:
                    break
            if len(coords) >= self.N:
                break

        if len(coords) < self.N:
            all_coords = [(i, j, k) for i in range(size0) for j in range(size1) for k in range(size2)]
            np.random.shuffle(all_coords)
            used = set(coords)
            for ijk in all_coords:
                if len(coords) >= self.N:
                    break
                if ijk not in used:
                    coords.append(ijk)
                    used.add(ijk)

        return coords[:self.N]

    def __call__(self, img: np.ndarray, masked_axes_local):
        if img.ndim < 3:
            raise ValueError(f"Expected img with shape (2, ...), got {img.shape}")
        if img.shape[0] != 2:
            raise ValueError(f"Expected first axis to be real/imag with size 2, got {img.shape[0]}")

        masked_axes_local = tuple(masked_axes_local)

        if len(masked_axes_local) not in (1, 2, 3):
            raise ValueError(
                f"masked_axes_local must contain 1, 2, or 3 axes, got {masked_axes_local}"
            )

        if any(ax == 0 for ax in masked_axes_local):
            raise ValueError("Axis 0 is the real/imag axis and must not be masked.")

        if any(ax < 0 or ax >= img.ndim for ax in masked_axes_local):
            raise ValueError(
                f"masked_axes_local {masked_axes_local} invalid for img shape {img.shape}"
            )

        tgt = img.copy()
        inp = img.copy()
        full_shape = img.shape

        # ---- 1D masking ---------------------------------------------------
        if len(masked_axes_local) == 1:
            ax = masked_axes_local[0]
            size = img.shape[ax]

            base_mask = np.zeros((size,), dtype=bool)
            coords = self._make_stratified_coords_1d(size)

            for i in coords:
                ii = self._sample_other_index_1d(i, size, self.rad)

                src_slices = [slice(None)] * img.ndim
                dst_slices = [slice(None)] * img.ndim
                src_slices[ax] = ii
                dst_slices[ax] = i

                if self.random_mask_noisy:
                    ch = np.random.randint(2)
                    src_slices[0] = ch
                    dst_slices[0] = ch
                    inp[tuple(dst_slices)] = img[tuple(src_slices)]
                else:
                    inp[tuple(dst_slices)] = img[tuple(src_slices)]

                base_mask[i] = True

            mask = self._broadcast_mask(base_mask, full_shape, masked_axes_local)

            if self.random_mask_noisy:
                inp = img.copy()
                mask = np.zeros(full_shape, dtype=bool)
                for i in coords:
                    ii = self._sample_other_index_1d(i, size, self.rad)
                    ch = np.random.randint(2)

                    src_slices = [slice(None)] * img.ndim
                    dst_slices = [slice(None)] * img.ndim
                    src_slices[0] = ch
                    dst_slices[0] = ch
                    src_slices[ax] = ii
                    dst_slices[ax] = i

                    inp[tuple(dst_slices)] = img[tuple(src_slices)]
                    mask[tuple(dst_slices)] = True

            else:
                mask = np.broadcast_to(mask[None, ...] if mask.ndim == img.ndim - 1 else mask, full_shape).copy()

            return inp, tgt, mask

        # ---- 2D masking ---------------------------------------------------
        if len(masked_axes_local) == 2:
            ax0, ax1 = masked_axes_local
            size0 = img.shape[ax0]
            size1 = img.shape[ax1]

            base_mask = np.zeros((size0, size1), dtype=bool)
            coords = self._make_stratified_coords_2d(size0, size1)

            if self.random_mask_noisy:
                mask = np.zeros(full_shape, dtype=bool)
                for i, j in coords:
                    ii, jj = self._sample_other_index_2d(i, j, size0, size1, self.rad)
                    ch = np.random.randint(2)

                    src_slices = [slice(None)] * img.ndim
                    dst_slices = [slice(None)] * img.ndim

                    src_slices[0] = ch
                    dst_slices[0] = ch
                    src_slices[ax0] = ii
                    src_slices[ax1] = jj
                    dst_slices[ax0] = i
                    dst_slices[ax1] = j

                    inp[tuple(dst_slices)] = img[tuple(src_slices)]
                    mask[tuple(dst_slices)] = True

                return inp, tgt, mask

            for i, j in coords:
                ii, jj = self._sample_other_index_2d(i, j, size0, size1, self.rad)

                src_slices = [slice(None)] * img.ndim
                dst_slices = [slice(None)] * img.ndim
                src_slices[ax0] = ii
                src_slices[ax1] = jj
                dst_slices[ax0] = i
                dst_slices[ax1] = j

                inp[tuple(dst_slices)] = img[tuple(src_slices)]
                base_mask[i, j] = True

            mask = self._broadcast_mask(base_mask, full_shape, masked_axes_local)
            return inp, tgt, mask

        # ---- 3D masking ---------------------------------------------------
        ax0, ax1, ax2 = masked_axes_local
        size0 = img.shape[ax0]
        size1 = img.shape[ax1]
        size2 = img.shape[ax2]

        base_mask = np.zeros((size0, size1, size2), dtype=bool)
        coords = self._make_stratified_coords_3d(size0, size1, size2)

        if self.random_mask_noisy:
            mask = np.zeros(full_shape, dtype=bool)
            for i, j, k in coords:
                ii, jj, kk = self._sample_other_index_3d(i, j, k, size0, size1, size2, self.rad)
                ch = np.random.randint(2)

                src_slices = [slice(None)] * img.ndim
                dst_slices = [slice(None)] * img.ndim

                src_slices[0] = ch
                dst_slices[0] = ch
                src_slices[ax0] = ii
                src_slices[ax1] = jj
                src_slices[ax2] = kk
                dst_slices[ax0] = i
                dst_slices[ax1] = j
                dst_slices[ax2] = k

                inp[tuple(dst_slices)] = img[tuple(src_slices)]
                mask[tuple(dst_slices)] = True

            return inp, tgt, mask

        for i, j, k in coords:
            ii, jj, kk = self._sample_other_index_3d(i, j, k, size0, size1, size2, self.rad)

            src_slices = [slice(None)] * img.ndim
            dst_slices = [slice(None)] * img.ndim
            src_slices[ax0] = ii
            src_slices[ax1] = jj
            src_slices[ax2] = kk
            dst_slices[ax0] = i
            dst_slices[ax1] = j
            dst_slices[ax2] = k

            inp[tuple(dst_slices)] = img[tuple(src_slices)]
            base_mask[i, j, k] = True

        mask = self._broadcast_mask(base_mask, full_shape, masked_axes_local)
        return inp, tgt, mask
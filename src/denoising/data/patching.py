from typing import List, Tuple
import numpy as np

def get_random_patch_slices(img_shape, patch_sizes):
    """
    Build random valid slices for a network input tensor.

    Parameters
    ----------
    img_shape : tuple[int, ...]
        Shape of img, including real/imag axis at dimension 0.
        Example without channel axis: (2, X, Y)
        Example with channel axis:    (2, C, X, Y)

    patch_sizes : tuple[int | None, ...]
        Patch sizes for axes AFTER the real/imag axis, i.e. for img.shape[1:].
        - None means: use full axis
        - int N means: take a random patch of size N along that axis

    Returns
    -------
    tuple[slice, ...]
        Slice tuple with same length as img_shape.
        The first axis (real/imag) is always kept completely.
    """
    if len(patch_sizes) != len(img_shape) - 1:
        raise ValueError(
            f"patch_sizes must have length {len(img_shape) - 1}, "
            f"but got {len(patch_sizes)}."
        )

    slices = [slice(None)]  # keep real/imag axis completely

    for axis_len, patch_size in zip(img_shape[1:], patch_sizes):
        if patch_size is None:
            slices.append(slice(None))
        else:
            patch_size = int(patch_size)

            if patch_size <= 0:
                raise ValueError(f"Patch size must be positive or None, got {patch_size}.")

            if patch_size > axis_len:
                raise ValueError(
                    f"Patch size {patch_size} is larger than axis length {axis_len}."
                )

            if patch_size == axis_len:
                slices.append(slice(None))
            else:
                start = np.random.randint(0, axis_len - patch_size + 1)
                slices.append(slice(start, start + patch_size))

    return tuple(slices)

def extract_patch(img, patch_slices):
    return img[tuple(patch_slices)]

from itertools import product
from typing import List, Optional, Sequence, Tuple
import numpy as np


def _compute_patch_starts(
    axis_size: int,
    patch_size: int,
    stride: int,
) -> List[int]:
    """
    Compute start indices for one axis such that:
    - all patches are valid
    - the full axis is covered
    - the last patch is shifted if needed to touch the end exactly

    Example:
        axis_size=22, patch_size=8, stride=4
        -> [0, 4, 8, 12, 14]
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if patch_size > axis_size:
        raise ValueError(
            f"patch_size ({patch_size}) must not exceed axis size ({axis_size})"
        )

    starts = list(range(0, axis_size - patch_size + 1, stride))

    last_start = axis_size - patch_size
    if not starts:
        starts = [last_start]
    elif starts[-1] != last_start:
        starts.append(last_start)

    return starts


def generate_inference_patches(
    arr: np.ndarray,
    patch_sizes: Sequence[Optional[int]],
    strides: Sequence[Optional[int]],
    return_patches: bool = True,
) -> Tuple[List[Tuple[slice, ...]], Optional[List[np.ndarray]]]:
    """
    Generate all valid inference patches with full coverage.

    Parameters
    ----------
    arr : np.ndarray
        Input array, e.g. shape (C, X, Y, Z) or generally any Nd array.
    patch_sizes : sequence of int or None
        Patch size for each axis of arr.
        - None means: use the full axis length.
    strides : sequence of int or None
        Stride for each axis of arr.
        - None means: full-axis step (i.e. only one patch along that axis).
    return_patches : bool
        If True, also return the extracted patch arrays.
        If False, only return the slice tuples.

    Returns
    -------
    slices_list : list of tuple[slice, ...]
        Slice tuple for each patch.
    patches : list[np.ndarray] or None
        Extracted patches if return_patches=True, else None.

    Notes
    -----
    Guarantees that every voxel is contained in at least one patch,
    provided patch_size <= axis_size for every axis.
    """
    arr = np.asarray(arr)
    ndim = arr.ndim

    if len(patch_sizes) != ndim:
        raise ValueError(
            f"patch_sizes must have length {ndim}, got {len(patch_sizes)}"
        )
    if len(strides) != ndim:
        raise ValueError(
            f"strides must have length {ndim}, got {len(strides)}"
        )

    normalized_patch_sizes = []
    normalized_strides = []

    for axis, (axis_size, p, s) in enumerate(zip(arr.shape, patch_sizes, strides)):
        patch_size = axis_size if p is None else int(p)
        if patch_size > axis_size:
            raise ValueError(
                f"patch_sizes[{axis}]={patch_size} exceeds axis size {axis_size}"
            )

        stride = patch_size if s is None else int(s)

        normalized_patch_sizes.append(patch_size)
        normalized_strides.append(stride)

    axis_starts = [
        _compute_patch_starts(axis_size, patch_size, stride)
        for axis_size, patch_size, stride in zip(
            arr.shape, normalized_patch_sizes, normalized_strides
        )
    ]

    slices_list: List[Tuple[slice, ...]] = []
    patches: Optional[List[np.ndarray]] = [] if return_patches else None

    for starts in product(*axis_starts):
        slc = tuple(
            slice(start, start + patch_size)
            for start, patch_size in zip(starts, normalized_patch_sizes)
        )
        slices_list.append(slc)

        if return_patches:
            patches.append(arr[slc])

    return slices_list, patches

from typing import List, Tuple
import numpy as np


from typing import List, Tuple
import numpy as np


def _make_patch_weights(
    patch_shape: Tuple[int, ...],
    patch_sizes: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    weight_mode: str = "average",
    eps: float = 1e-6,
) -> np.ndarray:
    if weight_mode == "average":
        return np.ones(patch_shape, dtype=np.float32)

    if weight_mode == "hann":
        weights = np.ones(patch_shape, dtype=np.float32)

        for axis, (size, patch_size, axis_size) in enumerate(
            zip(patch_shape, patch_sizes, output_shape)
        ):
            # nur auf gepatchten Achsen anwenden
            if patch_size >= axis_size:
                continue

            if size == 1:
                w1d = np.ones((1,), dtype=np.float32)
            else:
                w1d = np.hanning(size).astype(np.float32)
                w1d /= w1d.max()  # normalize
                w1d = w1d + eps   # avoid zeros

            reshape_shape = [1] * len(patch_shape)
            reshape_shape[axis] = size
            weights *= w1d.reshape(reshape_shape)

        return weights

    raise ValueError(
        f"Unknown weight_mode '{weight_mode}'. Expected 'average' or 'hann'."
    )


def reconstruct_from_patches(
    patches: List[np.ndarray],
    slices_list: List[Tuple[slice, ...]],
    output_shape: Tuple[int, ...],
    patch_sizes: Tuple[int, ...],
    weight_mode: str = "average",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Reconstruct full array from patches using weighted averaging.

    Hann weighting is applied only along axes that are actually patched
    (i.e. patch_size < axis_size).
    """
    if len(patches) != len(slices_list):
        raise ValueError(
            f"patches and slices_list must have same length, got "
            f"{len(patches)} and {len(slices_list)}"
        )

    if len(patches) == 0:
        raise ValueError("patches must not be empty")

    output_sum = np.zeros(output_shape, dtype=np.float32)
    weight_sum = np.zeros(output_shape, dtype=np.float32)

    for patch, slc in zip(patches, slices_list):
        patch = np.asarray(patch, dtype=np.float32)

        expected_shape = tuple(s.stop - s.start for s in slc)
        if patch.shape != expected_shape:
            raise ValueError(
                f"Patch shape {patch.shape} does not match slice shape {expected_shape}"
            )

        weights = _make_patch_weights(
            patch_shape=patch.shape,
            patch_sizes=patch_sizes,
            output_shape=output_shape,
            weight_mode=weight_mode,
            eps=eps,
        )

        output_sum[slc] += patch * weights
        weight_sum[slc] += weights

    recon = output_sum / weight_sum
    return recon



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
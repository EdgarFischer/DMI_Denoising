from denoising.data.patching import *

import numpy as np
import pytest

def test_random_patch_shape():
    img = np.zeros((2, 16, 16, 64))
    patch_sizes = (5, 5, 32)

    sl = get_random_patch_slices(img.shape, patch_sizes)
    patch = extract_patch(img, sl)

    assert patch.shape == (2, 5, 5, 32)


def test_none_means_full_axis():
    img = np.zeros((2, 16, 16, 64))
    patch_sizes = (None, 5, None)

    sl = get_random_patch_slices(img.shape, patch_sizes)
    patch = extract_patch(img, sl)

    assert patch.shape == (2, 16, 5, 64)


def test_first_axis_is_kept_full():
    img_shape = (2, 16, 16, 64)
    patch_sizes = (5, 5, 32)

    sl = get_random_patch_slices(img_shape, patch_sizes)

    assert sl[0] == slice(None)


def test_slices_are_in_valid_range():
    img_shape = (2, 16, 16, 64)
    patch_sizes = (5, 5, 32)

    for _ in range(100):
        sl = get_random_patch_slices(img_shape, patch_sizes)

        for axis_len, s in zip(img_shape[1:], sl[1:]):
            start = 0 if s.start is None else s.start
            stop = axis_len if s.stop is None else s.stop

            assert 0 <= start <= stop <= axis_len


def test_raises_if_patch_sizes_length_is_wrong():
    img_shape = (2, 16, 16)

    with pytest.raises(ValueError):
        get_random_patch_slices(img_shape, (5,))


def test_raises_if_patch_too_large():
    img_shape = (2, 16, 16)

    with pytest.raises(ValueError):
        get_random_patch_slices(img_shape, (20, 5))


def test_raises_if_patch_size_is_nonpositive():
    img_shape = (2, 16, 16)

    with pytest.raises(ValueError):
        get_random_patch_slices(img_shape, (0, 5))

    with pytest.raises(ValueError):
        get_random_patch_slices(img_shape, (-1, 5))
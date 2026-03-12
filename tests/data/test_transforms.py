import numpy as np
import pytest

from denoising.data.transforms import StratifiedAxisMasking


def test_1d_mask_is_replicated_over_other_axes():
    np.random.seed(0)

    img = np.random.randn(2, 5, 3, 6).astype(np.float32)
    tfm = StratifiedAxisMasking(
        num_masked_pixels=2,
        window_size=3,
        random_mask_noisy=False,
    )

    inp, tgt, mask = tfm(img, masked_axes_local=(3,))

    assert inp.shape == img.shape
    assert tgt.shape == img.shape
    assert mask.shape == img.shape

    # mask should be identical across real/imag, channel and H
    for c in range(img.shape[1]):
        for h in range(img.shape[2]):
            assert np.array_equal(mask[0, 0, 0, :], mask[0, c, h, :])
            assert np.array_equal(mask[0, 0, 0, :], mask[1, c, h, :])

def test_1d_channel_mask_masks_full_channel():
    np.random.seed(1)

    img = np.random.randn(2, 5, 3, 6).astype(np.float32)
    tfm = StratifiedAxisMasking(
        num_masked_pixels=1,
        window_size=3,
        random_mask_noisy=False,
    )

    inp, tgt, mask = tfm(img, masked_axes_local=(1,))

    # there should be exactly one masked channel index
    masked_channels = np.where(mask[0, :, 0, 0])[0]
    assert len(masked_channels) == 1

    ch = masked_channels[0]

    # the whole selected channel should be masked over H,W and both real/imag
    assert np.all(mask[:, ch, :, :])

    # all other channels should be unmasked
    other_channels = [i for i in range(img.shape[1]) if i != ch]
    for oc in other_channels:
        assert not np.any(mask[:, oc, :, :])

def test_2d_mask_is_replicated_over_channels():
    np.random.seed(2)

    img = np.random.randn(2, 5, 4, 6).astype(np.float32)
    tfm = StratifiedAxisMasking(
        num_masked_pixels=3,
        window_size=3,
        random_mask_noisy=False,
    )

    inp, tgt, mask = tfm(img, masked_axes_local=(2, 3))

    assert inp.shape == img.shape
    assert tgt.shape == img.shape
    assert mask.shape == img.shape

    # mask should be identical across channels and real/imag
    ref = mask[0, 0]
    for ch in range(img.shape[1]):
        assert np.array_equal(ref, mask[0, ch])
        assert np.array_equal(ref, mask[1, ch])

def test_random_mask_noisy_masks_only_one_realimag_channel():
    np.random.seed(3)

    img = np.random.randn(2, 5, 3, 6).astype(np.float32)
    tfm = StratifiedAxisMasking(
        num_masked_pixels=2,
        window_size=3,
        random_mask_noisy=True,
    )

    inp, tgt, mask = tfm(img, masked_axes_local=(3,))

    # For every masked position, not both real/imag channels should always be true
    # There should exist masked positions where exactly one of the two channels is masked.
    per_position_sum = mask.sum(axis=0)   # shape (5,3,6)
    assert np.any(per_position_sum == 1)
import numpy as np
import pytest

from denoising.data.mrsi_nd_dataset import MRSiNDataset

def test_dataset_2d_without_channel_axis():
    data = (
        np.random.randn(2, 3, 4, 5, 6)
        + 1j * np.random.randn(2, 3, 4, 5, 6)
    ).astype(np.complex64)

    ds = MRSiNDataset(
        data=data,
        image_axes=(3, 4),
        channel_axis=None,
        masked_axes=(4,),
        num_samples=1,
        phase_prob=0.0,
    )

    inp, tgt, mask = ds[0]

    assert tuple(inp.shape) == (2, 5, 6)
    assert tuple(tgt.shape) == (2, 5, 6)
    assert tuple(mask.shape) == (2, 5, 6)

def test_dataset_2d_with_channel_axis_flattens_correctly():
    data = (
        np.random.randn(2, 3, 4, 5, 6)
        + 1j * np.random.randn(2, 3, 4, 5, 6)
    ).astype(np.complex64)

    # image_axes=(1,4), channel_axis=3
    # => spatial = (3,6), C = size(axis 3) = 5
    ds = MRSiNDataset(
        data=data,
        image_axes=(1, 4),
        channel_axis=3,
        masked_axes=(4,),
        num_samples=1,
        phase_prob=0.0,
    )

    inp, tgt, mask = ds[0]

    assert tuple(inp.shape) == (2 * 5, 3, 6)
    assert tuple(tgt.shape) == (2 * 5, 3, 6)
    assert tuple(mask.shape) == (2 * 5, 3, 6)

class CaptureTransform:
    def __init__(self):
        self.last_img_shape = None
        self.last_masked_axes_local = None

    def __call__(self, img, masked_axes_local=None):
        self.last_img_shape = img.shape
        self.last_masked_axes_local = masked_axes_local
        mask = np.ones_like(img, dtype=bool)
        return img.copy(), img.copy(), mask


def test_dataset_maps_global_masked_axes_to_local_axes():
    data = (
        np.random.randn(2, 3, 4, 5, 6)
        + 1j * np.random.randn(2, 3, 4, 5, 6)
    ).astype(np.complex64)

    tfm = CaptureTransform()

    ds = MRSiNDataset(
        data=data,
        image_axes=(1, 4),
        channel_axis=3,
        masked_axes=(4,),
        transform=tfm,
        num_samples=1,
        phase_prob=0.0,
    )

    _ = ds[0]

    # local structure with channel axis:
    # img shape = (2, C, *spatial) = (2, 5, 3, 6)
    # axis 0 = real/imag
    # axis 1 = channel_axis(global 3)
    # axis 2 = global axis 1
    # axis 3 = global axis 4
    assert tfm.last_img_shape == (2, 5, 3, 6)
    assert tfm.last_masked_axes_local == (3,)

def test_dataset_maps_channel_axis_to_local_axis_1():
    data = (
        np.random.randn(2, 3, 4, 5, 6)
        + 1j * np.random.randn(2, 3, 4, 5, 6)
    ).astype(np.complex64)

    tfm = CaptureTransform()

    ds = MRSiNDataset(
        data=data,
        image_axes=(1, 4),
        channel_axis=3,
        masked_axes=(3,),   # same as channel_axis
        transform=tfm,
        num_samples=1,
        phase_prob=0.0,
    )

    _ = ds[0]

    assert tfm.last_img_shape == (2, 5, 3, 6)
    assert tfm.last_masked_axes_local == (1,)
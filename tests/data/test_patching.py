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


def test_patch_too_large_uses_full_axis():
    img_shape = (2, 16, 16)

    sl = get_random_patch_slices(img_shape, (20, 5))

    assert sl[0] == slice(None)
    assert sl[1] == slice(None)
    assert (sl[2].stop - sl[2].start) == 5


def test_raises_if_patch_size_is_nonpositive():
    img_shape = (2, 16, 16)

    with pytest.raises(ValueError):
        get_random_patch_slices(img_shape, (0, 5))

    with pytest.raises(ValueError):
        get_random_patch_slices(img_shape, (-1, 5))


def test_generate_inference_patches_full_coverage():
    arr = np.zeros((3, 22, 22, 21), dtype=np.float32)

    slices_list, patches, _ = generate_inference_patches(
        arr,
        patch_sizes=(None, 8, 8, 8),
        strides=(None, 4, 4, 4),
        return_patches=True,
    )

    coverage = np.zeros(arr.shape, dtype=np.int32)
    for slc in slices_list:
        coverage[slc] += 1

    assert coverage.min() >= 1
    assert len(slices_list) == len(patches)


def test_generate_inference_patches_reaches_array_end():
    arr = np.zeros((3, 22, 22, 21), dtype=np.float32)

    slices_list, _, _ = generate_inference_patches(
        arr,
        patch_sizes=(None, 8, 8, 8),
        strides=(None, 4, 4, 4),
        return_patches=False,
    )

    coverage = np.zeros(arr.shape, dtype=np.int32)
    for slc in slices_list:
        coverage[slc] += 1

    assert np.all(coverage[:, -1, :, :] >= 1)
    assert np.all(coverage[:, :, -1, :] >= 1)
    assert np.all(coverage[:, :, :, -1] >= 1)


def test_generate_inference_patches_patch_shapes():
    arr = np.zeros((3, 22, 22, 21), dtype=np.float32)

    _, patches, _ = generate_inference_patches(
        arr,
        patch_sizes=(None, 8, 8, 8),
        strides=(None, 4, 4, 4),
        return_patches=True,
    )

    assert len(patches) > 0
    for p in patches:
        assert p.shape == (3, 8, 8, 8)


def test_generate_inference_patches_none_means_full_axis():
    arr = np.zeros((5, 10, 12), dtype=np.float32)

    _, patches, _ = generate_inference_patches(
        arr,
        patch_sizes=(None, 4, 6),
        strides=(None, 2, 3),
        return_patches=True,
    )

    assert len(patches) > 0
    for p in patches:
        assert p.shape[0] == 5


def test_patch_pipeline_identity_no_overlap():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 16, 16, 16), dtype=np.float32)

    slices_list, patches, norm_sizes = generate_inference_patches(
        arr,
        patch_sizes=(None, 8, 8, 8),
        strides=(None, 8, 8, 8),
        return_patches=True,
    )

    recon = reconstruct_from_patches(
        patches=patches,
        slices_list=slices_list,
        output_shape=arr.shape,
        patch_sizes=norm_sizes,
        weight_mode="average",
    )

    np.testing.assert_allclose(recon, arr, rtol=1e-6, atol=1e-6)


def test_patch_pipeline_identity_with_overlap_average():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((3, 22, 22, 21), dtype=np.float32)

    slices_list, patches, norm_sizes = generate_inference_patches(
        arr,
        patch_sizes=(None, 8, 8, 8),
        strides=(None, 4, 4, 4),
        return_patches=True,
    )

    recon = reconstruct_from_patches(
        patches=patches,
        slices_list=slices_list,
        output_shape=arr.shape,
        patch_sizes=norm_sizes,
        weight_mode="average",
    )

    np.testing.assert_allclose(recon, arr, rtol=1e-6, atol=1e-6)


def test_patch_pipeline_identity_with_overlap_hann():
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((3, 22, 22, 21), dtype=np.float32)

    slices_list, patches, norm_sizes = generate_inference_patches(
        arr,
        patch_sizes=(None, 8, 8, 8),
        strides=(None, 4, 4, 4),
        return_patches=True,
    )

    recon = reconstruct_from_patches(
        patches=patches,
        slices_list=slices_list,
        output_shape=arr.shape,
        patch_sizes=norm_sizes,
        weight_mode="hann",
    )

    np.testing.assert_allclose(recon, arr, rtol=1e-5, atol=1e-5)


def test_full_coverage():
    arr = np.zeros((3, 22, 22, 21), dtype=np.float32)

    slices_list, _, _ = generate_inference_patches(
        arr,
        patch_sizes=(None, 8, 8, 8),
        strides=(None, 4, 4, 4),
        return_patches=False,
    )

    coverage = np.zeros(arr.shape, dtype=np.int32)
    for slc in slices_list:
        coverage[slc] += 1

    assert coverage.min() >= 1


def test_reconstruction_shape():
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((2, 11, 13, 7), dtype=np.float32)

    slices_list, patches, norm_sizes = generate_inference_patches(
        arr,
        patch_sizes=(None, 5, 6, 4),
        strides=(None, 3, 4, 2),
        return_patches=True,
    )

    recon = reconstruct_from_patches(
        patches=patches,
        slices_list=slices_list,
        output_shape=arr.shape,
        patch_sizes=norm_sizes,
        weight_mode="average",
    )

    assert recon.shape == arr.shape
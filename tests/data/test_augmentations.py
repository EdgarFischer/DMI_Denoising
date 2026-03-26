import numpy as np
import pytest
from types import SimpleNamespace

from denoising.data.augmentations import (
    apply_global_phase,
    apply_global_scale,
    apply_inversion,
    apply_permutation,
    PatchAugmentationPipeline,
)


def make_structured_complex_img(shape_without_ri):
    """
    Build structured img with shape (2, *shape_without_ri).
    """
    total_shape = (2,) + tuple(shape_without_ri)
    img = np.arange(np.prod(total_shape), dtype=np.float32).reshape(total_shape)
    return img


def make_nontrivial_complex_img(shape_without_ri):
    """
    Structured complex image where real/imag are not trivial and magnitudes are nonzero.
    """
    rng = np.random.default_rng(0)
    real = rng.standard_normal(shape_without_ri).astype(np.float32) + 1.0
    imag = rng.standard_normal(shape_without_ri).astype(np.float32) + 0.5
    return np.stack([real, imag], axis=0).astype(np.float32)


def complex_from_structured(img):
    return img[0] + 1j * img[1]


def test_apply_global_phase_preserves_magnitude_and_changes_values(monkeypatch):
    img = make_nontrivial_complex_img((3, 4, 5))

    # theta = pi/2
    monkeypatch.setattr(np.random, "rand", lambda: 0.25)

    out = apply_global_phase(img)

    z_in = complex_from_structured(img)
    z_out = complex_from_structured(out)

    assert out.shape == img.shape
    assert out.dtype == np.float32

    np.testing.assert_allclose(np.abs(z_out), np.abs(z_in), rtol=1e-6, atol=1e-6)
    assert not np.allclose(out, img)


def test_apply_global_scale_scales_everything_with_same_factor(monkeypatch):
    img = make_nontrivial_complex_img((3, 4, 5))

    monkeypatch.setattr(np.random, "uniform", lambda a, b: 2.5)

    out = apply_global_scale(img, scale_min=0.9, scale_max=1.1)

    assert out.shape == img.shape
    assert out.dtype == np.float32

    np.testing.assert_allclose(out, img * 2.5, rtol=1e-6, atol=1e-6)

    z_in = complex_from_structured(img)
    z_out = complex_from_structured(out)
    np.testing.assert_allclose(np.abs(z_out), 2.5 * np.abs(z_in), rtol=1e-6, atol=1e-6)


def test_apply_inversion_flips_expected_axis(monkeypatch):
    # shape = (2, X, Y)
    img = make_structured_complex_img((3, 4))

    # global axis 0 -> local axis 1
    local_axis_map = {0: 1, 1: 2}

    # First axis flips, second axis does not
    vals = iter([0.0, 1.0])  # <0.5 => flip, >=0.5 => no flip
    monkeypatch.setattr(np.random, "rand", lambda: next(vals))

    out = apply_inversion(
        img,
        local_axis_map=local_axis_map,
        axes_global=(0, 1),
    )

    expected = np.flip(img, axis=1).copy()

    assert out.shape == img.shape
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)


def test_apply_inversion_ignores_nonvisible_axes(monkeypatch):
    img = make_structured_complex_img((3, 4))
    local_axis_map = {0: 1, 1: 2}

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)

    out = apply_inversion(
        img,
        local_axis_map=local_axis_map,
        axes_global=(5,),  # not visible
    )

    np.testing.assert_array_equal(out, img)


def test_apply_permutation_permutes_axes_and_updates_axis_map(monkeypatch):
    # shape = (2, X, Y, Z), with X=Y=Z=3 to allow permutation
    img = make_structured_complex_img((3, 3, 3))

    # global -> local
    local_axis_map = {0: 1, 1: 2, 2: 3}

    # Force permutation [1, 0, 2]
    monkeypatch.setattr(
        np.random,
        "permutation",
        lambda x: np.array([1, 0, 2], dtype=int),
    )

    out, new_map = apply_permutation(
        img,
        local_axis_map=local_axis_map,
        axes_global=(0, 1, 2),
    )

    expected = np.transpose(img, axes=(0, 2, 1, 3)).copy()

    assert out.shape == img.shape
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)

    expected_map = {0: 2, 1: 1, 2: 3}
    assert new_map == expected_map


def test_apply_permutation_noop_if_less_than_two_visible_axes():
    img = make_structured_complex_img((3, 3, 3))
    local_axis_map = {0: 1, 1: 2, 2: 3}

    out, new_map = apply_permutation(
        img,
        local_axis_map=local_axis_map,
        axes_global=(5,),  # not visible
    )

    np.testing.assert_array_equal(out, img)
    assert new_map == local_axis_map


def test_apply_permutation_noop_if_axis_sizes_do_not_match(monkeypatch):
    # shape = (2, 2, 3, 4), sizes differ
    img = make_structured_complex_img((2, 3, 4))
    local_axis_map = {0: 1, 1: 2, 2: 3}

    monkeypatch.setattr(
        np.random,
        "permutation",
        lambda x: np.array([1, 0, 2], dtype=int),
    )

    out, new_map = apply_permutation(
        img,
        local_axis_map=local_axis_map,
        axes_global=(0, 1, 2),
    )

    np.testing.assert_array_equal(out, img)
    assert new_map == local_axis_map


def test_pipeline_identity_if_cfg_none():
    img = make_nontrivial_complex_img((3, 4, 5))
    local_axis_map = {0: 1, 1: 2, 2: 3}

    pipe = PatchAugmentationPipeline(None)
    out, new_map = pipe(img, local_axis_map)

    np.testing.assert_array_equal(out, img)
    assert new_map == local_axis_map


def test_pipeline_identity_if_disabled():
    img = make_nontrivial_complex_img((3, 4, 5))
    local_axis_map = {0: 1, 1: 2, 2: 3}

    cfg = SimpleNamespace(
        enabled=False,
        global_phase=SimpleNamespace(enabled=True, p=1.0),
        global_scale=SimpleNamespace(enabled=True, p=1.0, min=2.0, max=2.0),
        inversion=SimpleNamespace(enabled=True, p=1.0, axes=(0, 1)),
        permutation=SimpleNamespace(enabled=True, p=1.0, axes=(0, 1)),
    )

    pipe = PatchAugmentationPipeline(cfg)
    out, new_map = pipe(img, local_axis_map)

    np.testing.assert_array_equal(out, img)
    assert new_map == local_axis_map


def test_pipeline_only_global_scale(monkeypatch):
    img = make_nontrivial_complex_img((3, 4, 5))
    local_axis_map = {0: 1, 1: 2, 2: 3}

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "uniform", lambda a, b: 3.0)

    cfg = SimpleNamespace(
        enabled=True,
        global_phase=SimpleNamespace(enabled=False, p=0.0),
        global_scale=SimpleNamespace(enabled=True, p=1.0, min=3.0, max=3.0),
        inversion=SimpleNamespace(enabled=False, p=0.0, axes=()),
        permutation=SimpleNamespace(enabled=False, p=0.0, axes=()),
    )

    pipe = PatchAugmentationPipeline(cfg)
    out, new_map = pipe(img, local_axis_map)

    np.testing.assert_allclose(out, img * 3.0, rtol=1e-6, atol=1e-6)
    assert new_map == local_axis_map
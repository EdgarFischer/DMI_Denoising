import numpy as np

from denoising.eval.empirical_noise_correlations import (
    estimate_axis_correlations,
    estimate_axis_correlations_for_dataset_list,
)


def _broadcast_mask(mask_3d, data_shape):
    n_extra_dims = len(data_shape) - mask_3d.ndim
    mask = mask_3d.reshape(mask_3d.shape + (1,) * n_extra_dims)
    return np.broadcast_to(mask, data_shape)


def _assert_white_noise_correlations_small(corrs, axes, threshold=0.1):
    assert set(corrs.keys()) == set(axes)

    for ax in axes:
        corr = corrs[ax]
        assert np.isclose(corr[0], 1.0)
        assert np.all(np.abs(corr[1:]) < threshold)


def _assert_constant_ones_correlations_are_one(corrs, axes):
    assert set(corrs.keys()) == set(axes)

    for ax in axes:
        corr = corrs[ax]
        assert np.allclose(corr, 1.0)


def test_estimate_axis_correlations_white_noise_nonspatial_axes():
    rng = np.random.default_rng(0)

    data = rng.normal(size=(6, 6, 6, 40, 20))
    mask_noise = np.ones(data.shape, dtype=bool)

    corrs = estimate_axis_correlations(
        data=data,
        mask_noise=mask_noise,
        axes=(3, 4),
        subtract_mean=True,
        normalize=True,
    )

    _assert_white_noise_correlations_small(corrs, axes=(3, 4))


def test_estimate_axis_correlations_constant_ones_no_mean_subtraction_nonspatial_axes():
    data = np.ones((3, 3, 3, 8, 5), dtype=np.float64)
    mask_noise = np.ones(data.shape, dtype=bool)

    corrs = estimate_axis_correlations(
        data=data,
        mask_noise=mask_noise,
        axes=(3, 4),
        subtract_mean=False,
        normalize=True,
    )

    _assert_constant_ones_correlations_are_one(corrs, axes=(3, 4))


def test_estimate_axis_correlations_for_dataset_list_single_dataset_matches_direct_call_nonspatial_axes():
    rng = np.random.default_rng(1)

    data = rng.normal(size=(4, 4, 4, 10, 6))
    mask_noise_3d = rng.random((4, 4, 4)) > 0.3
    mask_noise = _broadcast_mask(mask_noise_3d, data.shape)

    direct_corrs = estimate_axis_correlations(
        data=data,
        mask_noise=mask_noise,
        axes=(3, 4),
        subtract_mean=True,
        normalize=True,
    )

    list_corrs = estimate_axis_correlations_for_dataset_list(
        dataset_list=[data],
        mask_list=[mask_noise],
        axes=(3, 4),
        subtract_mean=True,
        normalize=True,
    )

    assert len(list_corrs) == 1
    assert set(list_corrs[0].keys()) == set(direct_corrs.keys())

    for ax in (3, 4):
        assert np.allclose(list_corrs[0][ax], direct_corrs[ax], equal_nan=True)


def test_estimate_axis_correlations_white_noise_spatial_axes():
    rng = np.random.default_rng(0)

    data = rng.normal(size=(12, 12, 12, 8, 4))
    mask_noise = np.ones(data.shape, dtype=bool)

    corrs = estimate_axis_correlations(
        data=data,
        mask_noise=mask_noise,
        axes=(0, 1, 2),
        subtract_mean=True,
        normalize=True,
    )

    _assert_white_noise_correlations_small(corrs, axes=(0, 1, 2))


def test_estimate_axis_correlations_constant_ones_no_mean_subtraction_spatial_axes():
    data = np.ones((6, 5, 4, 3, 2), dtype=np.float64)
    mask_noise = np.ones(data.shape, dtype=bool)

    corrs = estimate_axis_correlations(
        data=data,
        mask_noise=mask_noise,
        axes=(0, 1, 2),
        subtract_mean=False,
        normalize=True,
    )

    _assert_constant_ones_correlations_are_one(corrs, axes=(0, 1, 2))


def test_estimate_axis_correlations_for_dataset_list_single_dataset_matches_direct_call_spatial_axes():
    rng = np.random.default_rng(1)

    data = rng.normal(size=(7, 6, 5, 4, 3))
    mask_noise_3d = rng.random((7, 6, 5)) > 0.3
    mask_noise = _broadcast_mask(mask_noise_3d, data.shape)

    direct_corrs = estimate_axis_correlations(
        data=data,
        mask_noise=mask_noise,
        axes=(0, 1, 2),
        max_lag=3,
        subtract_mean=True,
        normalize=True,
    )

    list_corrs = estimate_axis_correlations_for_dataset_list(
        dataset_list=[data],
        mask_list=[mask_noise],
        axes=(0, 1, 2),
        max_lag=3,
        subtract_mean=True,
        normalize=True,
    )

    assert len(list_corrs) == 1
    assert set(list_corrs[0].keys()) == set(direct_corrs.keys())

    for ax in (0, 1, 2):
        assert np.allclose(list_corrs[0][ax], direct_corrs[ax], equal_nan=True)
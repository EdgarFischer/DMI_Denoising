import numpy as np

from denoising.eval.empirical_noise_correlations import *

def test_estimate_1d_acf_from_masked_voxels_white_noise():
    rng = np.random.default_rng(0)

    # genug Samples, damit die empirischen Korrelationen klein werden
    data = rng.normal(size=(6, 6, 6, 40, 20))
    mask_noise = np.ones((6, 6, 6), dtype=bool)

    acfs = estimate_1d_acf_from_masked_voxels(
        data,
        mask_noise,
        subtract_mean=True,
        normalize=True,
    )

    # 2 nicht-räumliche Achsen: t und T
    assert len(acfs) == 2

    for acf in acfs:
        # Lag 0 sollte nach Normierung 1 sein
        assert np.isclose(acf[0], 1.0)

        # alle weiteren Lags sollen klein sein
        assert np.all(np.abs(acf[1:]) < 0.1)


def test_estimate_1d_acf_from_masked_voxels_constant_ones_no_mean_subtraction():
    data = np.ones((3, 3, 3, 8, 5), dtype=np.float64)
    mask_noise = np.ones((3, 3, 3), dtype=bool)

    acfs = estimate_1d_acf_from_masked_voxels(
        data,
        mask_noise,
        subtract_mean=False,
        normalize=True,
    )

    assert len(acfs) == 2

    for acf in acfs:
        assert np.allclose(acf, 1.0)


def test_estimate_1d_acfs_for_dataset_list_single_dataset_matches_direct_call():
    rng = np.random.default_rng(1)

    data = rng.normal(size=(4, 4, 4, 10, 6))
    mask_noise = rng.random((4, 4, 4)) > 0.3

    direct_acfs = estimate_1d_acf_from_masked_voxels(
        data,
        mask_noise,
        subtract_mean=True,
        normalize=True,
    )

    list_acfs = estimate_1d_acfs_for_dataset_list(
        dataset_list=[data],
        mask_list=[mask_noise],
        subtract_mean=True,
        normalize=True,
    )

    assert len(list_acfs) == 1
    assert len(list_acfs[0]) == len(direct_acfs)

    for acf_from_list, acf_direct in zip(list_acfs[0], direct_acfs):
        assert np.allclose(acf_from_list, acf_direct)

import numpy as np

from denoising.eval.empirical_noise_correlations import (
    estimate_spatial_correlations,
    estimate_spatial_correlations_for_dataset_list,
)


def test_estimate_spatial_correlations_white_noise():
    rng = np.random.default_rng(0)

    data = rng.normal(size=(12, 12, 12, 8, 4))
    mask_noise = np.ones((12, 12, 12), dtype=bool)

    spatial_corrs = estimate_spatial_correlations(
        data,
        mask_noise,
        subtract_mean=True,
        normalize=True,
    )

    assert set(spatial_corrs.keys()) == {"x", "y", "z"}

    for axis in ["x", "y", "z"]:
        corr = spatial_corrs[axis]

        assert np.isclose(corr[0], 1.0)
        assert np.all(np.abs(corr[1:]) < 0.1)


def test_estimate_spatial_correlations_constant_ones_no_mean_subtraction():
    data = np.ones((6, 5, 4, 3, 2), dtype=np.float64)
    mask_noise = np.ones((6, 5, 4), dtype=bool)

    spatial_corrs = estimate_spatial_correlations(
        data,
        mask_noise,
        subtract_mean=False,
        normalize=True,
    )

    assert set(spatial_corrs.keys()) == {"x", "y", "z"}

    for axis in ["x", "y", "z"]:
        corr = spatial_corrs[axis]
        assert np.allclose(corr, 1.0)


def test_estimate_spatial_correlations_for_dataset_list_single_dataset_matches_direct_call():
    rng = np.random.default_rng(1)

    data = rng.normal(size=(7, 6, 5, 4, 3))
    mask_noise = rng.random((7, 6, 5)) > 0.3

    direct_corrs = estimate_spatial_correlations(
        data=data,
        mask_noise=mask_noise,
        max_lag=3,
        subtract_mean=True,
        normalize=True,
    )

    list_corrs = estimate_spatial_correlations_for_dataset_list(
        dataset_list=[data],
        mask_list=[mask_noise],
        max_lag=3,
        subtract_mean=True,
        normalize=True,
    )

    assert len(list_corrs) == 1
    assert set(list_corrs[0].keys()) == {"x", "y", "z"}

    for axis in ["x", "y", "z"]:
        assert np.allclose(list_corrs[0][axis], direct_corrs[axis], equal_nan=True)
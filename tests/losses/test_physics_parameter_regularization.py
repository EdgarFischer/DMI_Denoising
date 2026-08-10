from types import SimpleNamespace

import torch

from denoising.training.trainers.trainer_n2v import (
    physics_parameter_regularization,
)


def _output(shifts, fwhm, standardized=None):
    parameters = SimpleNamespace(
        metabolite_frequency_shift_hz=shifts,
        metabolite_lorentzian_fwhm_hz=fwhm,
        lineshape_kernel=None,
        baseline_coefficients_real=None,
        baseline_coefficients_imag=None,
        amplitudes=torch.zeros(*shifts.shape[:-1], shifts.shape[-1]),
    )
    return SimpleNamespace(
        reconstruction=torch.zeros(1), parameters=parameters,
        standardized_parameter_maps=standardized,
    )


def test_parameter_priors_are_standardized_and_brain_masked():
    # (B, X, Y, M), with only voxel [0, 0] included by the sampling mask.
    shifts = torch.tensor([[[[1.0, -1.0], [100.0, 100.0]]]])
    fwhm = torch.tensor([[[[7.5, 2.5], [100.0, 100.0]]]])
    sampling_mask = torch.zeros(1, 2, 1, 2, 4)
    sampling_mask[:, :, 0, 0, 0] = 1
    cfg = SimpleNamespace(
        enabled=True,
        shift_mean_hz=0.0,
        shift_std_hz=1.0,
        fwhm_mean_hz=5.0,
        fwhm_std_hz=2.5,
        shift_weight=1.0,
        fwhm_weight=1.0,
        kernel_curvature_weight=0.0,
        baseline_curvature_weight=0.0,
        voigt_nuisance_weight=0.0,
    )

    shift_prior, fwhm_prior, kernel_prior, baseline_prior, nuisance_prior = physics_parameter_regularization(
        _output(shifts, fwhm), sampling_mask, spectral_axis=2, cfg=cfg
    )

    assert torch.isclose(shift_prior, torch.tensor(1.0))
    assert torch.isclose(fwhm_prior, torch.tensor(1.0))
    assert kernel_prior == 0
    assert baseline_prior == 0
    assert nuisance_prior == 0


def test_voigt_nuisance_prior_sums_five_standardized_terms():
    shifts = torch.zeros(1, 1, 1, 2)
    fwhm = torch.ones(1, 1, 1, 2)
    # Two amplitude channels followed by five nuisance Z-score channels.
    standardized = torch.zeros(1, 7, 1, 1)
    standardized[:, 2:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sampling_mask = torch.ones(1, 1, 1, 1, 4)
    cfg = SimpleNamespace(
        enabled=True,
        shift_mean_hz=0.0,
        shift_std_hz=1.0,
        fwhm_mean_hz=0.0,
        fwhm_std_hz=1.0,
        shift_weight=0.0,
        fwhm_weight=0.0,
        kernel_curvature_weight=0.0,
        baseline_curvature_weight=0.0,
        voigt_nuisance_weight=0.01,
    )

    *_, nuisance_prior = physics_parameter_regularization(
        _output(shifts, fwhm, standardized),
        sampling_mask,
        spectral_axis=2,
        cfg=cfg,
    )
    assert torch.isclose(nuisance_prior, torch.tensor(55.0))

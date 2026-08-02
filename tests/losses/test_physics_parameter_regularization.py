from types import SimpleNamespace

import torch

from denoising.training.trainers.trainer_n2v import (
    physics_parameter_regularization,
)


def _output(shifts, fwhm):
    parameters = SimpleNamespace(
        metabolite_frequency_shift_hz=shifts,
        metabolite_lorentzian_fwhm_hz=fwhm,
    )
    return SimpleNamespace(
        reconstruction=torch.zeros(1), parameters=parameters
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
    )

    shift_prior, fwhm_prior = physics_parameter_regularization(
        _output(shifts, fwhm), sampling_mask, spectral_axis=2, cfg=cfg
    )

    assert torch.isclose(shift_prior, torch.tensor(1.0))
    assert torch.isclose(fwhm_prior, torch.tensor(1.0))

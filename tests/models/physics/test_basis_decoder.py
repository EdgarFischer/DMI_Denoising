from __future__ import annotations

import math

import pytest
import torch

from denoising.models.physics import (
    BaselineFreeBasisDecoder,
    SpectralParameters,
)


def _basis(n_basis: int = 3, n_timepoints: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.complex(
        torch.randn(n_basis, n_timepoints, generator=generator),
        torch.randn(n_basis, n_timepoints, generator=generator),
    )


def _parameters(
    leading_shape: tuple[int, ...], n_basis: int
) -> SpectralParameters:
    generator = torch.Generator().manual_seed(23)
    return SpectralParameters(
        amplitudes=torch.rand(*leading_shape, n_basis, generator=generator),
        frequency_shift_hz=torch.randn(*leading_shape, generator=generator),
        lorentzian_fwhm_hz=(
            2.0 + torch.rand(*leading_shape, generator=generator)
        ),
        gaussian_fwhm_hz=(
            1.0 + torch.rand(*leading_shape, generator=generator)
        ),
        zero_order_phase_radians=(
            0.2 * torch.randn(*leading_shape, generator=generator)
        ),
        first_order_phase_rad_per_hz=(
            1e-3 * torch.randn(*leading_shape, generator=generator)
        ),
    )


def _walinet_reference(
    basis: torch.Tensor,
    dwell_time: float,
    p: SpectralParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent transcription of WALINET MetaboliteSimulator."""
    metabolite_fids = p.amplitudes.to(basis.dtype) @ basis
    time = torch.arange(basis.shape[-1], dtype=basis.real.dtype) * dwell_time
    phase = torch.polar(
        torch.ones_like(p.frequency_shift_hz[..., None] * time),
        p.zero_order_phase_radians[..., None]
        + 2 * math.pi * p.frequency_shift_hz[..., None] * time,
    )
    decay = torch.exp(
        -math.pi * p.lorentzian_fwhm_hz[..., None] * time
        - (math.pi * p.gaussian_fwhm_hz[..., None] * time).square()
        / (4 * math.log(2))
    )
    affected = metabolite_fids * phase * decay
    frequency = torch.fft.fftfreq(
        basis.shape[-1], d=dwell_time, dtype=basis.real.dtype
    )
    phase1_angle = p.first_order_phase_rad_per_hz[..., None] * frequency
    phase1 = torch.polar(torch.ones_like(phase1_angle), phase1_angle)
    fid = torch.fft.ifft(torch.fft.fft(affected) * phase1)
    spectrum = torch.fft.fftshift(torch.fft.fft(fid), dim=-1)
    return fid, spectrum


@pytest.mark.parametrize("leading_shape", [(4,), (2, 3, 4)])
def test_matches_walinet_equations(leading_shape):
    basis = _basis()
    parameters = _parameters(leading_shape, basis.shape[0])
    decoder = BaselineFreeBasisDecoder(basis, dwell_time_seconds=1 / 2000)

    expected_fid, expected_spectrum = _walinet_reference(
        basis, 1 / 2000, parameters
    )

    torch.testing.assert_close(decoder.decode_fids(parameters), expected_fid)
    torch.testing.assert_close(
        decoder.decode_spectra(parameters), expected_spectrum
    )


def test_spatial_decoding_equals_independent_voxel_decoding():
    basis = _basis(n_basis=2)
    decoder = BaselineFreeBasisDecoder(basis, 1 / 1500)
    parameters = _parameters((2, 3), basis.shape[0])

    spatial = decoder(parameters)
    for row in range(2):
        for column in range(3):
            voxel = SpectralParameters(
                amplitudes=parameters.amplitudes[row, column],
                frequency_shift_hz=parameters.frequency_shift_hz[row, column],
                lorentzian_fwhm_hz=(
                    parameters.lorentzian_fwhm_hz[row, column]
                ),
                gaussian_fwhm_hz=parameters.gaussian_fwhm_hz[row, column],
                zero_order_phase_radians=(
                    parameters.zero_order_phase_radians[row, column]
                ),
                first_order_phase_rad_per_hz=(
                    parameters.first_order_phase_rad_per_hz[row, column]
                ),
            )
            torch.testing.assert_close(spatial[row, column], decoder(voxel))


def test_all_physical_inputs_receive_finite_gradients():
    basis = _basis(n_basis=2)
    decoder = BaselineFreeBasisDecoder(basis, 1 / 2000)
    values = {
        name: value.detach().requires_grad_(True)
        for name, value in _parameters((3,), basis.shape[0]).__dict__.items()
    }
    parameters = SpectralParameters(**values)

    spectrum = decoder(parameters)
    spectrum.abs().square().mean().backward()

    for name, value in values.items():
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name


def test_decoder_has_no_trainable_parameters_or_baseline_state():
    decoder = BaselineFreeBasisDecoder(_basis(), 1 / 2000)

    assert list(decoder.parameters()) == []
    assert not any("baseline" in name.lower() for name in decoder.state_dict())


def test_parameter_shape_mismatch_is_rejected():
    basis = _basis()
    decoder = BaselineFreeBasisDecoder(basis, 1 / 2000)
    parameters = _parameters((2,), basis.shape[0])
    invalid = SpectralParameters(
        **{
            **parameters.__dict__,
            "frequency_shift_hz": torch.zeros(2, 1),
        }
    )

    with pytest.raises(ValueError, match="frequency_shift_hz"):
        decoder(invalid)

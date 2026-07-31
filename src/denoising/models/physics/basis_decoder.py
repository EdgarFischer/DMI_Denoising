"""Differentiable LCModel-basis decoder without a baseline component."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from .parameters import SpectralParameters


OutputDomain = Literal["fid", "spectrum"]


class BaselineFreeBasisDecoder(nn.Module):
    """Synthesize complex FIDs or spectra from physical parameter maps.

    The implementation deliberately contains neither a spline/polynomial
    baseline nor a learned residual path. Measured macromolecule components
    may still be included explicitly as rows of ``basis_fids``.

    The signal convention is identical to WALINET's ``MetaboliteSimulator``:

    ``sum(a_m b_m(t)) * exp(+i(phi0 + 2*pi*df*t))``
    ``* exp(-pi*L*t - (pi*G*t)^2/(4*ln(2)))``

    First-order phase is subsequently applied in the unshifted frequency
    domain as ``exp(+i * phi1 * f)``.
    """

    def __init__(self, basis_fids: Tensor, dwell_time_seconds: float) -> None:
        super().__init__()

        if basis_fids.ndim != 2:
            raise ValueError(
                "basis_fids must have shape (n_basis, n_timepoints), "
                f"but found {tuple(basis_fids.shape)}."
            )
        if not torch.is_complex(basis_fids):
            raise TypeError("basis_fids must be a complex-valued tensor.")
        if basis_fids.shape[0] < 1 or basis_fids.shape[1] < 2:
            raise ValueError(
                "basis_fids must contain at least one basis component and "
                "two timepoints."
            )
        if not math.isfinite(dwell_time_seconds) or dwell_time_seconds <= 0:
            raise ValueError("dwell_time_seconds must be finite and > 0.")

        basis_fids = basis_fids.detach().contiguous()
        real_dtype = basis_fids.real.dtype
        n_timepoints = int(basis_fids.shape[-1])

        time_axis_seconds = torch.arange(
            n_timepoints,
            device=basis_fids.device,
            dtype=real_dtype,
        ) * float(dwell_time_seconds)
        frequency_axis_hz = torch.fft.fftfreq(
            n_timepoints,
            d=float(dwell_time_seconds),
            device=basis_fids.device,
            dtype=real_dtype,
        )

        self.register_buffer("basis_fids", basis_fids)
        self.register_buffer("time_axis_seconds", time_axis_seconds)
        self.register_buffer("frequency_axis_hz", frequency_axis_hz)
        self.dwell_time_seconds = float(dwell_time_seconds)

    @property
    def n_basis_components(self) -> int:
        return int(self.basis_fids.shape[0])

    @property
    def n_timepoints(self) -> int:
        return int(self.basis_fids.shape[1])

    def _validate_parameters(self, parameters: SpectralParameters) -> None:
        if not isinstance(parameters, SpectralParameters):
            raise TypeError("parameters must be a SpectralParameters instance.")
        if parameters.amplitudes.shape[-1] != self.n_basis_components:
            raise ValueError(
                "The final amplitudes dimension must match the basis: "
                f"expected {self.n_basis_components}, found "
                f"{parameters.amplitudes.shape[-1]}."
            )

        expected = parameters.leading_shape
        nuisance_parameters = {
            "frequency_shift_hz": parameters.frequency_shift_hz,
            "lorentzian_fwhm_hz": parameters.lorentzian_fwhm_hz,
            "gaussian_fwhm_hz": parameters.gaussian_fwhm_hz,
            "zero_order_phase_radians": parameters.zero_order_phase_radians,
            "first_order_phase_rad_per_hz": (
                parameters.first_order_phase_rad_per_hz
            ),
        }
        for name, value in nuisance_parameters.items():
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, but found "
                    f"{tuple(value.shape)}."
                )

    def decode_fids(self, parameters: SpectralParameters) -> Tensor:
        """Decode parameters to complex FIDs with shape ``(..., time)``."""
        self._validate_parameters(parameters)

        amplitudes = parameters.amplitudes.to(dtype=self.basis_fids.dtype)
        metabolite_fids = amplitudes @ self.basis_fids

        time = self.time_axis_seconds
        phase_angle = (
            parameters.zero_order_phase_radians[..., None]
            + 2.0
            * math.pi
            * parameters.frequency_shift_hz[..., None]
            * time
        )
        phase_factor = torch.polar(torch.ones_like(phase_angle), phase_angle)

        lorentzian_exponent = (
            math.pi * parameters.lorentzian_fwhm_hz[..., None] * time
        )
        gaussian_exponent = (
            (math.pi * parameters.gaussian_fwhm_hz[..., None] * time).square()
            / (4.0 * math.log(2.0))
        )
        affected_fids = metabolite_fids * phase_factor * torch.exp(
            -lorentzian_exponent - gaussian_exponent
        )

        first_order_phase = parameters.first_order_phase_rad_per_hz
        spectra_unshifted = torch.fft.fft(affected_fids, dim=-1)
        phase1_angle = (
            first_order_phase[..., None] * self.frequency_axis_hz
        )
        phase1_factor = torch.polar(
            torch.ones_like(phase1_angle), phase1_angle
        )
        return torch.fft.ifft(
            spectra_unshifted * phase1_factor, dim=-1
        ).contiguous()

    def decode_spectra(self, parameters: SpectralParameters) -> Tensor:
        """Decode parameters to complex fftshifted spectra."""
        return torch.fft.fftshift(
            torch.fft.fft(self.decode_fids(parameters), dim=-1), dim=-1
        ).contiguous()

    def forward(
        self,
        parameters: SpectralParameters,
        output_domain: OutputDomain = "spectrum",
    ) -> Tensor:
        if output_domain == "fid":
            return self.decode_fids(parameters)
        if output_domain == "spectrum":
            return self.decode_spectra(parameters)
        raise ValueError(
            "output_domain must be either 'fid' or 'spectrum', "
            f"but found {output_domain!r}."
        )

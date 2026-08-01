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

        lcmodel_values = (
            parameters.metabolite_frequency_shift_hz,
            parameters.metabolite_lorentzian_fwhm_hz,
            parameters.lineshape_kernel,
        )
        if any(value is not None for value in lcmodel_values):
            if not all(value is not None for value in lcmodel_values):
                raise ValueError(
                    "LCModel-like shift, Lorentzian, and kernel parameters "
                    "must either all be provided or all be omitted."
                )
            component_shape = (*expected, self.n_basis_components)
            if tuple(parameters.metabolite_frequency_shift_hz.shape) != component_shape:
                raise ValueError(
                    "metabolite_frequency_shift_hz must have shape "
                    f"{component_shape}."
                )
            if tuple(parameters.metabolite_lorentzian_fwhm_hz.shape) != component_shape:
                raise ValueError(
                    "metabolite_lorentzian_fwhm_hz must have shape "
                    f"{component_shape}."
                )
            if parameters.lineshape_kernel.shape[:-1] != expected:
                raise ValueError(
                    "lineshape_kernel leading dimensions must match amplitudes."
                )
            if (
                parameters.lineshape_kernel.shape[-1] < 3
                or parameters.lineshape_kernel.shape[-1] % 2 != 1
            ):
                raise ValueError("lineshape_kernel length must be odd and >= 3.")

    @staticmethod
    def _same_convolution(signal: Tensor, kernel: Tensor) -> Tensor:
        """Batched linear convolution with one real kernel per spectrum."""
        n_signal = signal.shape[-1]
        n_kernel = kernel.shape[-1]
        full_length = n_signal + n_kernel - 1
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(kernel.dtype).eps
        )
        convolved = torch.fft.ifft(
            torch.fft.fft(signal, n=full_length, dim=-1)
            * torch.fft.fft(kernel, n=full_length, dim=-1),
            n=full_length,
            dim=-1,
        )
        start = n_kernel // 2
        return convolved[..., start : start + n_signal]

    def _decode_lcmodel_spectra(self, parameters: SpectralParameters) -> Tensor:
        """Phive/LCModel-like synthesis without any baseline component."""
        time = self.time_axis_seconds
        amplitudes = parameters.amplitudes.to(dtype=self.basis_fids.dtype)
        # Accumulate components one at a time. Materializing
        # (..., n_metabolites, n_timepoints) would be prohibitively large for
        # the configured 100 x 16 x 16 training batches.
        fid = torch.zeros(
            *parameters.leading_shape,
            self.n_timepoints,
            dtype=self.basis_fids.dtype,
            device=self.basis_fids.device,
        )
        for index in range(self.n_basis_components):
            effective_shift = (
                parameters.frequency_shift_hz
                + parameters.metabolite_frequency_shift_hz[..., index]
            )
            phase = torch.polar(
                torch.ones_like(effective_shift[..., None] * time),
                2.0 * math.pi * effective_shift[..., None] * time,
            )
            decay = torch.exp(
                -math.pi
                * parameters.metabolite_lorentzian_fwhm_hz[..., index, None]
                * time
            )
            fid = fid + (
                amplitudes[..., index, None]
                * self.basis_fids[index]
                * phase
                * decay
            )
        phase0 = torch.polar(
            torch.ones_like(parameters.zero_order_phase_radians[..., None]),
            parameters.zero_order_phase_radians[..., None],
        )
        spectrum = torch.fft.fft(fid * phase0, dim=-1)
        phase1_angle = (
            parameters.first_order_phase_rad_per_hz[..., None]
            * self.frequency_axis_hz
        )
        spectrum = spectrum * torch.polar(
            torch.ones_like(phase1_angle), phase1_angle
        )
        spectrum = torch.fft.fftshift(spectrum, dim=-1)
        return self._same_convolution(
            spectrum, parameters.lineshape_kernel.to(spectrum.real.dtype)
        ).contiguous()

    def decode_fids(self, parameters: SpectralParameters) -> Tensor:
        """Decode parameters to complex FIDs with shape ``(..., time)``."""
        self._validate_parameters(parameters)

        if parameters.lineshape_kernel is not None:
            return torch.fft.ifft(
                torch.fft.ifftshift(
                    self._decode_lcmodel_spectra(parameters), dim=-1
                ),
                dim=-1,
            ).contiguous()

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
        self._validate_parameters(parameters)
        if parameters.lineshape_kernel is not None:
            return self._decode_lcmodel_spectra(parameters)
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

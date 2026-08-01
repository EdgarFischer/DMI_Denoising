"""Canonical physical parameters used by the spectral decoder."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class SpectralParameters:
    """Physical parameters for basis reconstruction with optional forD baseline.

    ``amplitudes`` has shape ``(..., n_basis)``. Every nuisance-parameter
    tensor has the matching leading shape ``(...)``. Leading dimensions may
    represent a batch alone or a batch plus arbitrary spatial dimensions.

    Units follow the WALINET metabolite simulator exactly:

    - ``frequency_shift_hz``: Hz
    - ``lorentzian_fwhm_hz``: Lorentzian FWHM in Hz
    - ``gaussian_fwhm_hz``: Gaussian FWHM in Hz
    - ``zero_order_phase_radians``: radians
    - ``first_order_phase_rad_per_hz``: radians per Hz, referenced to 0 Hz
    """

    amplitudes: Tensor
    frequency_shift_hz: Tensor
    lorentzian_fwhm_hz: Tensor
    gaussian_fwhm_hz: Tensor
    zero_order_phase_radians: Tensor
    first_order_phase_rad_per_hz: Tensor
    metabolite_frequency_shift_hz: Tensor | None = None
    metabolite_lorentzian_fwhm_hz: Tensor | None = None
    lineshape_kernel: Tensor | None = None
    baseline_coefficients_real: Tensor | None = None
    baseline_coefficients_imag: Tensor | None = None

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return tuple(self.amplitudes.shape[:-1])

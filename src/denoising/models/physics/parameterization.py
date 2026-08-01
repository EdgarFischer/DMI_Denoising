"""Minimal physical constraints for decoder parameters."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .parameters import SpectralParameters


class MinimalPhysicalParameterization(nn.Module):
    """Convert raw maps using only physically necessary constraints.

    Amplitudes are non-negative through softplus. Both FWHM values use
    log1p(softplus(raw)): they remain positive and unbounded, while their
    gradients become progressively smaller at large linewidths. Frequency
    shift and phases remain completely unbounded. No additional coordinate
    scaling, bounds, or priors are applied.
    """

    def __init__(self, n_basis_components: int) -> None:
        super().__init__()
        self.n_basis_components = int(n_basis_components)
        if self.n_basis_components < 1:
            raise ValueError("n_basis_components must be >= 1.")

    @property
    def n_output_parameters(self) -> int:
        return self.n_basis_components + 5

    def forward(self, raw: Tensor) -> SpectralParameters:
        if raw.ndim != 4:
            raise ValueError(
                "raw parameter maps must have shape (B, P, X, Y), "
                f"but found {tuple(raw.shape)}."
            )
        if raw.shape[1] != self.n_output_parameters:
            raise ValueError(
                f"Expected {self.n_output_parameters} parameter channels, "
                f"found {raw.shape[1]}."
            )

        amplitudes = torch.nn.functional.softplus(
            raw[:, : self.n_basis_components]
        ).movedim(1, -1)
        nuisance = raw[:, self.n_basis_components :]
        lorentzian_fwhm_hz = torch.log1p(
            torch.nn.functional.softplus(nuisance[:, 1])
        )
        gaussian_fwhm_hz = torch.log1p(
            torch.nn.functional.softplus(nuisance[:, 2])
        )
        return SpectralParameters(
            amplitudes=amplitudes,
            frequency_shift_hz=nuisance[:, 0],
            lorentzian_fwhm_hz=lorentzian_fwhm_hz,
            gaussian_fwhm_hz=gaussian_fwhm_hz,
            zero_order_phase_radians=nuisance[:, 3],
            first_order_phase_rad_per_hz=nuisance[:, 4],
        )


class StandardizedPhysicalParameterization(nn.Module):
    """Predict standardized coordinates and decode them to physical units."""

    def __init__(
        self,
        n_basis_components: int,
        means: tuple[float, ...],
        stds: tuple[float, ...],
        teacher_to_model_amplitude_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_basis_components = int(n_basis_components)
        expected = self.n_basis_components + 5
        if len(means) != expected or len(stds) != expected:
            raise ValueError(f"Expected {expected} parameter means/stds.")
        mean = torch.tensor(means, dtype=torch.float32)
        std = torch.tensor(stds, dtype=torch.float32)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Parameter means/stds must be finite.")
        if not torch.all(std > 0):
            raise ValueError("Every parameter standard deviation must be > 0.")
        if (
            not torch.isfinite(torch.tensor(teacher_to_model_amplitude_scale))
            or teacher_to_model_amplitude_scale <= 0
        ):
            raise ValueError("teacher_to_model_amplitude_scale must be > 0.")
        self.register_buffer("parameter_mean", mean)
        self.register_buffer("parameter_std", std)
        self.register_buffer(
            "teacher_to_model_amplitude_scale",
            torch.tensor(float(teacher_to_model_amplitude_scale)),
        )

    @property
    def n_output_parameters(self) -> int:
        return self.n_basis_components + 5

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        return value + torch.log(-torch.expm1(-value))

    @classmethod
    def _inverse_log_softplus(cls, value: Tensor) -> Tensor:
        return cls._inverse_softplus(torch.expm1(value))

    def standardized_maps(self, raw: Tensor) -> Tensor:
        """Return constrained Z coordinates in channel-first layout."""
        if raw.ndim != 4 or raw.shape[1] != self.n_output_parameters:
            raise ValueError(
                f"Expected raw shape (B, {self.n_output_parameters}, X, Y), "
                f"found {tuple(raw.shape)}."
            )
        mean = self.parameter_mean[None, :, None, None]
        std = self.parameter_std[None, :, None, None]
        z = raw.clone()
        positive_indices = [
            *range(self.n_basis_components),
            self.n_basis_components + 1,
            self.n_basis_components + 2,
        ]
        amplitude_indices = positive_indices[: self.n_basis_components]
        fwhm_indices = positive_indices[self.n_basis_components :]
        if amplitude_indices:
            physical = torch.nn.functional.softplus(raw[:, amplitude_indices]) * std[:, amplitude_indices]
            z[:, amplitude_indices] = (physical - mean[:, amplitude_indices]) / std[:, amplitude_indices]
        physical_fwhm = torch.log1p(
            torch.nn.functional.softplus(raw[:, fwhm_indices])
        ) * std[:, fwhm_indices]
        z[:, fwhm_indices] = (
            physical_fwhm - mean[:, fwhm_indices]
        ) / std[:, fwhm_indices]
        return z

    def physical_channels(self, raw: Tensor) -> Tensor:
        z = self.standardized_maps(raw)
        return (
            z * self.parameter_std[None, :, None, None]
            + self.parameter_mean[None, :, None, None]
        )

    def standardize_physical_channels(self, physical: Tensor) -> Tensor:
        return (
            physical - self.parameter_mean[None, :, None, None]
        ) / self.parameter_std[None, :, None, None]

    def convert_teacher_physical_channels(self, teacher: Tensor) -> Tensor:
        """Convert forD amplitude units to WALINET-decoder amplitude units."""
        converted = teacher.clone()
        converted[:, : self.n_basis_components] = (
            converted[:, : self.n_basis_components]
            * self.teacher_to_model_amplitude_scale
        )
        return converted

    def raw_at_population_mean(self) -> Tensor:
        """Raw head bias whose physical output equals every stored mean."""
        result = torch.zeros_like(self.parameter_mean)
        n = self.n_basis_components
        result[:n] = self._inverse_softplus(
            self.parameter_mean[:n] / self.parameter_std[:n]
        )
        for index in (n + 1, n + 2):
            result[index] = self._inverse_log_softplus(
                self.parameter_mean[index] / self.parameter_std[index]
            )
        # Unconstrained coordinates equal their Z score; the mean is z=0.
        return result

    def forward(self, raw: Tensor) -> SpectralParameters:
        physical = self.physical_channels(raw)
        n = self.n_basis_components
        return SpectralParameters(
            amplitudes=physical[:, :n].movedim(1, -1),
            frequency_shift_hz=physical[:, n],
            lorentzian_fwhm_hz=physical[:, n + 1],
            gaussian_fwhm_hz=physical[:, n + 2],
            zero_order_phase_radians=physical[:, n + 3],
            first_order_phase_rad_per_hz=physical[:, n + 4],
        )

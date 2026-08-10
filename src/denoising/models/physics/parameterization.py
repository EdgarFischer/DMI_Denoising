"""Minimal physical constraints for decoder parameters."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .parameters import SpectralParameters


class MinimalPhysicalParameterization(nn.Module):
    """Convert raw maps using only physically necessary constraints.

    Amplitudes and both FWHM values are non-negative through softplus.
    Frequency shift and phases remain completely unbounded. No additional
    coordinate scaling, bounds, or priors are applied.
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
        lorentzian_fwhm_hz = torch.nn.functional.softplus(nuisance[:, 1])
        gaussian_fwhm_hz = torch.nn.functional.softplus(nuisance[:, 2])
        return SpectralParameters(
            amplitudes=amplitudes,
            frequency_shift_hz=nuisance[:, 0],
            lorentzian_fwhm_hz=lorentzian_fwhm_hz,
            gaussian_fwhm_hz=gaussian_fwhm_hz,
            zero_order_phase_radians=nuisance[:, 3],
            first_order_phase_rad_per_hz=nuisance[:, 4],
        )


class LCModelKernelParameterization(nn.Module):
    """LCModel-like baseline-free parameterization used by Phive.

    The common non-parametric lineshape is a positive, normalized spectral
    convolution kernel. Each basis component additionally receives its own
    positive Lorentzian FWHM and an unbounded relative frequency shift.
    """

    def __init__(
        self,
        n_basis_components: int,
        *,
        lineshape_kernel_size: int = 23,
        metabolite_shift_mean_hz: float = 0.0,
        metabolite_shift_std_hz: float = 1.0,
        metabolite_fwhm_mean_hz: float = 5.0,
        metabolite_fwhm_std_hz: float = 2.5,
        baseline_n_splines: int = 0,
        baseline_ford_to_model_scale: float = 1.0,
        baseline_real_mean: tuple[float, ...] | None = None,
        baseline_real_std: tuple[float, ...] | None = None,
        baseline_imag_mean: tuple[float, ...] | None = None,
        baseline_imag_std: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.n_basis_components = int(n_basis_components)
        self.lineshape_kernel_size = int(lineshape_kernel_size)
        self.baseline_n_splines = int(baseline_n_splines)
        scalar_statistics = torch.tensor(
            [metabolite_shift_mean_hz, metabolite_shift_std_hz,
             metabolite_fwhm_mean_hz, metabolite_fwhm_std_hz],
            dtype=torch.float32,
        )
        if not torch.isfinite(scalar_statistics).all():
            raise ValueError("Metabolite shift/FWHM statistics must be finite.")
        if metabolite_shift_std_hz <= 0 or metabolite_fwhm_std_hz <= 0:
            raise ValueError("Metabolite shift/FWHM standard deviations must be > 0.")
        if metabolite_fwhm_mean_hz <= 0:
            raise ValueError("Metabolite FWHM mean must be > 0.")
        self.register_buffer("metabolite_shift_mean_hz", scalar_statistics[0])
        self.register_buffer("metabolite_shift_std_hz", scalar_statistics[1])
        fwhm_raw_mean = self._inverse_softplus(scalar_statistics[2])
        # Choose raw-coordinate scale so dz=1 changes physical FWHM locally
        # by one requested physical standard deviation at z=0.
        fwhm_raw_std = scalar_statistics[3] / torch.sigmoid(fwhm_raw_mean)
        self.register_buffer("metabolite_fwhm_raw_mean", fwhm_raw_mean)
        self.register_buffer("metabolite_fwhm_raw_std", fwhm_raw_std)
        if not torch.isfinite(torch.tensor(baseline_ford_to_model_scale)):
            raise ValueError("baseline_ford_to_model_scale must be finite.")
        if baseline_ford_to_model_scale <= 0:
            raise ValueError("baseline_ford_to_model_scale must be > 0.")
        self.register_buffer(
            "baseline_ford_to_model_scale",
            torch.tensor(float(baseline_ford_to_model_scale)),
        )
        if self.n_basis_components < 1:
            raise ValueError("n_basis_components must be >= 1.")
        if self.lineshape_kernel_size < 3 or self.lineshape_kernel_size % 2 != 1:
            raise ValueError("lineshape_kernel_size must be an odd integer >= 3.")
        if self.baseline_n_splines < 0:
            raise ValueError("baseline_n_splines must be >= 0.")
        baseline_statistics = (
            baseline_real_mean, baseline_real_std,
            baseline_imag_mean, baseline_imag_std,
        )
        if self.baseline_n_splines:
            if any(values is None for values in baseline_statistics):
                raise ValueError(
                    "All four baseline mean/std arrays are required when "
                    "baseline_n_splines > 0."
                )
            if any(len(values) != self.baseline_n_splines for values in baseline_statistics):
                raise ValueError("Baseline mean/std lengths must match baseline_n_splines.")
            for name, values in zip(
                ("real_mean", "real_std", "imag_mean", "imag_std"),
                baseline_statistics,
            ):
                tensor = torch.tensor(values, dtype=torch.float32)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Baseline {name} must be finite.")
                if name.endswith("std") and not torch.all(tensor > 0):
                    raise ValueError(f"Baseline {name} must be positive.")
                self.register_buffer(f"baseline_{name}", tensor)

    @property
    def n_output_parameters(self) -> int:
        return (
            3 * self.n_basis_components
            + self.lineshape_kernel_size
            + 3
            + 2 * self.baseline_n_splines
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Checkpoints created immediately before lineshape Z-scaling do not
        # contain these config-derived buffers. Supply the current config
        # values so both old and new checkpoints remain strict-loadable.
        for name in (
            "metabolite_shift_mean_hz",
            "metabolite_shift_std_hz",
            "metabolite_fwhm_raw_mean",
            "metabolite_fwhm_raw_std",
        ):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @property
    def _sections(self) -> dict[str, slice | int]:
        n = self.n_basis_components
        kernel_start = 2 * n + 1 + n
        baseline_start = kernel_start + self.lineshape_kernel_size + 2
        return {
            "amplitudes": slice(0, n),
            "global_shift": n,
            "metabolite_shifts": slice(n + 1, 2 * n + 1),
            "metabolite_lorentz": slice(2 * n + 1, 3 * n + 1),
            "kernel": slice(kernel_start, kernel_start + self.lineshape_kernel_size),
            "phase0": kernel_start + self.lineshape_kernel_size,
            "phase1": kernel_start + self.lineshape_kernel_size + 1,
            "baseline_real": slice(
                baseline_start, baseline_start + self.baseline_n_splines
            ),
            "baseline_imag": slice(
                baseline_start + self.baseline_n_splines,
                baseline_start + 2 * self.baseline_n_splines,
            ),
        }

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        return value + torch.log(-torch.expm1(-value))

    def raw_at_initial_values(
        self, amplitudes: Tensor, lorentzian_fwhm_hz: float
    ) -> Tensor:
        if amplitudes.shape != (self.n_basis_components,):
            raise ValueError("Initial amplitudes have the wrong shape.")
        raw = amplitudes.new_zeros(self.n_output_parameters)
        sections = self._sections
        raw[sections["amplitudes"]] = self._inverse_softplus(amplitudes)
        # Both new parameter families start at z=0 (their physical means).
        raw[sections["metabolite_shifts"]] = 0.0
        raw[sections["metabolite_lorentz"]] = 0.0
        # Start centered, but avoid a saturated softmax so that the side bins
        # retain useful gradients from the first optimization step onward.
        kernel_logits = amplitudes.new_full((self.lineshape_kernel_size,), -2.0)
        kernel_logits[self.lineshape_kernel_size // 2] = 2.0
        raw[sections["kernel"]] = kernel_logits
        if self.baseline_n_splines:
            # Keep the population standard deviations as well-conditioned
            # output coordinates, but initialize the physical baseline at
            # exactly zero rather than at the (forD-derived) population mean.
            raw[sections["baseline_real"]] = (
                -self.baseline_real_mean / self.baseline_real_std
            ).to(device=raw.device, dtype=raw.dtype)
            raw[sections["baseline_imag"]] = (
                -self.baseline_imag_mean / self.baseline_imag_std
            ).to(device=raw.device, dtype=raw.dtype)
        return raw

    def forward(self, raw: Tensor) -> SpectralParameters:
        if raw.ndim != 4 or raw.shape[1] != self.n_output_parameters:
            raise ValueError(
                f"Expected raw shape (B, {self.n_output_parameters}, X, Y), "
                f"found {tuple(raw.shape)}."
            )
        sections = self._sections
        amplitudes = torch.nn.functional.softplus(
            raw[:, sections["amplitudes"]]
        ).movedim(1, -1)
        # Unbounded standardized coordinates; no clamp or saturating tanh.
        metabolite_shifts = (
            self.metabolite_shift_mean_hz
            + self.metabolite_shift_std_hz * raw[:, sections["metabolite_shifts"]]
        ).movedim(1, -1)
        metabolite_lorentz = torch.nn.functional.softplus(
            self.metabolite_fwhm_raw_mean
            + self.metabolite_fwhm_raw_std * raw[:, sections["metabolite_lorentz"]]
        ).movedim(1, -1)
        kernel = torch.softmax(raw[:, sections["kernel"]], dim=1).movedim(1, -1)
        leading = raw[:, 0]
        zeros = torch.zeros_like(leading)
        baseline_real = baseline_imag = None
        if self.baseline_n_splines:
            baseline_real = (
                (
                    raw[:, sections["baseline_real"]]
                    * self.baseline_real_std[None, :, None, None]
                    + self.baseline_real_mean[None, :, None, None]
                ) * self.baseline_ford_to_model_scale
            ).movedim(1, -1)
            baseline_imag = (
                (
                    raw[:, sections["baseline_imag"]]
                    * self.baseline_imag_std[None, :, None, None]
                    + self.baseline_imag_mean[None, :, None, None]
                ) * self.baseline_ford_to_model_scale
            ).movedim(1, -1)
        return SpectralParameters(
            amplitudes=amplitudes,
            frequency_shift_hz=raw[:, sections["global_shift"]],
            # The legacy global Voigt terms are deliberately inactive.
            lorentzian_fwhm_hz=zeros,
            gaussian_fwhm_hz=zeros,
            zero_order_phase_radians=raw[:, sections["phase0"]],
            first_order_phase_rad_per_hz=raw[:, sections["phase1"]],
            metabolite_frequency_shift_hz=metabolite_shifts,
            metabolite_lorentzian_fwhm_hz=metabolite_lorentz,
            lineshape_kernel=kernel,
            baseline_coefficients_real=baseline_real,
            baseline_coefficients_imag=baseline_imag,
        )


class StandardizedLCModelKernelParameterization(LCModelKernelParameterization):
    """Hybrid LCModel parameterization using legacy in-vivo Z-score scales.

    The calibrated amplitudes and the retained global shift/phase parameters
    use their population mean and standard deviation. New per-metabolite
    lineshape parameters remain in their direct constrained coordinates.
    """

    def __init__(
        self,
        n_basis_components: int,
        means: tuple[float, ...],
        stds: tuple[float, ...],
        *,
        lineshape_kernel_size: int = 23,
        metabolite_shift_mean_hz: float = 0.0,
        metabolite_shift_std_hz: float = 1.0,
        metabolite_fwhm_mean_hz: float = 5.0,
        metabolite_fwhm_std_hz: float = 2.5,
        baseline_n_splines: int = 0,
        baseline_ford_to_model_scale: float = 1.0,
        baseline_real_mean: tuple[float, ...] | None = None,
        baseline_real_std: tuple[float, ...] | None = None,
        baseline_imag_mean: tuple[float, ...] | None = None,
        baseline_imag_std: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(
            n_basis_components,
            lineshape_kernel_size=lineshape_kernel_size,
            metabolite_shift_mean_hz=metabolite_shift_mean_hz,
            metabolite_shift_std_hz=metabolite_shift_std_hz,
            metabolite_fwhm_mean_hz=metabolite_fwhm_mean_hz,
            metabolite_fwhm_std_hz=metabolite_fwhm_std_hz,
            baseline_n_splines=baseline_n_splines,
            baseline_ford_to_model_scale=baseline_ford_to_model_scale,
            baseline_real_mean=baseline_real_mean,
            baseline_real_std=baseline_real_std,
            baseline_imag_mean=baseline_imag_mean,
            baseline_imag_std=baseline_imag_std,
        )
        expected = self.n_basis_components + 5
        if len(means) != expected or len(stds) != expected:
            raise ValueError(f"Expected {expected} legacy parameter means/stds.")
        mean = torch.tensor(means, dtype=torch.float32)
        std = torch.tensor(stds, dtype=torch.float32)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Parameter means/stds must be finite.")
        if not torch.all(std > 0):
            raise ValueError("Every parameter standard deviation must be > 0.")
        self.register_buffer("legacy_parameter_mean", mean)
        self.register_buffer("legacy_parameter_std", std)

    def raw_at_initial_values(
        self, amplitudes: Tensor, lorentzian_fwhm_hz: float
    ) -> Tensor:
        raw = super().raw_at_initial_values(amplitudes, lorentzian_fwhm_hz)
        n = self.n_basis_components
        raw[self._sections["amplitudes"]] = self._inverse_softplus(
            self.legacy_parameter_mean[:n] / self.legacy_parameter_std[:n]
        ).to(device=raw.device, dtype=raw.dtype)
        # Z=0 maps to the calibrated population mean for global shift/phases.
        raw[self._sections["global_shift"]] = 0.0
        raw[self._sections["phase0"]] = 0.0
        raw[self._sections["phase1"]] = 0.0
        return raw

    def forward(self, raw: Tensor) -> SpectralParameters:
        parameters = super().forward(raw)
        n = self.n_basis_components
        mean = self.legacy_parameter_mean.to(device=raw.device, dtype=raw.dtype)
        std = self.legacy_parameter_std.to(device=raw.device, dtype=raw.dtype)
        amplitudes = (
            torch.nn.functional.softplus(raw[:, self._sections["amplitudes"]])
            * std[None, :n, None, None]
        ).movedim(1, -1)
        global_shift = (
            raw[:, self._sections["global_shift"]] * std[n] + mean[n]
        )
        phase0 = raw[:, self._sections["phase0"]] * std[n + 3] + mean[n + 3]
        phase1 = raw[:, self._sections["phase1"]] * std[n + 4] + mean[n + 4]
        return SpectralParameters(
            amplitudes=amplitudes,
            frequency_shift_hz=global_shift,
            lorentzian_fwhm_hz=parameters.lorentzian_fwhm_hz,
            gaussian_fwhm_hz=parameters.gaussian_fwhm_hz,
            zero_order_phase_radians=phase0,
            first_order_phase_rad_per_hz=phase1,
            metabolite_frequency_shift_hz=(
                parameters.metabolite_frequency_shift_hz
            ),
            metabolite_lorentzian_fwhm_hz=(
                parameters.metabolite_lorentzian_fwhm_hz
            ),
            lineshape_kernel=parameters.lineshape_kernel,
            baseline_coefficients_real=parameters.baseline_coefficients_real,
            baseline_coefficients_imag=parameters.baseline_coefficients_imag,
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

    def _standardized_base_maps(self, raw: Tensor) -> Tensor:
        expected = self.n_basis_components + 5
        if raw.ndim != 4 or raw.shape[1] != expected:
            raise ValueError(
                f"Expected raw shape (B, {expected}, X, Y), "
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
        physical_fwhm = (
            torch.nn.functional.softplus(raw[:, fwhm_indices])
            * std[:, fwhm_indices]
        )
        z[:, fwhm_indices] = (
            physical_fwhm - mean[:, fwhm_indices]
        ) / std[:, fwhm_indices]
        return z

    def standardized_maps(self, raw: Tensor) -> Tensor:
        """Return constrained Z coordinates in channel-first layout."""
        return self._standardized_base_maps(raw)

    def physical_channels(self, raw: Tensor) -> Tensor:
        z = self._standardized_base_maps(raw)
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
            result[index] = self._inverse_softplus(
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


class StandardizedVoigtBaselineParameterization(
    StandardizedPhysicalParameterization
):
    """Global standardized Voigt model with the exact complex forD baseline.

    The first ``n_basis + 5`` channels retain the existing in-vivo Z-score
    convention.  In particular, the two global Voigt widths are decoded from
    standardized coordinates to Lorentzian and Gaussian FWHM in Hz.  The
    additional baseline channels use their own forD coefficient statistics.
    """

    def __init__(
        self,
        n_basis_components: int,
        means: tuple[float, ...],
        stds: tuple[float, ...],
        *,
        teacher_to_model_amplitude_scale: float = 1.0,
        baseline_n_splines: int,
        baseline_ford_to_model_scale: float,
        baseline_real_mean: tuple[float, ...],
        baseline_real_std: tuple[float, ...],
        baseline_imag_mean: tuple[float, ...],
        baseline_imag_std: tuple[float, ...],
    ) -> None:
        super().__init__(
            n_basis_components,
            means,
            stds,
            teacher_to_model_amplitude_scale=teacher_to_model_amplitude_scale,
        )
        self.baseline_n_splines = int(baseline_n_splines)
        if self.baseline_n_splines < 1:
            raise ValueError("baseline_n_splines must be >= 1.")
        if not torch.isfinite(torch.tensor(baseline_ford_to_model_scale)):
            raise ValueError("baseline_ford_to_model_scale must be finite.")
        if baseline_ford_to_model_scale <= 0:
            raise ValueError("baseline_ford_to_model_scale must be > 0.")
        self.register_buffer(
            "baseline_ford_to_model_scale",
            torch.tensor(float(baseline_ford_to_model_scale)),
        )
        statistics = {
            "real_mean": baseline_real_mean,
            "real_std": baseline_real_std,
            "imag_mean": baseline_imag_mean,
            "imag_std": baseline_imag_std,
        }
        for name, values in statistics.items():
            if len(values) != self.baseline_n_splines:
                raise ValueError(
                    f"Baseline {name} length must match baseline_n_splines."
                )
            tensor = torch.tensor(values, dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Baseline {name} must be finite.")
            if name.endswith("std") and not torch.all(tensor > 0):
                raise ValueError(f"Baseline {name} must be positive.")
            self.register_buffer(f"baseline_{name}", tensor)

    @property
    def base_parameter_count(self) -> int:
        return self.n_basis_components + 5

    @property
    def n_output_parameters(self) -> int:
        return self.base_parameter_count + 2 * self.baseline_n_splines

    @property
    def _baseline_sections(self) -> tuple[slice, slice]:
        start = self.base_parameter_count
        return (
            slice(start, start + self.baseline_n_splines),
            slice(
                start + self.baseline_n_splines,
                start + 2 * self.baseline_n_splines,
            ),
        )

    def standardized_maps(self, raw: Tensor) -> Tensor:
        if raw.ndim != 4 or raw.shape[1] != self.n_output_parameters:
            raise ValueError(
                f"Expected raw shape (B, {self.n_output_parameters}, X, Y), "
                f"found {tuple(raw.shape)}."
            )
        # Baseline head outputs already are standardized coordinates.
        return torch.cat(
            (
                self._standardized_base_maps(raw[:, : self.base_parameter_count]),
                raw[:, self.base_parameter_count :],
            ),
            dim=1,
        )

    def raw_at_population_mean(self) -> Tensor:
        raw = self.parameter_mean.new_zeros(self.n_output_parameters)
        raw[: self.base_parameter_count] = super().raw_at_population_mean()
        real_section, imag_section = self._baseline_sections
        # Preserve the existing convention: initialize the physical baseline
        # at exactly zero, while retaining its population-standardized scale.
        raw[real_section] = -self.baseline_real_mean / self.baseline_real_std
        raw[imag_section] = -self.baseline_imag_mean / self.baseline_imag_std
        return raw

    def forward(self, raw: Tensor) -> SpectralParameters:
        base = super().forward(raw[:, : self.base_parameter_count])
        real_section, imag_section = self._baseline_sections
        baseline_real = (
            raw[:, real_section]
            * self.baseline_real_std[None, :, None, None]
            + self.baseline_real_mean[None, :, None, None]
        ) * self.baseline_ford_to_model_scale
        baseline_imag = (
            raw[:, imag_section]
            * self.baseline_imag_std[None, :, None, None]
            + self.baseline_imag_mean[None, :, None, None]
        ) * self.baseline_ford_to_model_scale
        return SpectralParameters(
            amplitudes=base.amplitudes,
            frequency_shift_hz=base.frequency_shift_hz,
            lorentzian_fwhm_hz=base.lorentzian_fwhm_hz,
            gaussian_fwhm_hz=base.gaussian_fwhm_hz,
            zero_order_phase_radians=base.zero_order_phase_radians,
            first_order_phase_rad_per_hz=base.first_order_phase_rad_per_hz,
            baseline_coefficients_real=baseline_real.movedim(1, -1),
            baseline_coefficients_imag=baseline_imag.movedim(1, -1),
        )

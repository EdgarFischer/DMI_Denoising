"""Spatially preserving 3D encoder with a fixed physical decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .basis_decoder import BaselineFreeBasisDecoder
from .parameterization import (
    MinimalPhysicalParameterization,
    StandardizedPhysicalParameterization,
)
from .parameters import SpectralParameters


@dataclass(frozen=True)
class PhysicsModelOutput:
    reconstruction: Tensor
    parameters: SpectralParameters
    raw_parameter_maps: Tensor
    standardized_parameter_maps: Tensor | None = None


class SpectralDownsamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        spectral_stride: int,
        spectral_kernel_size: int,
        spatial_kernel_size: int,
    ) -> None:
        super().__init__()
        kernel = (spectral_kernel_size, spatial_kernel_size, spatial_kernel_size)
        padding = tuple(value // 2 for value in kernel)
        normalization_groups = self._normalization_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=(spectral_stride, 1, 1),
                padding=padding,
                bias=True,
            ),
            nn.GroupNorm(normalization_groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                padding=padding,
                bias=True,
            ),
            nn.GroupNorm(normalization_groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    @staticmethod
    def _normalization_groups(channels: int) -> int:
        """Use up to four groups while keeping an integral group size."""
        for groups in range(min(4, int(channels)), 0, -1):
            if int(channels) % groups == 0:
                return groups
        return 1

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class PhysicsConv3D(nn.Module):
    """Predict per-voxel parameters and reconstruct with basis physics.

    Input and reconstruction have shape ``(B, 2, D0, D1, D2)``. The local
    ``spectral_axis`` identifies which of D0..D2 is frequency. Internally the
    tensor is arranged as ``(B, 2, F, X, Y)``. Spatial resolution is never
    downsampled; only the frequency dimension uses strides.
    """

    def __init__(
        self,
        physical_decoder: BaselineFreeBasisDecoder,
        *,
        input_n_timepoints: int,
        spectral_axis: int,
        features: tuple[int, ...],
        spectral_strides: tuple[int, ...],
        spectral_kernel_size: int = 5,
        spatial_kernel_size: int = 3,
        parameter_head_hidden_channels: int = 256,
        initial_reconstruction_rms: float = 0.025,
        initial_lorentzian_fwhm_hz: float = 5.0,
        initial_gaussian_fwhm_hz: float = 3.0,
        parameter_head_weight_std: float = 1e-3,
        basis_names: tuple[str, ...] | None = None,
        parameter_means: tuple[float, ...] | None = None,
        parameter_stds: tuple[float, ...] | None = None,
        teacher_to_model_amplitude_scale: float = 1.0,
        denoising_ppm_range: tuple[float, float] | None = None,
        ppm_reference: float = 4.65,
        hz_per_ppm: float | None = None,
    ) -> None:
        super().__init__()
        if spectral_axis not in (0, 1, 2):
            raise ValueError("spectral_axis must be 0, 1, or 2.")
        if len(features) == 0 or len(features) != len(spectral_strides):
            raise ValueError("features and spectral_strides must be non-empty and equal length.")
        if input_n_timepoints != physical_decoder.n_timepoints:
            raise ValueError(
                "Input spectrum length and decoder basis length differ: "
                f"{input_n_timepoints} vs {physical_decoder.n_timepoints}."
            )
        if initial_reconstruction_rms <= 0:
            raise ValueError("initial_reconstruction_rms must be > 0.")
        if initial_lorentzian_fwhm_hz <= 0 or initial_gaussian_fwhm_hz <= 0:
            raise ValueError("Initial FWHM values must be > 0.")
        if parameter_head_weight_std <= 0:
            raise ValueError("parameter_head_weight_std must be > 0.")

        self.physical_decoder = physical_decoder
        if (parameter_means is None) != (parameter_stds is None):
            raise ValueError("parameter_means and parameter_stds must be provided together.")
        if parameter_means is None:
            self.parameterization = MinimalPhysicalParameterization(
                physical_decoder.n_basis_components
            )
        else:
            self.parameterization = StandardizedPhysicalParameterization(
                physical_decoder.n_basis_components,
                parameter_means,
                parameter_stds,
                teacher_to_model_amplitude_scale=(
                    teacher_to_model_amplitude_scale
                ),
            )
        self.spectral_axis = int(spectral_axis)
        if denoising_ppm_range is not None:
            if len(denoising_ppm_range) != 2:
                raise ValueError("denoising_ppm_range must contain two values.")
            ppm_lower, ppm_upper = sorted(float(x) for x in denoising_ppm_range)
            if hz_per_ppm is None or hz_per_ppm <= 0:
                raise ValueError(
                    "A positive hz_per_ppm is required with denoising_ppm_range."
                )
            shifted_frequency_hz = torch.fft.fftshift(
                physical_decoder.frequency_axis_hz
            )
            ppm_axis = float(ppm_reference) - shifted_frequency_hz / float(hz_per_ppm)
            denoising_mask = (ppm_axis >= ppm_lower) & (ppm_axis <= ppm_upper)
            if not torch.any(denoising_mask):
                raise ValueError("denoising_ppm_range selects no frequency points.")
        else:
            denoising_mask = torch.ones(
                input_n_timepoints,
                dtype=torch.bool,
                device=physical_decoder.basis_fids.device,
            )
        self.register_buffer(
            "denoising_frequency_mask", denoising_mask, persistent=False
        )
        self.denoising_ppm_range = denoising_ppm_range
        self.basis_names = basis_names or tuple(
            str(index) for index in range(physical_decoder.n_basis_components)
        )

        blocks = []
        channels = 2
        latent_frequency = int(input_n_timepoints)
        for output_channels, stride in zip(features, spectral_strides):
            blocks.append(
                SpectralDownsamplingBlock(
                    channels,
                    int(output_channels),
                    spectral_stride=int(stride),
                    spectral_kernel_size=spectral_kernel_size,
                    spatial_kernel_size=spatial_kernel_size,
                )
            )
            channels = int(output_channels)
            latent_frequency = (latent_frequency + int(stride) - 1) // int(stride)
        self.encoder = nn.Sequential(*blocks)

        flattened_channels = channels * latent_frequency
        self.parameter_head = nn.Sequential(
            nn.Conv2d(
                flattened_channels,
                int(parameter_head_hidden_channels),
                kernel_size=1,
            ),
            nn.GroupNorm(
                SpectralDownsamplingBlock._normalization_groups(
                    int(parameter_head_hidden_channels)
                ),
                int(parameter_head_hidden_channels),
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                int(parameter_head_hidden_channels),
                self.parameterization.n_output_parameters,
                kernel_size=1,
            ),
        )
        self._initialize_parameter_head(
            reconstruction_rms=float(initial_reconstruction_rms),
            lorentzian_fwhm_hz=float(initial_lorentzian_fwhm_hz),
            gaussian_fwhm_hz=float(initial_gaussian_fwhm_hz),
            weight_std=float(parameter_head_weight_std),
        )

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        return value + torch.log(-torch.expm1(-value))

    @classmethod
    def _inverse_log_softplus(cls, value: Tensor) -> Tensor:
        return cls._inverse_softplus(torch.expm1(value))

    def _initial_amplitudes(
        self,
        *,
        reconstruction_rms: float,
        lorentzian_fwhm_hz: float,
        gaussian_fwhm_hz: float,
    ) -> Tensor:
        """Equalize basis response RMS and set the combined starting RMS."""
        decoder = self.physical_decoder
        n_basis = decoder.n_basis_components
        dtype = decoder.basis_fids.real.dtype
        device = decoder.basis_fids.device
        zeros = torch.zeros(n_basis, dtype=dtype, device=device)
        unit_parameters = SpectralParameters(
            amplitudes=torch.eye(n_basis, dtype=dtype, device=device),
            frequency_shift_hz=zeros,
            lorentzian_fwhm_hz=torch.full_like(
                zeros, lorentzian_fwhm_hz
            ),
            gaussian_fwhm_hz=torch.full_like(zeros, gaussian_fwhm_hz),
            zero_order_phase_radians=zeros,
            first_order_phase_rad_per_hz=zeros,
        )
        component_spectra = decoder.decode_spectra(unit_parameters)
        component_rms = component_spectra.abs().square().mean(-1).sqrt()
        relative_amplitudes = component_rms.clamp_min(
            torch.finfo(dtype).eps
        ).reciprocal()

        one = zeros[:1]
        combined_parameters = SpectralParameters(
            amplitudes=relative_amplitudes[None],
            frequency_shift_hz=one,
            lorentzian_fwhm_hz=torch.full_like(one, lorentzian_fwhm_hz),
            gaussian_fwhm_hz=torch.full_like(one, gaussian_fwhm_hz),
            zero_order_phase_radians=one,
            first_order_phase_rad_per_hz=one,
        )
        combined = decoder.decode_spectra(combined_parameters)
        combined_rms = combined.abs().square().mean().sqrt()
        return relative_amplitudes * (
            reconstruction_rms / combined_rms.clamp_min(torch.finfo(dtype).eps)
        )

    def _initialize_parameter_head(
        self,
        *,
        reconstruction_rms: float,
        lorentzian_fwhm_hz: float,
        gaussian_fwhm_hz: float,
        weight_std: float,
    ) -> None:
        final_layer = self.parameter_head[-1]
        with torch.no_grad():
            nn.init.normal_(final_layer.weight, mean=0.0, std=weight_std)
            if isinstance(
                self.parameterization, StandardizedPhysicalParameterization
            ):
                final_layer.bias.copy_(
                    self.parameterization.raw_at_population_mean().to(
                        device=final_layer.bias.device,
                        dtype=final_layer.bias.dtype,
                    )
                )
                return
            amplitudes = self._initial_amplitudes(
                reconstruction_rms=reconstruction_rms,
                lorentzian_fwhm_hz=lorentzian_fwhm_hz,
                gaussian_fwhm_hz=gaussian_fwhm_hz,
            ).to(device=final_layer.bias.device, dtype=final_layer.bias.dtype)
            bias = torch.zeros_like(final_layer.bias)
            n_basis = self.physical_decoder.n_basis_components
            bias[:n_basis] = self._inverse_softplus(amplitudes)
            bias[n_basis + 1] = self._inverse_log_softplus(
                bias.new_tensor(lorentzian_fwhm_hz)
            )
            bias[n_basis + 2] = self._inverse_log_softplus(
                bias.new_tensor(gaussian_fwhm_hz)
            )
            final_layer.bias.copy_(bias)
    def _to_internal(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError("Input must have shape (B, 2, D0, D1, D2).")
        spectral_tensor_axis = self.spectral_axis + 2
        spatial_tensor_axes = [axis for axis in (2, 3, 4) if axis != spectral_tensor_axis]
        return x.permute(0, 1, spectral_tensor_axis, *spatial_tensor_axes)

    def _from_internal(self, x: Tensor) -> Tensor:
        # x is (B, 2, F, X, Y); place F back at the configured local axis.
        if self.spectral_axis == 0:
            return x
        if self.spectral_axis == 1:
            return x.permute(0, 1, 3, 2, 4)
        return x.permute(0, 1, 3, 4, 2)

    def encode_parameters(
        self, x: Tensor
    ) -> tuple[SpectralParameters, Tensor]:
        internal = self._to_internal(x)
        features = self.encoder(internal)
        batch, channels, frequency, size_x, size_y = features.shape
        flattened = features.reshape(
            batch, channels * frequency, size_x, size_y
        )
        raw = self.parameter_head(flattened)
        return self.parameterization(raw), raw

    def forward(
        self, x: Tensor, *, return_parameters: bool = False
    ) -> Tensor | PhysicsModelOutput:
        parameters, raw = self.encode_parameters(x)
        spectra = self.physical_decoder.decode_spectra(parameters)
        internal_reconstruction = torch.stack(
            (spectra.real, spectra.imag), dim=1
        ).movedim(-1, 2)
        reconstruction = self._from_internal(internal_reconstruction).contiguous()
        if not bool(self.denoising_frequency_mask.all()):
            mask_shape = [1] * reconstruction.ndim
            mask_shape[self.spectral_axis + 2] = self.denoising_frequency_mask.numel()
            reconstruction = torch.where(
                self.denoising_frequency_mask.reshape(mask_shape),
                reconstruction,
                x,
            )
        if return_parameters:
            standardized = (
                self.parameterization.standardized_maps(raw)
                if isinstance(
                    self.parameterization, StandardizedPhysicalParameterization
                )
                else None
            )
            return PhysicsModelOutput(
                reconstruction, parameters, raw, standardized
            )
        return reconstruction

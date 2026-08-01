"""Central model construction for training and inference."""

from __future__ import annotations

import json
from pathlib import Path

from torch import nn

from denoising.models.physics.basis_factory import (
    decoder_from_walinet_simulation_config,
)
from denoising.models.physics.physics_conv3d import PhysicsConv3D
from denoising.models.unet2d import UNet2D
from denoising.models.unet3d import UNet3D


def build_model(cfg, sample_shape: tuple[int, ...]) -> nn.Module:
    """Build the configured model from one unbatched dataset sample shape."""
    spatial_dim = len(sample_shape) - 1
    in_channels = int(sample_shape[0])
    architecture = str(getattr(cfg.model, "architecture", "auto_unet"))

    if architecture == "auto_unet":
        architecture = "unet2d" if spatial_dim == 2 else "unet3d"
    if architecture == "unet2d":
        if spatial_dim != 2:
            raise ValueError("unet2d requires a two-dimensional dataset sample.")
        return UNet2D(in_channels, in_channels, cfg.model.features)
    if architecture == "unet3d":
        if spatial_dim != 3:
            raise ValueError("unet3d requires a three-dimensional dataset sample.")
        return UNet3D(in_channels, in_channels, cfg.model.features)
    if architecture == "physics_conv3d":
        physics = cfg.model.physics
        if physics is None:
            raise ValueError("model.physics is required for physics_conv3d.")
        decoder, basis_names = decoder_from_walinet_simulation_config(
            physics.simulation_config,
            dataset_name=physics.basis_dataset,
            active_metabolites_only=physics.active_metabolites_only,
            basis_components=physics.basis_components,
        )
        spectral_axis = int(cfg.data.spectral_axis)
        input_n_timepoints = int(sample_shape[spectral_axis + 1])
        parameter_means = parameter_stds = None
        teacher_to_model_amplitude_scale = 1.0
        if physics.parameter_statistics_path is not None:
            statistics_path = Path(physics.parameter_statistics_path)
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            expected_names = [
                *basis_names,
                "frequency_shift_hz",
                "lorentzian_fwhm_hz",
                "gaussian_fwhm_hz",
                "phase0_radians",
                "phase1_rad_per_hz",
            ]
            if list(statistics["parameter_names"]) != expected_names:
                raise ValueError(
                    "Parameter statistics order does not match model basis: "
                    f"{statistics_path}"
                )
            parameter_means = tuple(float(x) for x in statistics["mean"])
            parameter_stds = tuple(float(x) for x in statistics["std"])
            teacher_to_model_amplitude_scale = float(
                statistics.get("teacher_to_model_amplitude_scale", 1.0)
            )
        return PhysicsConv3D(
            decoder,
            input_n_timepoints=input_n_timepoints,
            spectral_axis=spectral_axis,
            features=cfg.model.features,
            spectral_strides=physics.spectral_strides,
            spectral_kernel_size=physics.spectral_kernel_size,
            spatial_kernel_size=physics.spatial_kernel_size,
            parameter_head_hidden_channels=(
                physics.parameter_head_hidden_channels
            ),
            initial_reconstruction_rms=physics.initial_reconstruction_rms,
            initial_lorentzian_fwhm_hz=physics.initial_lorentzian_fwhm_hz,
            initial_gaussian_fwhm_hz=physics.initial_gaussian_fwhm_hz,
            parameter_head_weight_std=physics.parameter_head_weight_std,
            basis_names=basis_names,
            parameter_means=parameter_means,
            parameter_stds=parameter_stds,
            teacher_to_model_amplitude_scale=(
                teacher_to_model_amplitude_scale
            ),
            denoising_ppm_range=physics.denoising_ppm_range,
            ppm_reference=physics.ppm_reference,
            hz_per_ppm=physics.hz_per_ppm,
        )
    raise ValueError(
        "Unknown model architecture "
        f"{architecture!r}; expected auto_unet, unet2d, unet3d, or "
        "physics_conv3d."
    )

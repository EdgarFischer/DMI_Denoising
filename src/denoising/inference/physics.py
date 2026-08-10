"""Volume inference for PhysicsConv3D models, including physical maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from denoising.models.factory import build_model


@dataclass(frozen=True)
class PhysicsParameterMaps:
    """Physical parameter maps in the original input-amplitude scale."""

    amplitudes: np.ndarray
    frequency_shift_hz: np.ndarray
    lorentzian_fwhm_hz: np.ndarray
    gaussian_fwhm_hz: np.ndarray
    zero_order_phase_radians: np.ndarray
    first_order_phase_rad_per_hz: np.ndarray
    metabolite_frequency_shift_hz: np.ndarray | None
    metabolite_lorentzian_fwhm_hz: np.ndarray | None
    lineshape_kernel: np.ndarray | None
    baseline_coefficients_real: np.ndarray | None
    baseline_coefficients_imag: np.ndarray | None
    raw_parameter_maps: np.ndarray
    standardized_parameter_maps: np.ndarray | None


@dataclass(frozen=True)
class PhysicsInferenceResult:
    """Reconstruction and parameter maps from one complex 4D FID volume."""

    reconstruction_fid: np.ndarray
    reconstruction_spectrum: np.ndarray
    baseline_spectrum: np.ndarray | None
    parameters: PhysicsParameterMaps
    basis_names: tuple[str, ...]
    frequency_axis_hz: np.ndarray
    normalization_scale: float
    metadata: dict[str, Any]


def _load_complex_fid(input_fid: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(input_fid, (str, Path)):
        path = Path(input_fid).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.load(path, mmap_mode="r")
    else:
        values = np.asarray(input_fid)
    if values.ndim != 4:
        raise ValueError(f"Expected FID shape (X,Y,Z,F), got {values.shape}")
    if not np.iscomplexobj(values):
        raise ValueError("Physics inference requires complex FID-domain input")
    return values


def _checkpoint_state(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    return checkpoint, state


def _channel_last(values: torch.Tensor) -> np.ndarray:
    if values.ndim != 4:
        raise ValueError(f"Expected channel-first (B,C,X,Y), got {tuple(values.shape)}")
    return values.permute(0, 2, 3, 1).detach().cpu().numpy()


def infer_physics_volume(
    *,
    cfg,
    checkpoint_path: str | Path,
    input_fid: str | Path | np.ndarray,
    device: str | torch.device | None = None,
    slice_batch_size: int = 1,
) -> PhysicsInferenceResult:
    """Infer a complete ``(X,Y,Z,F)`` FID volume and all physical maps.

    Normalization follows training exactly: when ``cfg.data.normalization`` is
    enabled, the complete input volume is divided once by its global maximum
    absolute FID value. After inference, reconstruction, amplitudes, baseline
    coefficients and baseline spectrum are multiplied by that same scalar.
    Frequency, linewidth, phase and normalized-kernel parameters are not
    amplitude-like and therefore remain unchanged.
    """
    if cfg.model.architecture != "physics_conv3d":
        raise ValueError("infer_physics_volume requires model.architecture=physics_conv3d")
    if tuple(cfg.data.image_axes) != (0, 1, 3) or cfg.data.channel_axis is not None:
        raise ValueError(
            "Physics volume inference currently requires image_axes=[0,1,3] "
            "and channel_axis=null"
        )
    if tuple(cfg.data.fourier_axes) != (3,):
        raise ValueError("Physics volume inference currently requires fourier_axes=[3]")
    if slice_batch_size < 1:
        raise ValueError("slice_batch_size must be at least 1")

    fid = _load_complex_fid(input_fid)
    scale = 1.0
    if bool(cfg.data.normalization):
        scale = float(np.max(np.abs(fid)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid global FID normalization scale: {scale}")
    normalized_fid = np.asarray(fid, dtype=np.complex64) / np.float32(scale)
    normalized_spectrum = np.fft.fftshift(
        np.fft.fft(normalized_fid, axis=-1), axes=-1
    ).astype(np.complex64)

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    size_x, size_y, size_z, n_frequency = normalized_spectrum.shape
    model = build_model(cfg, (2, size_x, size_y, n_frequency)).to(resolved_device)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint, state = _checkpoint_state(checkpoint_path, resolved_device)
    model.load_state_dict(state, strict=True)
    model.eval()

    reconstruction = np.empty_like(normalized_spectrum)
    baseline_spectrum = (
        np.empty_like(normalized_spectrum)
        if model.ford_baseline_design_matrix.shape[1] > 0
        else None
    )
    collected: dict[str, np.ndarray | None] | None = None

    with torch.inference_mode():
        for z_start in range(0, size_z, slice_batch_size):
            z_stop = min(z_start + slice_batch_size, size_z)
            spectra_batch = normalized_spectrum[:, :, z_start:z_stop, :].transpose(2, 0, 1, 3)
            network_batch = torch.from_numpy(
                np.stack((spectra_batch.real, spectra_batch.imag), axis=1)
            ).to(resolved_device)
            output = model(network_batch, return_parameters=True)
            reconstruction_batch = (
                output.reconstruction[:, 0] + 1j * output.reconstruction[:, 1]
            ).detach().cpu().numpy()
            reconstruction[:, :, z_start:z_stop, :] = reconstruction_batch.transpose(1, 2, 0, 3)

            parameters = output.parameters
            batch_maps: dict[str, np.ndarray | None] = {
                "amplitudes": parameters.amplitudes.detach().cpu().numpy(),
                "frequency_shift_hz": parameters.frequency_shift_hz.detach().cpu().numpy(),
                "lorentzian_fwhm_hz": parameters.lorentzian_fwhm_hz.detach().cpu().numpy(),
                "gaussian_fwhm_hz": parameters.gaussian_fwhm_hz.detach().cpu().numpy(),
                "zero_order_phase_radians": parameters.zero_order_phase_radians.detach().cpu().numpy(),
                "first_order_phase_rad_per_hz": parameters.first_order_phase_rad_per_hz.detach().cpu().numpy(),
                "metabolite_frequency_shift_hz": None if parameters.metabolite_frequency_shift_hz is None else parameters.metabolite_frequency_shift_hz.detach().cpu().numpy(),
                "metabolite_lorentzian_fwhm_hz": None if parameters.metabolite_lorentzian_fwhm_hz is None else parameters.metabolite_lorentzian_fwhm_hz.detach().cpu().numpy(),
                "lineshape_kernel": None if parameters.lineshape_kernel is None else parameters.lineshape_kernel.detach().cpu().numpy(),
                "baseline_coefficients_real": None if parameters.baseline_coefficients_real is None else parameters.baseline_coefficients_real.detach().cpu().numpy(),
                "baseline_coefficients_imag": None if parameters.baseline_coefficients_imag is None else parameters.baseline_coefficients_imag.detach().cpu().numpy(),
                "raw_parameter_maps": _channel_last(output.raw_parameter_maps),
                "standardized_parameter_maps": None if output.standardized_parameter_maps is None else _channel_last(output.standardized_parameter_maps),
            }
            if collected is None:
                collected = {}
                for name, values in batch_maps.items():
                    collected[name] = (
                        None
                        if values is None
                        else np.empty((size_x, size_y, size_z, *values.shape[3:]), dtype=values.dtype)
                    )
            for name, values in batch_maps.items():
                if values is not None:
                    collected[name][:, :, z_start:z_stop, ...] = values.transpose(1, 2, 0, *range(3, values.ndim))

            if baseline_spectrum is not None:
                baseline_real = parameters.baseline_coefficients_real @ model.ford_baseline_design_matrix.T
                baseline_imag = parameters.baseline_coefficients_imag @ model.ford_baseline_design_matrix.T
                if model.baseline_conjugate_subject_signals:
                    baseline_imag = -baseline_imag
                baseline_batch = (baseline_real + 1j * baseline_imag).detach().cpu().numpy()
                baseline_spectrum[:, :, z_start:z_stop, :] = baseline_batch.transpose(1, 2, 0, 3)

    assert collected is not None
    amplitude_scale = np.float32(scale)
    reconstruction *= amplitude_scale
    if baseline_spectrum is not None:
        baseline_spectrum *= amplitude_scale
    for name in ("amplitudes", "baseline_coefficients_real", "baseline_coefficients_imag"):
        if collected[name] is not None:
            collected[name] *= amplitude_scale

    reconstruction_fid = np.fft.ifft(
        np.fft.ifftshift(reconstruction, axes=-1), axis=-1
    ).astype(np.complex64)
    parameter_maps = PhysicsParameterMaps(**collected)
    frequency_axis_hz = np.fft.fftshift(
        np.fft.fftfreq(n_frequency, d=model.physical_decoder.dwell_time_seconds)
    )
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_val_loss": checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None,
        "device": str(resolved_device),
        "input_shape": list(fid.shape),
        "normalization_enabled": bool(cfg.data.normalization),
        "normalization_scale": scale,
        "amplitude_like_outputs_rescaled": [
            "reconstruction_fid",
            "reconstruction_spectrum",
            "amplitudes",
            "baseline_coefficients_real",
            "baseline_coefficients_imag",
            "baseline_spectrum",
        ],
    }
    return PhysicsInferenceResult(
        reconstruction_fid=reconstruction_fid,
        reconstruction_spectrum=reconstruction,
        baseline_spectrum=baseline_spectrum,
        parameters=parameter_maps,
        basis_names=tuple(model.basis_names),
        frequency_axis_hz=frequency_axis_hz,
        normalization_scale=scale,
        metadata=metadata,
    )

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List
import time

from scipy.io import loadmat, savemat
import numpy as np
import torch

from denoising.models.unet2d import UNet2D
from denoising.models.unet3d import UNet3D


def _resolve_ckpt_path(ckpt: Union[str, Path]) -> Path:
    p = Path(ckpt)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _load_fid_file(input_path: Union[str, Path]) -> np.ndarray:
    """
    Loads a single volume from .npy or CombinedCSI.mat.

    Supported raw input shapes:
      - 4D: (X,Y,Z,F)
      - 5D: (X,Y,Z,F,T)

    Returns a standardized complex array with shape:
      (X,Y,Z,F,T)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix == ".npy":
        arr = np.load(input_path)
    elif input_path.suffix == ".mat":
        mat = loadmat(input_path)
        arr = np.asarray(mat["csi"]["Data"][0, 0])
    else:
        raise ValueError(
            f"Unsupported input format: {input_path}. Expected .npy or .mat"
        )

    arr = np.asarray(arr)

    if arr.ndim == 4:
        arr = arr[..., np.newaxis]   # -> (X,Y,Z,F,1)
    elif arr.ndim == 5:
        pass
    else:
        raise ValueError(
            f"Expected 4D or 5D input, got shape {arr.shape}"
        )

    return arr.astype(np.complex64, copy=False)


def _normalize_fid_inplace(arr: np.ndarray) -> float:
    """
    Normalize in FID domain to max(|.|)=1. Returns scale used.
    """
    maxv = float(np.max(np.abs(arr)))
    if maxv > 0:
        arr /= maxv
    return maxv


def _apply_fft(arr: np.ndarray, fourier_axes: List[int]) -> np.ndarray:
    out = arr
    for ax in fourier_axes:
        out = np.fft.fft(out, axis=ax)
        out = np.fft.fftshift(out, axes=ax)
    return out


def _apply_ifft(arr: np.ndarray, fourier_axes: List[int]) -> np.ndarray:
    out = arr
    for ax in fourier_axes:
        out = np.fft.ifftshift(out, axes=ax)
        out = np.fft.ifft(out, axis=ax)
    return out


def _to_model_input(cfg, arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert standardized complex array with shape (X,Y,Z,F,T)
    to model input.

    Global axes are interpreted exactly as in training.

    Output:
      x_np:
        - without channel_axis: (N, 2, *spatial)
        - with channel_axis   : (N, 2*C, *spatial)

    where:
      spatial = axes selected by image_axes
      C       = size of channel_axis
      N       = product of all remaining leading axes
    """
    arr = np.asarray(arr)
    if arr.ndim != 5:
        raise ValueError(f"Expected standardized 5D array, got {arr.shape}")

    image_axes = tuple(cfg.data.image_axes)
    channel_axis = cfg.data.channel_axis

    if len(image_axes) not in (2, 3):
        raise ValueError(f"image_axes must have length 2 or 3, got {image_axes}")

    if channel_axis is not None and channel_axis in image_axes:
        raise ValueError("channel_axis must not be part of image_axes")

    network_axes = list(image_axes)
    if channel_axis is not None:
        network_axes.append(channel_axis)

    leading_axes = [ax for ax in range(arr.ndim) if ax not in network_axes]

    if channel_axis is None:
        # (leading..., *image_axes)
        perm = leading_axes + list(image_axes)
        arr_p = np.transpose(arr, axes=perm)

        leading_shape = arr_p.shape[:len(leading_axes)]
        spatial_shape = arr_p.shape[len(leading_axes):]
        N = int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1

        arr_p = arr_p.reshape(N, *spatial_shape)  # (N, *spatial)

        real = arr_p.real.astype(np.float32, copy=False)
        imag = arr_p.imag.astype(np.float32, copy=False)
        x_np = np.stack([real, imag], axis=1)     # (N, 2, *spatial)

    else:
        # (leading..., channel, *image_axes)
        perm = leading_axes + [channel_axis] + list(image_axes)
        arr_p = np.transpose(arr, axes=perm)

        n_lead = len(leading_axes)
        leading_shape = arr_p.shape[:n_lead]
        C = arr_p.shape[n_lead]
        spatial_shape = arr_p.shape[n_lead + 1:]
        N = int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1

        arr_p = arr_p.reshape(N, C, *spatial_shape)   # (N, C, *spatial)

        real = arr_p.real.astype(np.float32, copy=False)
        imag = arr_p.imag.astype(np.float32, copy=False)
        x_np = np.stack([real, imag], axis=1)         # (N, 2, C, *spatial)
        x_np = x_np.reshape(N, 2 * C, *spatial_shape) # (N, 2C, *spatial)

    shape_info = {
        "orig_shape": tuple(arr.shape),
        "image_axes": image_axes,
        "channel_axis": channel_axis,
        "leading_axes": tuple(leading_axes),
        "leading_shape": tuple(leading_shape),
        "spatial_shape": tuple(spatial_shape),
    }
    if channel_axis is not None:
        shape_info["channel_size"] = int(C)

    return x_np.astype(np.float32, copy=False), shape_info


def _from_model_output(y_np: np.ndarray, shape_info: Dict[str, Any]) -> np.ndarray:
    """
    Inverse of _to_model_input.

    Input:
      y_np:
        - without channel_axis: (N, 2, *spatial)
        - with channel_axis   : (N, 2*C, *spatial)

    Output:
      complex array with original standardized shape (X,Y,Z,F,T)
    """
    orig_shape = tuple(shape_info["orig_shape"])
    image_axes = tuple(shape_info["image_axes"])
    channel_axis = shape_info["channel_axis"]
    leading_axes = tuple(shape_info["leading_axes"])
    leading_shape = tuple(shape_info["leading_shape"])
    spatial_shape = tuple(shape_info["spatial_shape"])

    if channel_axis is None:
        if y_np.ndim != 2 + len(spatial_shape) or y_np.shape[1] != 2:
            raise ValueError(
                f"Expected y_np shape (N,2,*spatial), got {y_np.shape}"
            )

        y_complex = y_np[:, 0] + 1j * y_np[:, 1]           # (N, *spatial)
        y_complex = y_complex.reshape(*leading_shape, *spatial_shape)

        perm = list(leading_axes) + list(image_axes)
        inv_perm = np.argsort(perm)
        y_complex = np.transpose(y_complex, axes=inv_perm)

    else:
        C = int(shape_info["channel_size"])
        expected_c = 2 * C

        if y_np.ndim != 2 + len(spatial_shape) or y_np.shape[1] != expected_c:
            raise ValueError(
                f"Expected y_np shape (N,{expected_c},*spatial), got {y_np.shape}"
            )

        y_np = y_np.reshape(y_np.shape[0], 2, C, *spatial_shape)  # (N,2,C,*spatial)
        y_complex = y_np[:, 0] + 1j * y_np[:, 1]                  # (N,C,*spatial)
        y_complex = y_complex.reshape(*leading_shape, C, *spatial_shape)

        perm = list(leading_axes) + [channel_axis] + list(image_axes)
        inv_perm = np.argsort(perm)
        y_complex = np.transpose(y_complex, axes=inv_perm)

    if y_complex.shape != orig_shape:
        raise RuntimeError(
            f"Shape mismatch after reconstruction: got {y_complex.shape}, expected {orig_shape}"
        )

    return y_complex.astype(np.complex64, copy=False)


def infer(
    *,
    cfg,
    ckpt_path: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    device: Optional[Union[str, torch.device]] = None,
    save_input: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Single-file inference.

    Supported input raw shapes:
      - 4D: (X,Y,Z,F)
      - 5D: (X,Y,Z,F,T)

    Uses the same axis semantics as training:
      - cfg.data.image_axes
      - cfg.data.channel_axis
      - cfg.data.fourier_axes

    Returns:
      y_fid: complex64 array with same shape as standardized input
             i.e. 4D input returns 4D, 5D input returns 5D
      meta: dict
    """
    ckpt_path = _resolve_ckpt_path(ckpt_path)
    input_path = Path(input_path)

    # device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # ---- load input ----
    x_fid = _load_fid_file(input_path)   # standardized to (X,Y,Z,F,T)
    standardized_shape = x_fid.shape
    original_input_ndim = 4 if standardized_shape[-1] == 1 else 5

    # ---- normalize in FID domain ----
    do_norm = bool(getattr(cfg.data, "normalization", True))
    scale = 1.0
    if do_norm:
        scale = _normalize_fid_inplace(x_fid)

    # ---- working domain ----
    fourier_axes = list(getattr(cfg.data, "fourier_axes", ()))
    if len(fourier_axes) > 0:
        x_work = _apply_fft(x_fid, fourier_axes)
    else:
        x_work = x_fid

    # ---- to model input ----
    x_np, shape_info = _to_model_input(cfg, x_work)

    # ---- infer model dimensions ----
    in_channels = int(x_np.shape[1])
    spatial_dim = x_np.ndim - 2
    out_channels = in_channels

    if spatial_dim == 2:
        model = UNet2D(in_channels, out_channels, cfg.model.features).to(device).eval()
    elif spatial_dim == 3:
        model = UNet3D(in_channels, out_channels, cfg.model.features).to(device).eval()
    else:
        raise ValueError(f"Unsupported spatial_dim={spatial_dim}. Expected 2 or 3.")

    # ---- load checkpoint ----
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state", state)
    model.load_state_dict(state_dict, strict=True)

    # ---- run inference ----
    y_np = np.empty_like(x_np)
    t0 = time.time()

    with torch.no_grad():
        for i0 in range(0, x_np.shape[0], batch_size):
            i1 = min(i0 + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[i0:i1]).to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            y_np[i0:i1] = yb

    dt = time.time() - t0

    # ---- reconstruct working-domain volume ----
    y_work = _from_model_output(y_np, shape_info)

    # ---- invert FFT if needed ----
    if len(fourier_axes) > 0:
        y_fid = _apply_ifft(y_work, fourier_axes)
    else:
        y_fid = y_work

    # ---- restore original scale ----
    if do_norm and scale > 0:
        y_fid = y_fid * scale

    y_fid = y_fid.astype(np.complex64, copy=False)

    # ---- drop singleton T again if original input was 4D ----
    if original_input_ndim == 4:
        y_out = y_fid[..., 0]
    else:
        y_out = y_fid

    meta = {
        "checkpoint": str(ckpt_path),
        "input_path": str(input_path),
        "standardized_input_shape": list(standardized_shape),
        "output_shape": list(y_out.shape),
        "batch_size": int(batch_size),
        "seconds": float(dt),
        "device": str(device),
        "fid_normalization_maxabs": scale,
        "fourier_axes": fourier_axes,
        "fft_applied": len(fourier_axes) > 0,
        "working_domain": "FFT" if len(fourier_axes) > 0 else "FID",
        "returned_domain": "FID",
        "image_axes": list(cfg.data.image_axes),
        "channel_axis": cfg.data.channel_axis,
        "spatial_dim": int(spatial_dim),
        "in_channels": int(in_channels),
    }

    # ---- save ----
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npy":
            np.save(output_path, y_out)

        elif output_path.suffix == ".mat":
            if input_path.suffix != ".mat":
                raise ValueError(
                    "Saving to .mat is currently only supported when the input is also a .mat file."
                )

            mat = loadmat(input_path)
            mat["csi"]["Data"][0, 0] = y_out
            savemat(output_path, mat, long_field_names=True)

        else:
            raise ValueError(
                f"Unsupported output format: {output_path}. Expected .npy or .mat"
            )

        if save_input:
            np.save(output_path.with_name(output_path.stem + "_input.npy"), x_fid)

    return y_out, meta
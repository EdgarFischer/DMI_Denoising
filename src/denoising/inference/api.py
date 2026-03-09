from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List
import time
from scipy.io import loadmat, savemat

import numpy as np
import torch

from denoising.models.unet2d import UNet2D


def _resolve_ckpt_path(ckpt: Union[str, Path]) -> Path:
    p = Path(ckpt)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _load_fid_file(input_path: Union[str, Path]) -> np.ndarray:
    """
    Loads a single FID volume from a .npy file or CombinedCSI.mat.
    Expected shape: (X,Y,Z,t,T) complex or real.
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
            f"Unsupported input format: {input_path}. "
            f"Expected .npy or .mat"
        )

    if arr.ndim != 5:
        raise ValueError(f"Expected FID shape (X,Y,Z,t,T), got {arr.shape}")

    return np.asarray(arr)


def _normalize_fid_inplace(arr: np.ndarray) -> float:
    """
    Normalize in FID domain to max(|.|)=1. Returns scale used (maxv).
    """
    maxv = float(np.max(np.abs(arr)))
    if maxv > 0:
        arr /= maxv
    return maxv


def _apply_fft(arr: np.ndarray, fourier_axes: List[int]) -> np.ndarray:
    """
    Apply fft + fftshift on given axes.
    """
    out = arr
    for ax in fourier_axes:
        out = np.fft.fft(out, axis=ax)
        out = np.fft.fftshift(out, axes=ax)
    return out


def _apply_ifft(arr: np.ndarray, fourier_axes: List[int]) -> np.ndarray:
    """
    Apply inverse of fftshift+fft: ifftshift + ifft on given axes.
    """
    out = arr
    for ax in fourier_axes:
        out = np.fft.ifftshift(out, axes=ax)
        out = np.fft.ifft(out, axis=ax)
    return out


def _to_n2ft_from_5d(cfg, arr_5d: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert 5D complex array -> (N,2,F,T) float32 + shape_info.

    arr_5d is expected to be (X,Y,Z,*,*) with cfg.data.image_axes selecting the 2D plane (F,T),
    typically image_axes=[3,4].

    Output is (N,2,F,T) where N = product of remaining dims (X*Y*Z).
    """
    arr = np.asarray(arr_5d)
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D array, got {arr.shape}")

    ax_f, ax_T = cfg.data.image_axes
    if ax_f == ax_T:
        raise ValueError("image_axes must contain two distinct axes")

    orig_shape = arr.shape
    perm = [i for i in range(arr.ndim) if i not in (ax_f, ax_T)] + [ax_f, ax_T]
    arr2 = np.transpose(arr, axes=perm)  # (leading..., F, T)

    F = arr2.shape[-2]
    T = arr2.shape[-1]
    leading_shape = arr2.shape[:-2]
    N = int(np.prod(leading_shape))

    arr2 = arr2.reshape(N, F, T)  # (N,F,T)

    if np.iscomplexobj(arr2):
        x = np.stack([arr2.real, arr2.imag], axis=1)  # (N,2,F,T)
    else:
        x = np.stack([arr2, np.zeros_like(arr2)], axis=1)

    shape_info = {
        "orig_shape": orig_shape,
        "perm": perm,
        "leading_shape": leading_shape,
        "F": F,
        "T": T,
    }
    return x.astype(np.float32, copy=False), shape_info


def _from_n2ft_to_5d(y_np: np.ndarray, shape_info: Dict[str, Any]) -> np.ndarray:
    """
    Convert (N,2,F,T) float32 -> complex 5D with original shape.
    """
    if y_np.ndim != 4 or y_np.shape[1] != 2:
        raise ValueError(f"Expected (N,2,F,T), got {y_np.shape}")

    orig_shape = tuple(shape_info["orig_shape"])
    perm = list(shape_info["perm"])
    leading_shape = tuple(shape_info["leading_shape"])
    F = int(shape_info["F"])
    T = int(shape_info["T"])

    y_complex = y_np[:, 0] + 1j * y_np[:, 1]  # (N,F,T)
    y_complex = y_complex.reshape(*leading_shape, F, T)  # (leading...,F,T)

    inv_perm = np.argsort(perm)
    y_complex = np.transpose(y_complex, axes=inv_perm)

    if y_complex.shape != orig_shape:
        raise RuntimeError(f"Shape mismatch: got {y_complex.shape}, expected {orig_shape}")

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

    Loads FID from input_path with shape (X,Y,Z,t,T).
    Normalizes in FID domain (max abs = 1), then applies FFT if cfg.data.fourier_axes is set.
    Runs UNet2D in the (F,T) plane defined by cfg.data.image_axes.
    If FFT was applied, inverts it at the end and returns/saves FID again.

    Returns:
      y_fid: complex64 array with SAME shape as input
      meta: dict (NOT saved to disk)
    """
    ckpt_path = _resolve_ckpt_path(ckpt_path)
    input_path = Path(input_path)

    # device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # ---- load FID ----
    x_fid = _load_fid_file(input_path).astype(np.complex64, copy=False)
    in_shape = x_fid.shape

    # ---- normalize in FID domain (your convention) ----
    do_norm = bool(getattr(cfg.data, "normalization", True))

    scale = 1.0
    if do_norm:
        scale = _normalize_fid_inplace(x_fid)

    # ---- forward domain used for network ----
    fourier_axes = list(getattr(cfg.data, "fourier_axes", ()))
    if len(fourier_axes) > 0:
        x_work = _apply_fft(x_fid, fourier_axes)
    else:
        x_work = x_fid

    # ---- convert to (N,2,F,T) ----
    x_np, shape_info = _to_n2ft_from_5d(cfg, x_work)

    # ---- model ----
    model = UNet2D(2, 2, cfg.model.features).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state", state)
    model.load_state_dict(state_dict, strict=True)

    # ---- run ----
    y_np = np.empty_like(x_np)
    t0 = time.time()

    with torch.no_grad():
        for i0 in range(0, x_np.shape[0], batch_size):
            i1 = min(i0 + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[i0:i1]).to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            y_np[i0:i1] = yb

    dt = time.time() - t0

    # ---- reconstruct 5D in working domain ----
    y_work = _from_n2ft_to_5d(y_np, shape_info)

    # ---- invert FFT back to FID if applied ----
    if len(fourier_axes) > 0:
        y_fid = _apply_ifft(y_work, fourier_axes)
    else:
        y_fid = y_work

    # ---- restore original scale if input was normalized ----
    if do_norm and scale > 0:
        y_fid = y_fid * scale

    y_fid = y_fid.astype(np.complex64, copy=False)

    meta = {
        "checkpoint": str(ckpt_path),
        "input_path": str(input_path),
        "input_shape": list(in_shape),
        "output_shape": list(y_fid.shape),
        "batch_size": int(batch_size),
        "seconds": float(dt),
        "device": str(device),
        "fid_normalization_maxabs": scale,
        "fourier_axes": fourier_axes,
        "fft_applied": len(fourier_axes) > 0,
        "working_domain": "FFT" if len(fourier_axes) > 0 else "FID",
        "returned_domain": "FID",
    }

    # ---- optional save (ONLY the arrays; no meta json) ----
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npy":
            np.save(output_path, y_fid)

        elif output_path.suffix == ".mat":
            if input_path.suffix != ".mat":
                raise ValueError(
                    "Saving to .mat is currently only supported when the input is also a .mat file."
                )

            mat = loadmat(input_path)
            mat["csi"]["Data"][0, 0] = y_fid
            savemat(output_path, mat, long_field_names=True)

        else:
            raise ValueError(
                f"Unsupported output format: {output_path}. Expected .npy or .mat"
            )

        if save_input:
            np.save(output_path.with_name(output_path.stem + "_input.npy"), x_fid)

    return y_fid, meta
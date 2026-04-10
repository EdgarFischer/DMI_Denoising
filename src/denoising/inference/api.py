from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List
import time

from scipy.io import loadmat, savemat
import numpy as np
import torch

from denoising.models.unet2d import UNet2D
from denoising.models.unet3d import UNet3D
from denoising.data.patching import *


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

def _resolve_relative_strides(
    effective_patch_sizes: Tuple[int, ...],
    reference_patch_sizes,
    reference_strides,
) -> Tuple[int, ...]:
    """
    Interpret reference_strides relative to reference_patch_sizes
    and rescale them to the effective patch sizes of the current sample/view.

    Example:
        reference_patch_sizes = [64, 64, None]
        reference_strides     = [32, 32, None]
        -> relative strides   = [0.5, 0.5, None]

    If a current sample/view has effective patch sizes [64, 35, 96],
    this becomes [32, 18, 96].
    """
    resolved = []

    for p_eff, p_ref, s_ref in zip(
        effective_patch_sizes,
        reference_patch_sizes,
        reference_strides,
    ):
        if s_ref is None or p_ref is None:
            resolved.append(p_eff)
            continue

        frac = float(s_ref) / float(p_ref)
        stride_eff = max(1, int(round(frac * p_eff)))
        resolved.append(stride_eff)

    return tuple(resolved)





def _infer_single_view(
    *,
    cfg,
    model,
    x_fid: np.ndarray,
    batch_size: int,
    device,
    image_axes_override=None,
    inference_patch_sizes=None,
    inference_strides=None,
    weight_mode="average",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run inference for exactly one view and return output in original
    standardized shape (X,Y,Z,F,T), i.e. before dropping singleton T.
    """
    do_norm = bool(getattr(cfg.data, "normalization", True))

    x_fid = np.asarray(x_fid).copy()
    scale = 1.0
    if do_norm:
        scale = _normalize_fid_inplace(x_fid)

    fourier_axes = list(getattr(cfg.data, "fourier_axes", ()))
    if len(fourier_axes) > 0:
        x_work = _apply_fft(x_fid, fourier_axes)
    else:
        x_work = x_fid

    x_np, shape_info = _to_structured_model_input(
        cfg,
        x_work,
        image_axes_override=image_axes_override,
    )

    has_channel_axis = cfg.data.channel_axis is not None

    if has_channel_axis:
        if inference_patch_sizes is not None:
            channel_patch_size = inference_patch_sizes[0]
            if channel_patch_size is None:
                channel_size_for_model = int(x_np.shape[2])
            else:
                channel_size_for_model = int(channel_patch_size)
        else:
            channel_size_for_model = int(x_np.shape[2])

        in_channels = 2 * channel_size_for_model
        spatial_dim = x_np.ndim - 3
    else:
        in_channels = 2
        spatial_dim = x_np.ndim - 2

    if inference_patch_sizes is None:
        y_np = np.empty_like(x_np)

        with torch.no_grad():
            for i0 in range(0, x_np.shape[0], batch_size):
                i1 = min(i0 + batch_size, x_np.shape[0])

                xb_struct = x_np[i0:i1]

                if has_channel_axis:
                    xb = xb_struct.reshape(
                        xb_struct.shape[0],
                        xb_struct.shape[1] * xb_struct.shape[2],
                        *xb_struct.shape[3:]
                    )
                else:
                    xb = xb_struct

                xb = torch.from_numpy(xb).to(device, non_blocking=True)
                yb = model(xb).detach().cpu().numpy()

                if has_channel_axis:
                    yb = yb.reshape(
                        yb.shape[0],
                        2,
                        xb_struct.shape[2],
                        *yb.shape[2:]
                    )

                y_np[i0:i1] = yb
    else:
        if inference_strides is None:
            inference_strides = inference_patch_sizes

        expected_len = len(shape_info["image_axes"]) + (1 if has_channel_axis else 0)

        if len(inference_patch_sizes) != expected_len:
            raise ValueError(
                f"inference_patch_sizes must have length {expected_len}, got {len(inference_patch_sizes)}"
            )
        if len(inference_strides) != expected_len:
            raise ValueError(
                f"inference_strides must have length {expected_len}, got {len(inference_strides)}"
            )

        y_np = _run_model_on_patches_structured(
            x_np=x_np,
            model=model,
            device=device,
            batch_size=batch_size,
            inference_patch_sizes=inference_patch_sizes,
            inference_strides=inference_strides,
            weight_mode=weight_mode,
            has_channel_axis=has_channel_axis,
        )

    y_work = _from_structured_model_output(y_np, shape_info)

    if len(fourier_axes) > 0:
        y_fid = _apply_ifft(y_work, fourier_axes)
    else:
        y_fid = y_work

    if do_norm and scale > 0:
        y_fid = y_fid * scale

    y_fid = y_fid.astype(np.complex64, copy=False)

    meta = {
        "image_axes": list(shape_info["image_axes"]),
        "channel_axis": cfg.data.channel_axis,
        "spatial_dim": int(spatial_dim),
        "in_channels": int(in_channels),
        "fourier_axes": fourier_axes,
        "fft_applied": len(fourier_axes) > 0,
    }

    return y_fid, meta

def infer(
    *,
    cfg,
    ckpt_path: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    device: Optional[Union[str, torch.device]] = None,
    save_input: bool = False,
    inference_patch_sizes=None,
    inference_strides=None,
    weight_mode="average",
    multi_view_mode: str = "single",   # "single", "stack", "average"
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

    Multi-view inference is optional:
      - "single"  : use cfg.data.image_axes only (backward compatible)
      - "stack"   : infer all configured views and stack results along a new leading axis
      - "average" : infer all configured views and average in image space

    Returns:
      y_out : complex64 array
        - single / average:
            same shape as original input (4D or 5D)
        - stack:
            shape (V, ...) where V = number of inferred views
      meta : dict
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

    # ---- resolve patch-wise inference settings from cfg if not explicitly passed ----
    inference_cfg = getattr(cfg, "inference", None)
    patching_cfg = getattr(cfg, "patching", None)

    if (
        inference_patch_sizes is None
        and patching_cfg is not None
        and getattr(patching_cfg, "enabled", False)
    ):
        inference_patch_sizes = patching_cfg.patch_sizes

    if inference_strides is None and inference_cfg is not None:
        inference_strides = inference_cfg.patch_strides

    if inference_cfg is not None and weight_mode == "average":
        weight_mode = inference_cfg.weight_mode

    if multi_view_mode not in ("single", "stack", "average"):
        raise ValueError(
            f"multi_view_mode must be one of 'single', 'stack', 'average', "
            f"got {multi_view_mode}"
        )

    # ---- determine which views to infer ----
    view_sampling_cfg = getattr(cfg.data, "view_sampling", None)

    if multi_view_mode == "single":
        views = [tuple(cfg.data.image_axes)]
    else:
        if view_sampling_cfg is None or not view_sampling_cfg.enabled:
            views = [tuple(cfg.data.image_axes)]
        else:
            views = [tuple(v) for v in view_sampling_cfg.views]

    # ---- build model once ----
    has_channel_axis = cfg.data.channel_axis is not None

    if has_channel_axis:
        if inference_patch_sizes is not None:
            channel_patch_size = inference_patch_sizes[0]
            if channel_patch_size is None:
                # infer from raw standardized input using global channel axis
                channel_axis = cfg.data.channel_axis
                channel_size_for_model = int(x_fid.shape[channel_axis])
            else:
                channel_size_for_model = int(channel_patch_size)
        else:
            channel_axis = cfg.data.channel_axis
            channel_size_for_model = int(x_fid.shape[channel_axis])

        in_channels = 2 * channel_size_for_model
        spatial_dim = len(cfg.data.image_axes)
    else:
        in_channels = 2
        spatial_dim = len(cfg.data.image_axes)

    out_channels = in_channels

    if spatial_dim == 2:
        model = UNet2D(in_channels, out_channels, cfg.model.features).to(device).eval()
    elif spatial_dim == 3:
        model = UNet3D(in_channels, out_channels, cfg.model.features).to(device).eval()
    else:
        raise ValueError(f"Unsupported spatial_dim={spatial_dim}. Expected 2 or 3.")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state", state)
    model.load_state_dict(state_dict, strict=True)

    # ---- run per-view inference ----
    t0 = time.time()

    view_outputs = []
    view_metas = []

    for view in views:
        y_view, meta_view = _infer_single_view(
            cfg=cfg,
            model=model,
            x_fid=x_fid,
            batch_size=batch_size,
            device=device,
            image_axes_override=view,
            inference_patch_sizes=inference_patch_sizes,
            inference_strides=inference_strides,
            weight_mode=weight_mode,
        )
        view_outputs.append(y_view)
        view_metas.append(meta_view)

    stacked_views = np.stack(view_outputs, axis=0)  # (V, X,Y,Z,F,T)

    if multi_view_mode == "single":
        y_fid = stacked_views[0]
    elif multi_view_mode == "stack":
        y_fid = stacked_views
    elif multi_view_mode == "average":
        y_fid = stacked_views.mean(axis=0).astype(np.complex64)
    else:
        raise RuntimeError(f"Unexpected multi_view_mode={multi_view_mode}")

    dt = time.time() - t0

    # ---- drop singleton T again if original input was 4D ----
    if multi_view_mode == "stack":
        if original_input_ndim == 4:
            y_out = y_fid[..., 0]   # -> (V, X,Y,Z,F)
        else:
            y_out = y_fid
    else:
        if original_input_ndim == 4:
            y_out = y_fid[..., 0]   # -> (X,Y,Z,F)
        else:
            y_out = y_fid

    y_out = y_out.astype(np.complex64, copy=False)

    # ---- meta ----
    fourier_axes = list(getattr(cfg.data, "fourier_axes", ()))

    meta = {
        "checkpoint": str(ckpt_path),
        "input_path": str(input_path),
        "standardized_input_shape": list(standardized_shape),
        "output_shape": list(y_out.shape),
        "batch_size": int(batch_size),
        "seconds": float(dt),
        "device": str(device),
        "fourier_axes": fourier_axes,
        "fft_applied": len(fourier_axes) > 0,
        "working_domain": "FFT" if len(fourier_axes) > 0 else "FID",
        "returned_domain": "FID",
        "image_axes": list(cfg.data.image_axes),
        "channel_axis": cfg.data.channel_axis,
        "spatial_dim": int(spatial_dim),
        "in_channels": int(in_channels),
        "multi_view_mode": multi_view_mode,
        "views": [list(v) for v in views],
        "num_views": len(views),
        "per_view_meta": view_metas,
    }

    # ---- save ----
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npy":
            np.save(output_path, y_out)

        elif output_path.suffix == ".mat":
            if multi_view_mode == "stack":
                raise ValueError(
                    "Saving stacked multi-view output to .mat is currently not supported."
                )

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

def _from_structured_model_output(
    y_np: np.ndarray,
    shape_info: Dict[str, Any]
) -> np.ndarray:
    """
    Inverse of _to_structured_model_input.

    Input:
      y_np:
        - without channel_axis: (N, 2, *spatial)
        - with channel_axis   : (N, 2, C, *spatial)

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
        expected_ndim = 2 + len(spatial_shape)
        if y_np.ndim != expected_ndim or y_np.shape[1] != 2:
            raise ValueError(
                f"Expected y_np shape (N,2,*spatial), got {y_np.shape}"
            )

        # (N, 2, *spatial) -> (N, *spatial) complex
        y_complex = y_np[:, 0] + 1j * y_np[:, 1]

        # (N, *spatial) -> (*leading, *spatial)
        y_complex = y_complex.reshape(*leading_shape, *spatial_shape)

        # inverse of perm = leading_axes + image_axes
        perm = list(leading_axes) + list(image_axes)
        inv_perm = np.argsort(perm)
        y_complex = np.transpose(y_complex, axes=inv_perm)

    else:
        C = int(shape_info["channel_size"])
        expected_ndim = 3 + len(spatial_shape)

        if y_np.ndim != expected_ndim:
            raise ValueError(
                f"Expected y_np shape (N,2,C,*spatial), got {y_np.shape}"
            )
        if y_np.shape[1] != 2:
            raise ValueError(
                f"Expected second axis to have size 2 (real/imag), got {y_np.shape}"
            )
        if y_np.shape[2] != C:
            raise ValueError(
                f"Expected channel axis size {C}, got {y_np.shape[2]}"
            )

        # (N, 2, C, *spatial) -> (N, C, *spatial) complex
        y_complex = y_np[:, 0] + 1j * y_np[:, 1]

        # (N, C, *spatial) -> (*leading, C, *spatial)
        y_complex = y_complex.reshape(*leading_shape, C, *spatial_shape)

        # inverse of perm = leading_axes + [channel_axis] + image_axes
        perm = list(leading_axes) + [channel_axis] + list(image_axes)
        inv_perm = np.argsort(perm)
        y_complex = np.transpose(y_complex, axes=inv_perm)

    if y_complex.shape != orig_shape:
        raise RuntimeError(
            f"Shape mismatch after reconstruction: got {y_complex.shape}, expected {orig_shape}"
        )

    return y_complex.astype(np.complex64, copy=False)

def _to_structured_model_input(
    cfg,
    arr: np.ndarray,
    image_axes_override=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert standardized complex array with shape (X,Y,Z,F,T)
    to STRUCTURED model input.

    Global axes are interpreted exactly as in training unless
    image_axes_override is provided.

    Output:
      x_np:
        - without channel_axis: (N, 2, *spatial)
        - with channel_axis   : (N, 2, C, *spatial)

    where:
      spatial = axes selected by image_axes
      C       = size of channel_axis
      N       = product of all remaining leading axes
    """
    arr = np.asarray(arr)
    if arr.ndim != 5:
        raise ValueError(f"Expected standardized 5D array, got {arr.shape}")

    image_axes = (
        tuple(image_axes_override)
        if image_axes_override is not None
        else tuple(cfg.data.image_axes)
    )
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
        perm = leading_axes + list(image_axes)
        arr_p = np.transpose(arr, axes=perm)

        leading_shape = arr_p.shape[:len(leading_axes)]
        spatial_shape = arr_p.shape[len(leading_axes):]
        N = int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1

        arr_p = arr_p.reshape(N, *spatial_shape)

        real = arr_p.real.astype(np.float32, copy=False)
        imag = arr_p.imag.astype(np.float32, copy=False)
        x_np = np.stack([real, imag], axis=1)

    else:
        perm = leading_axes + [channel_axis] + list(image_axes)
        arr_p = np.transpose(arr, axes=perm)

        n_lead = len(leading_axes)
        leading_shape = arr_p.shape[:n_lead]
        C = arr_p.shape[n_lead]
        spatial_shape = arr_p.shape[n_lead + 1:]
        N = int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1

        arr_p = arr_p.reshape(N, C, *spatial_shape)

        real = arr_p.real.astype(np.float32, copy=False)
        imag = arr_p.imag.astype(np.float32, copy=False)
        x_np = np.stack([real, imag], axis=1)

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

def _run_model_on_patches_structured(
    x_np: np.ndarray,
    model,
    device,
    batch_size: int,
    inference_patch_sizes,
    inference_strides,
    weight_mode: str = "average",
    has_channel_axis: bool = False,
) -> np.ndarray:
    """
    Patch-wise inference on STRUCTURED representation.

    Input:
      x_np:
        - without channel_axis: (N, 2, *spatial)
        - with channel_axis   : (N, 2, C, *spatial)

    The configured patch_sizes / strides refer to the structured axes
    EXCLUDING the leading real/imag axis. Therefore we expand them here
    to full sample.ndim by prepending the real/imag axis.

    Strides are interpreted relative to the configured patch sizes, so
    e.g. patch_sizes=[64,64,None], patch_strides=[32,32,None] means
    a relative stride of 0.5 on the first two axes.
    """
    y_np = np.empty_like(x_np)

    with torch.no_grad():
        for n in range(x_np.shape[0]):
            sample = x_np[n]

            expanded_patch_sizes = (sample.shape[0],) + tuple(
                sample.shape[i + 1] if p is None else int(p)
                for i, p in enumerate(inference_patch_sizes)
            )

            effective_non_ri_patch_sizes = tuple(expanded_patch_sizes[1:])

            effective_non_ri_strides = _resolve_relative_strides(
                effective_patch_sizes=effective_non_ri_patch_sizes,
                reference_patch_sizes=tuple(inference_patch_sizes),
                reference_strides=tuple(inference_strides),
            )

            expanded_strides = (sample.shape[0],) + effective_non_ri_strides

            slices_list, patches, normalized_patch_sizes = generate_inference_patches(
                arr=sample,
                patch_sizes=expanded_patch_sizes,
                strides=expanded_strides,
                return_patches=True,
            )

            pred_patches = []

            for i0 in range(0, len(patches), batch_size):
                i1 = min(i0 + batch_size, len(patches))
                batch_struct = patches[i0:i1]

                batch_model_in = []
                patch_struct_shapes = []

                for patch in batch_struct:
                    patch_struct_shapes.append(patch.shape)

                    if has_channel_axis:
                        patch_model_in = patch.reshape(
                            patch.shape[0] * patch.shape[1],
                            *patch.shape[2:]
                        )
                    else:
                        patch_model_in = patch

                    batch_model_in.append(patch_model_in)

                xb = np.stack(batch_model_in, axis=0)
                xb = torch.from_numpy(xb).to(device, non_blocking=True)
                yb = model(xb).detach().cpu().numpy()

                for k in range(yb.shape[0]):
                    patch_shape = patch_struct_shapes[k]

                    if has_channel_axis:
                        pred_patch = yb[k].reshape(patch_shape)
                    else:
                        pred_patch = yb[k]

                    pred_patches.append(pred_patch)

            recon = reconstruct_from_patches(
                patches=pred_patches,
                slices_list=slices_list,
                output_shape=sample.shape,
                patch_sizes=normalized_patch_sizes,
                weight_mode=weight_mode,
            )

            y_np[n] = recon

    return y_np
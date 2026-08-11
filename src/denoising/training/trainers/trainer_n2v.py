# ───────────────────────── denoising/training/trainers/trainer_n2v.py ─────────────────────────
# Noise2Void / Noise2Self, flexible 2D/3D version

import os
import random
import logging
from typing import List

import numpy as np
import torch
import torch.nn.utils as tnn_utils
from torch.utils.data import DataLoader

from denoising.data.data_utils import (
    load_and_preprocess_data,
    load_spatial_masks_for_preprocessed_data,
)
from denoising.data.mrsi_nd_dataset import MRSiNDataset
from denoising.losses.n2v_loss import (
    masked_mse_loss,
    residual_standard_deviation,
    residual_variance_scaled_masked_mse_loss,
)
from denoising.models.factory import build_model

# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def prepare_dataset(
    folders: List[str],
    transform,
    num_samples: int,
    *,
    base_path: str,
    data_filename: str,
    fourier_axes,
    image_axes,
    channel_axis,
    masked_axes,
    fixed_indices,
    augmentation,
    normalization: bool,
    patching_enabled: bool,
    patch_sizes,
    view_sampling=None,
    spatial_mask_filename=None,
):
    data = load_and_preprocess_data(
        folder_names=folders,
        base_path=base_path,
        fourier_axes=list(fourier_axes),
        normalization=normalization,
        npy_name=data_filename,
    )

    spatial_mask = None
    if spatial_mask_filename is not None:
        spatial_mask = load_spatial_masks_for_preprocessed_data(
            folder_names=folders,
            base_path=base_path,
            mask_filename=spatial_mask_filename,
        )

    return MRSiNDataset(
        data=data,
        spatial_mask=spatial_mask,
        image_axes=tuple(image_axes),
        channel_axis=channel_axis,
        masked_axes=tuple(masked_axes),
        fixed_indices=fixed_indices,
        transform=transform,
        num_samples=num_samples,
        augmentation=augmentation,
        patching_enabled=patching_enabled,
        patch_sizes=tuple(patch_sizes),
        view_sampling=view_sampling,
    )


def sample_n2s_mask(shape, p=0.03, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.rand(shape, device=device) < p).float()


def _restrict_to_physics_denoising_window(inp, tgt, mask, model):
    """Restrict corruption and self-supervised loss to the configured ppm range."""
    frequency_mask = getattr(model, "denoising_frequency_mask", None)
    if frequency_mask is None or bool(frequency_mask.all()):
        return inp, mask
    spectral_axis = int(model.spectral_axis) + 2
    shape = [1] * inp.ndim
    shape[spectral_axis] = frequency_mask.numel()
    window = frequency_mask.reshape(shape).to(device=inp.device)
    inp = torch.where(window, inp, tgt)
    mask = mask.to(device=inp.device) * window.to(dtype=mask.dtype)
    return inp, mask


def _physics_reconstruction_loss(
    reconstruction, target, mask, model, cfg, epoch
):
    """Apply the configured physics data term without changing its mask mean."""
    if (
        cfg is None
        or not cfg.residual_variance_scaling
        or epoch <= cfg.residual_variance_warmup_epochs
    ):
        return masked_mse_loss(reconstruction, target, mask)
    return residual_variance_scaled_masked_mse_loss(
        reconstruction,
        target,
        mask,
        spectral_axis=model.spectral_axis,
        frequency_mask=getattr(model, "denoising_frequency_mask", None),
        epsilon=cfg.residual_std_epsilon,
    )


def physics_parameter_regularization(
    output, sampling_mask, spectral_axis, cfg, baseline_design_matrix=None
):
    """Return shift/FWHM priors and lineshape curvature regularization."""
    zero = output.reconstruction.new_zeros(())
    if cfg is None or not cfg.enabled:
        return zero, zero, zero, zero, zero
    parameters = output.parameters
    # Dataset N2V masks are already intersected with the spatial brain mask.
    # Collapse channel and frequency, leaving (B, X, Y).
    frequency_dim = int(spectral_axis) + 2
    voxel_mask = sampling_mask.bool().any(dim=1).any(dim=frequency_dim - 1)
    if not bool(voxel_mask.any()):
        return zero, zero, zero, zero, zero
    shifts = parameters.metabolite_frequency_shift_hz
    fwhm = parameters.metabolite_lorentzian_fwhm_hz
    shift_prior = zero
    fwhm_prior = zero
    if shifts is not None:
        shift_z2 = ((shifts - cfg.shift_mean_hz) / cfg.shift_std_hz).square()
        shift_prior = shift_z2[voxel_mask].mean()
    elif cfg.shift_weight > 0:
        # In the simple global-Voigt model there are no metabolite-specific
        # shifts. Apply the same explicitly configured physical prior only to
        # the single global frequency-shift map.
        global_shift = parameters.frequency_shift_hz
        global_shift_z2 = (
            (global_shift - cfg.shift_mean_hz) / cfg.shift_std_hz
        ).square()
        shift_prior = global_shift_z2[voxel_mask].mean()
    if fwhm is not None:
        fwhm_z2 = ((fwhm - cfg.fwhm_mean_hz) / cfg.fwhm_std_hz).square()
        fwhm_prior = fwhm_z2[voxel_mask].mean()
    elif cfg.fwhm_weight > 0:
        raise ValueError(
            "A positive metabolite FWHM prior requires the LCModel-kernel "
            "parameterization."
        )
    kernel = parameters.lineshape_kernel
    kernel_prior = zero
    if kernel is not None:
        kernel_second_difference = (
            kernel[..., :-2] - 2.0 * kernel[..., 1:-1] + kernel[..., 2:]
        )
        # ||D²k||²: sum over bins, then average over valid brain voxels.
        kernel_curvature = kernel_second_difference.square().sum(dim=-1)
        kernel_prior = kernel_curvature[voxel_mask].mean()
    elif cfg.kernel_curvature_weight > 0:
        raise ValueError(
            "A positive kernel curvature weight requires the LCModel-kernel "
            "parameterization."
        )
    baseline_curvature = zero
    real_coefficients = parameters.baseline_coefficients_real
    imag_coefficients = parameters.baseline_coefficients_imag
    if real_coefficients is not None or imag_coefficients is not None:
        if (
            real_coefficients is None
            or imag_coefficients is None
            or baseline_design_matrix is None
        ):
            raise ValueError(
                "Baseline curvature requires real/imaginary coefficients "
                "and the baseline design matrix."
            )
        baseline_real = real_coefficients @ baseline_design_matrix.T
        baseline_imag = imag_coefficients @ baseline_design_matrix.T
        real_d2 = (
            baseline_real[..., :-2]
            - 2.0 * baseline_real[..., 1:-1]
            + baseline_real[..., 2:]
        )
        imag_d2 = (
            baseline_imag[..., :-2]
            - 2.0 * baseline_imag[..., 1:-1]
            + baseline_imag[..., 2:]
        )
        # Complex squared curvature, averaged over frequency and valid voxels.
        per_voxel_baseline_curvature = (
            real_d2.square() + imag_d2.square()
        ).mean(dim=-1)
        baseline_curvature = per_voxel_baseline_curvature[voxel_mask].mean()
    voigt_nuisance_prior = zero
    if cfg.voigt_nuisance_weight > 0:
        standardized = output.standardized_parameter_maps
        if standardized is None:
            raise ValueError(
                "Voigt nuisance Z-score regularization requires standardized "
                "parameter maps."
            )
        n_basis = parameters.amplitudes.shape[-1]
        nuisance_z = standardized[:, n_basis : n_basis + 5].movedim(1, -1)
        if nuisance_z.shape[-1] != 5:
            raise ValueError("Expected five standardized Voigt nuisance maps.")
        # Sum E[z_i^2] over the five nuisance parameters so the configured
        # alpha applies equally to each term rather than being divided by 5.
        voigt_nuisance_prior = nuisance_z[voxel_mask].square().mean(dim=0).sum()
    return (
        shift_prior,
        fwhm_prior,
        kernel_prior,
        baseline_curvature,
        voigt_nuisance_prior,
    )


def physics_parameter_squared_sums(output, sampling_mask, spectral_axis):
    """Return per-metabolite physical squared sums and included voxel count."""
    frequency_dim = int(spectral_axis) + 2
    voxel_mask = sampling_mask.bool().any(dim=1).any(dim=frequency_dim - 1)
    shifts = output.parameters.metabolite_frequency_shift_hz
    fwhm = output.parameters.metabolite_lorentzian_fwhm_hz
    if shifts is None or fwhm is None or not bool(voxel_mask.any()):
        return None, None, 0
    return (
        shifts[voxel_mask].square().sum(dim=0),
        fwhm[voxel_mask].square().sum(dim=0),
        int(voxel_mask.sum().item()),
    )

def get_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state_dict):
    if not state_dict:
        return

    if "python_random_state" in state_dict:
        random.setstate(state_dict["python_random_state"])

    if "numpy_random_state" in state_dict:
        np.random.set_state(state_dict["numpy_random_state"])

    if "torch_random_state" in state_dict:
        torch.set_rng_state(state_dict["torch_random_state"])

    if torch.cuda.is_available() and "torch_cuda_random_state_all" in state_dict:
        torch.cuda.set_rng_state_all(state_dict["torch_cuda_random_state_all"])


# ---------------------------
# Main entry
# ---------------------------
def train(
    *,
    cfg,
    run_dir: str,
    checkpoint_dir: str,
    log_dir: str,
    transform_train=None,
    transform_val=None,
):
    logger = setup_logger(log_dir)

    # ----- Repro / mode -----
    seed = cfg.run.seed
    set_seed(seed)

    phive_mode = bool(getattr(cfg, "phive_mode", False))
    if phive_mode:
        self_mode = "phive"
    else:
        self_mode = getattr(cfg, "self_supervised_mode", None)
        if self_mode is None:
            self_mode = getattr(cfg.run, "self_supervised_mode", None)
        if self_mode is None:
            self_mode = "n2v"

    assert self_mode in ("n2v", "n2s", "phive"), (
        f"Unsupported training loss mode: {self_mode}"
    )

    logger.info(f"Start Training ({self_mode.upper()}) – Seed {seed}")
    if phive_mode:
        logger.info(
            "[phive] Blind-spot masking disabled; input equals target and "
            "MSE uses the configured physics ppm window in valid brain voxels."
        )

    # ----- Dataset params -----
    base_path = cfg.data.base_dir
    data_filename = cfg.data.data_filename
    image_axes = cfg.data.image_axes
    channel_axis = cfg.data.channel_axis
    fourier_axes = cfg.data.fourier_axes
    masked_axes = cfg.mask.masked_axes
    view_sampling = getattr(cfg.data, "view_sampling", None)
    spatial_mask_filename = getattr(cfg.data, "spatial_mask_filename", None)

    num_samples = cfg.data.num_samples
    val_samples = cfg.data.val_samples

    fixed_indices = getattr(cfg.data, "fixed_indices", None)
    do_norm = getattr(cfg.data, "normalization", True)
    patching_enabled = cfg.patching.enabled
    patch_sizes = cfg.patching.patch_sizes

    # ----- Build datasets -----
    train_ds = prepare_dataset(
        folders=list(cfg.data.train),
        transform=transform_train,
        num_samples=num_samples,
        base_path=base_path,
        data_filename=data_filename,
        fourier_axes=fourier_axes,
        image_axes=image_axes,
        channel_axis=channel_axis,
        masked_axes=masked_axes,
        fixed_indices=fixed_indices,
        augmentation=cfg.augmentation,
        normalization=do_norm,
        patching_enabled=patching_enabled,
        patch_sizes=patch_sizes,
        view_sampling=view_sampling,
        spatial_mask_filename=spatial_mask_filename,
    )

    val_ds = prepare_dataset(
        folders=list(cfg.data.val),
        transform=transform_val,
        num_samples=val_samples,
        base_path=base_path,
        data_filename=data_filename,
        fourier_axes=fourier_axes,
        image_axes=image_axes,
        channel_axis=channel_axis,
        masked_axes=masked_axes,
        fixed_indices=fixed_indices,
        augmentation=None,
        normalization=do_norm,
        patching_enabled=patching_enabled,
        patch_sizes=patch_sizes,
        view_sampling=view_sampling,
        spatial_mask_filename=spatial_mask_filename,
    )

    # ----- Device & dataloaders -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    batch_size = cfg.optim.batch_size
    num_workers = cfg.optim.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ----- Infer model dimensions from dataset -----
    sample_inp, sample_tgt, sample_mask = train_ds[0]
    in_channels = int(sample_inp.shape[0])
    out_channels = int(sample_tgt.shape[0])
    spatial_dim = sample_inp.ndim - 1

    logger.info(
        f"[dataset] sample_inp shape={tuple(sample_inp.shape)}, "
        f"sample_tgt shape={tuple(sample_tgt.shape)}, "
        f"sample_mask shape={tuple(sample_mask.shape)}"
    )
    logger.info(
        f"[model] inferred in_channels={in_channels}, "
        f"out_channels={out_channels}, spatial_dim={spatial_dim}"
    )

    # ----- Debug mask -----
    if self_mode == "n2v":
        try:
            _, _, mask_dbg = next(iter(train_loader))
            global_mask_fraction = mask_dbg.float().mean().item()

            if spatial_mask_filename is not None:
                # The N2V mask has already been intersected with the spatial
                # support mask by MRSiNDataset. Recover which spatial locations
                # are valid by collapsing channel and configured masked axes,
                # then evaluate the fraction only inside that support. Network
                # tensors have shape (B, C, *image_axes), while masked_axes are
                # indexed relative to image_axes.
                collapse_dims = tuple(
                    sorted({1, *(2 + int(axis) for axis in masked_axes)})
                )
                valid_spatial = mask_dbg.bool().any(dim=collapse_dims)
                values_per_valid_location = 1
                for dim in collapse_dims:
                    values_per_valid_location *= int(mask_dbg.shape[dim])
                valid_count = int(valid_spatial.sum().item()) * values_per_valid_location
                inside_mask_fraction = (
                    float(mask_dbg.sum().item()) / valid_count
                    if valid_count > 0
                    else float("nan")
                )
                logger.info(
                    f"[DEBUG] N2V mask shape {tuple(mask_dbg.shape)}, "
                    f"fraction_inside_spatial_mask={inside_mask_fraction:.4f}, "
                    f"global_mean={global_mask_fraction:.4f}"
                )
            else:
                logger.info(
                    f"[DEBUG] N2V mask shape {tuple(mask_dbg.shape)}, "
                    f"fraction={global_mask_fraction:.4f}"
                )
        except StopIteration:
            logger.warning("[DEBUG] Train Loader leer – keine Maske.")
        except Exception as e:
            logger.warning(f"[DEBUG] Mask debug failed: {e}")

    # ----- Model / optim / scheduler -----
    model = build_model(cfg, tuple(sample_inp.shape)).to(device)
    logger.info(
        f"[model] Using {cfg.model.architecture}: "
        f"{model.__class__.__name__}"
    )
    if hasattr(model, "denoising_frequency_mask"):
        logger.info(
            "[model] Physics denoising window %s ppm: %d/%d spectral points",
            model.denoising_ppm_range,
            int(model.denoising_frequency_mask.sum().item()),
            int(model.denoising_frequency_mask.numel()),
        )

    regularization_cfg = getattr(cfg, "parameter_regularization", None)
    physics_data_loss_cfg = getattr(cfg, "physics_data_loss", None)
    if (
        physics_data_loss_cfg is not None
        and physics_data_loss_cfg.residual_variance_scaling
    ):
        logger.info(
            "[physics data loss] residual variance scaling enabled: "
            "sigma=std_f(abs(prediction-target)) in the physics ppm window, "
            "detached; epsilon=%g; unweighted-MSE warm-up epochs=%d; "
            "scaling starts at epoch %d",
            physics_data_loss_cfg.residual_std_epsilon,
            physics_data_loss_cfg.residual_variance_warmup_epochs,
            physics_data_loss_cfg.residual_variance_warmup_epochs + 1,
        )
    regularization_enabled = bool(
        regularization_cfg is not None and regularization_cfg.enabled
    )
    if regularization_enabled and cfg.model.architecture != "physics_conv3d":
        raise ValueError("Parameter regularization is only supported by physics_conv3d")
    if regularization_enabled:
        logger.info(
            "[regularization] shift: weight=%g, mean=%g Hz, std=%g Hz; "
            "FWHM: weight=%g, mean=%g Hz, std=%g Hz; "
            "normalized-kernel curvature: weight=%g; "
            "baseline curvature: weight=%g; "
            "global Voigt nuisance Z-score: weight=%g",
            regularization_cfg.shift_weight,
            regularization_cfg.shift_mean_hz,
            regularization_cfg.shift_std_hz,
            regularization_cfg.fwhm_weight,
            regularization_cfg.fwhm_mean_hz,
            regularization_cfg.fwhm_std_hz,
            regularization_cfg.kernel_curvature_weight,
            regularization_cfg.baseline_curvature_weight,
            regularization_cfg.voigt_nuisance_weight,
        )

    optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    def lr_lambda(epoch: int):
        lr = cfg.optim.lr * (cfg.optim.factor ** (epoch // cfg.optim.step_size))
        lr = max(lr, cfg.optim.min_lr)
        return lr / cfg.optim.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ----- Resume / pretrained -----
    # Two modes:
    #   1) resume_training=True  -> restore epoch, best_val, optimizer, scheduler, RNG
    #   2) pretrained_ckpt only  -> load weights, optionally optimizer
    resume_training = bool(getattr(cfg, "resume_training", False))
    ckpt_path = getattr(cfg, "resume_ckpt", "" if not resume_training else "")

    if ckpt_path == "":
        ckpt_path = getattr(cfg, "pretrained_ckpt", "")
    if ckpt_path == "":
        ckpt_path = getattr(cfg.optim, "pretrained_ckpt", "") if hasattr(cfg, "optim") else ""

    start_epoch = 0
    best_val = float("inf")

    if ckpt_path and not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Configured pretrained checkpoint does not exist: {ckpt_path}"
        )

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state", ckpt)
        strict_flag = getattr(cfg, "pretrained_strict", True)

        model.load_state_dict(state_dict, strict=strict_flag)
        logger.info(f"Gewichte geladen aus {ckpt_path}")

        if resume_training:
            # Resume full training state as exactly as possible
            if "optimizer_state" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optimizer_state"])
                    logger.info("Optimizer-State mitgeladen.")
                except Exception as e:
                    logger.warning(f"Optimizer-State konnte nicht geladen werden: {e}")

            if "scheduler_state" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                    logger.info("Scheduler-State mitgeladen.")
                except Exception as e:
                    logger.warning(f"Scheduler-State konnte nicht geladen werden: {e}")

            start_epoch = int(ckpt.get("epoch", 0))
            best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))

            if "rng_state" in ckpt:
                try:
                    set_rng_state(ckpt["rng_state"])
                    logger.info("RNG-State mitgeladen.")
                except Exception as e:
                    logger.warning(f"RNG-State konnte nicht geladen werden: {e}")

            logger.info(
                f"Resume training from epoch {start_epoch} "
                f"(next epoch will be {start_epoch + 1}), best_val={best_val:.4e}"
            )
        else:
            if getattr(cfg, "load_optimizer_from_pretrained", False) and "optimizer_state" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optimizer_state"])
                    logger.info("Optimizer-State mitgeladen.")
                except Exception as e:
                    logger.warning(f"Optimizer-State konnte nicht geladen werden: {e}")

    # ----- Checkpoints -----
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "best.pt")
    last_ckpt = os.path.join(checkpoint_dir, "last.pt")

    epochs = cfg.optim.epochs

    if resume_training and start_epoch >= epochs:
        logger.info(
            f"Checkpoint epoch {start_epoch} is already >= configured epochs {epochs}. "
            f"Nothing to do."
        )
        print(
            f"Checkpoint epoch {start_epoch} is already >= configured epochs {epochs}. "
            f"Nothing to do."
        )
        return

    # ----- Training loop -----
    for epoch in range(start_epoch + 1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        running = 0.0
        running_data = running_shift = running_fwhm = running_kernel = 0.0
        running_baseline = running_voigt_nuisance = 0.0
        running_mse = running_sigma_loss = running_sigma_median = 0.0
        train_shift_squares = train_fwhm_squares = None
        train_parameter_voxels = 0

        for inp, tgt, mask_n2v in train_loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            mask_n2v = mask_n2v.to(device, non_blocking=True)

            if phive_mode:
                # Keep the input completely uncorrupted. Reuse only the
                # frequency-window part of this helper to define where the
                # direct reconstruction loss is evaluated.
                _, phive_loss_mask = _restrict_to_physics_denoising_window(
                    inp, tgt, mask_n2v, model
                )
                model_output = model(inp, return_parameters=regularization_enabled)
                reconstruction = (
                    model_output.reconstruction
                    if regularization_enabled else model_output
                )
                data_loss = _physics_reconstruction_loss(
                    reconstruction, tgt, phive_loss_mask, model,
                    physics_data_loss_cfg, epoch,
                )
                regularization_mask = phive_loss_mask
            elif self_mode == "n2s":
                mask = sample_n2s_mask(inp[:, :1].shape, p=0.03, device=device)
                inp, mask = _restrict_to_physics_denoising_window(
                    inp, tgt, mask, model
                )
                inp_masked = inp * (1 - mask)
                model_output = model(
                    inp_masked, return_parameters=regularization_enabled
                )
                reconstruction = (
                    model_output.reconstruction
                    if regularization_enabled else model_output
                )
                data_loss = _physics_reconstruction_loss(
                    reconstruction, tgt, mask, model, physics_data_loss_cfg,
                    epoch,
                )
                regularization_mask = mask
            else:
                inp, mask_n2v = _restrict_to_physics_denoising_window(
                    inp, tgt, mask_n2v, model
                )
                model_output = model(inp, return_parameters=regularization_enabled)
                reconstruction = (
                    model_output.reconstruction
                    if regularization_enabled else model_output
                )
                data_loss = _physics_reconstruction_loss(
                    reconstruction, tgt, mask_n2v, model,
                    physics_data_loss_cfg, epoch,
                )
                regularization_mask = mask_n2v

            with torch.no_grad():
                plain_mse = masked_mse_loss(
                    reconstruction, tgt, regularization_mask
                )
                if (
                    physics_data_loss_cfg is not None
                    and physics_data_loss_cfg.residual_variance_scaling
                ):
                    sigma_loss_metric = residual_variance_scaled_masked_mse_loss(
                        reconstruction,
                        tgt,
                        regularization_mask,
                        spectral_axis=model.spectral_axis,
                        frequency_mask=getattr(
                            model, "denoising_frequency_mask", None
                        ),
                        epsilon=physics_data_loss_cfg.residual_std_epsilon,
                    )
                    sigma_metric = residual_standard_deviation(
                        reconstruction,
                        tgt,
                        spectral_axis=model.spectral_axis,
                        frequency_mask=getattr(
                            model, "denoising_frequency_mask", None
                        ),
                        epsilon=physics_data_loss_cfg.residual_std_epsilon,
                    ).median()
                else:
                    sigma_loss_metric = plain_mse
                    sigma_metric = plain_mse.new_zeros(())

            shift_prior, fwhm_prior, kernel_prior, baseline_prior, voigt_nuisance_prior = physics_parameter_regularization(
                model_output, regularization_mask, model.spectral_axis,
                regularization_cfg, model.ford_baseline_design_matrix,
            ) if regularization_enabled else (
                data_loss.new_zeros(()), data_loss.new_zeros(()),
                data_loss.new_zeros(()), data_loss.new_zeros(()),
                data_loss.new_zeros(()),
            )
            loss = (
                data_loss
                + (regularization_cfg.shift_weight * shift_prior if regularization_enabled else 0.0)
                + (regularization_cfg.fwhm_weight * fwhm_prior if regularization_enabled else 0.0)
                + (regularization_cfg.kernel_curvature_weight * kernel_prior if regularization_enabled else 0.0)
                + (regularization_cfg.baseline_curvature_weight * baseline_prior if regularization_enabled else 0.0)
                + (regularization_cfg.voigt_nuisance_weight * voigt_nuisance_prior if regularization_enabled else 0.0)
            )

            optim.zero_grad()
            loss.backward()
            tnn_utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += loss.item() * inp.size(0)
            running_data += data_loss.item() * inp.size(0)
            running_shift += shift_prior.item() * inp.size(0)
            running_fwhm += fwhm_prior.item() * inp.size(0)
            running_kernel += kernel_prior.item() * inp.size(0)
            running_baseline += baseline_prior.item() * inp.size(0)
            running_voigt_nuisance += voigt_nuisance_prior.item() * inp.size(0)
            running_mse += plain_mse.item() * inp.size(0)
            running_sigma_loss += sigma_loss_metric.item() * inp.size(0)
            running_sigma_median += sigma_metric.item() * inp.size(0)
            if regularization_enabled:
                shift_squares, fwhm_squares, parameter_voxels = (
                    physics_parameter_squared_sums(
                        model_output, regularization_mask, model.spectral_axis
                    )
                )
                if parameter_voxels:
                    train_shift_squares = (
                        shift_squares if train_shift_squares is None
                        else train_shift_squares + shift_squares
                    )
                    train_fwhm_squares = (
                        fwhm_squares if train_fwhm_squares is None
                        else train_fwhm_squares + fwhm_squares
                    )
                    train_parameter_voxels += parameter_voxels

        avg_train = running / len(train_loader.dataset)
        avg_train_data = running_data / len(train_loader.dataset)
        avg_train_shift = running_shift / len(train_loader.dataset)
        avg_train_fwhm = running_fwhm / len(train_loader.dataset)
        avg_train_kernel = running_kernel / len(train_loader.dataset)
        avg_train_baseline = running_baseline / len(train_loader.dataset)
        avg_train_voigt_nuisance = running_voigt_nuisance / len(train_loader.dataset)
        avg_train_mse = running_mse / len(train_loader.dataset)
        avg_train_sigma_loss = running_sigma_loss / len(train_loader.dataset)
        avg_train_sigma_median = running_sigma_median / len(train_loader.dataset)
        shift_weight = regularization_cfg.shift_weight if regularization_enabled else 0.0
        fwhm_weight = regularization_cfg.fwhm_weight if regularization_enabled else 0.0
        kernel_weight = regularization_cfg.kernel_curvature_weight if regularization_enabled else 0.0
        baseline_weight = regularization_cfg.baseline_curvature_weight if regularization_enabled else 0.0
        voigt_nuisance_weight = regularization_cfg.voigt_nuisance_weight if regularization_enabled else 0.0
        avg_train_shift_weighted = shift_weight * avg_train_shift
        avg_train_fwhm_weighted = fwhm_weight * avg_train_fwhm
        avg_train_kernel_weighted = kernel_weight * avg_train_kernel
        avg_train_baseline_weighted = baseline_weight * avg_train_baseline
        avg_train_voigt_nuisance_weighted = voigt_nuisance_weight * avg_train_voigt_nuisance

        # ---- VALID ----
        model.eval()
        running = 0.0
        running_data = running_shift = running_fwhm = running_kernel = 0.0
        running_baseline = running_voigt_nuisance = 0.0
        running_mse = running_sigma_loss = running_sigma_median = 0.0
        val_shift_squares = val_fwhm_squares = None
        val_parameter_voxels = 0
        with torch.no_grad():
            for inp, tgt, mask_n2v in val_loader:
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                mask_n2v = mask_n2v.to(device, non_blocking=True)

                if phive_mode:
                    _, phive_loss_mask = _restrict_to_physics_denoising_window(
                        inp, tgt, mask_n2v, model
                    )
                    model_output = model(inp, return_parameters=regularization_enabled)
                    reconstruction = (
                        model_output.reconstruction
                        if regularization_enabled else model_output
                    )
                    data_loss = _physics_reconstruction_loss(
                        reconstruction, tgt, phive_loss_mask, model,
                        physics_data_loss_cfg, epoch,
                    )
                    regularization_mask = phive_loss_mask
                elif self_mode == "n2s":
                    mask = sample_n2s_mask(inp[:, :1].shape, p=0.03, device=device)
                    inp, mask = _restrict_to_physics_denoising_window(
                        inp, tgt, mask, model
                    )
                    inp_masked = inp * (1 - mask)
                    model_output = model(
                        inp_masked, return_parameters=regularization_enabled
                    )
                    reconstruction = (
                        model_output.reconstruction
                        if regularization_enabled else model_output
                    )
                    data_loss = _physics_reconstruction_loss(
                        reconstruction, tgt, mask, model, physics_data_loss_cfg,
                        epoch,
                    )
                    regularization_mask = mask
                else:
                    inp, mask_n2v = _restrict_to_physics_denoising_window(
                        inp, tgt, mask_n2v, model
                    )
                    model_output = model(inp, return_parameters=regularization_enabled)
                    reconstruction = (
                        model_output.reconstruction
                        if regularization_enabled else model_output
                    )
                    data_loss = _physics_reconstruction_loss(
                        reconstruction, tgt, mask_n2v, model,
                        physics_data_loss_cfg, epoch,
                    )
                    regularization_mask = mask_n2v

                plain_mse = masked_mse_loss(
                    reconstruction, tgt, regularization_mask
                )
                if (
                    physics_data_loss_cfg is not None
                    and physics_data_loss_cfg.residual_variance_scaling
                ):
                    sigma_loss_metric = residual_variance_scaled_masked_mse_loss(
                        reconstruction,
                        tgt,
                        regularization_mask,
                        spectral_axis=model.spectral_axis,
                        frequency_mask=getattr(
                            model, "denoising_frequency_mask", None
                        ),
                        epsilon=physics_data_loss_cfg.residual_std_epsilon,
                    )
                    sigma_metric = residual_standard_deviation(
                        reconstruction,
                        tgt,
                        spectral_axis=model.spectral_axis,
                        frequency_mask=getattr(
                            model, "denoising_frequency_mask", None
                        ),
                        epsilon=physics_data_loss_cfg.residual_std_epsilon,
                    ).median()
                else:
                    sigma_loss_metric = plain_mse
                    sigma_metric = plain_mse.new_zeros(())

                shift_prior, fwhm_prior, kernel_prior, baseline_prior, voigt_nuisance_prior = physics_parameter_regularization(
                    model_output, regularization_mask, model.spectral_axis,
                    regularization_cfg, model.ford_baseline_design_matrix,
                ) if regularization_enabled else (
                    data_loss.new_zeros(()), data_loss.new_zeros(()),
                    data_loss.new_zeros(()), data_loss.new_zeros(()),
                    data_loss.new_zeros(()),
                )
                loss = (
                    data_loss
                    + (regularization_cfg.shift_weight * shift_prior if regularization_enabled else 0.0)
                    + (regularization_cfg.fwhm_weight * fwhm_prior if regularization_enabled else 0.0)
                    + (regularization_cfg.kernel_curvature_weight * kernel_prior if regularization_enabled else 0.0)
                    + (regularization_cfg.baseline_curvature_weight * baseline_prior if regularization_enabled else 0.0)
                    + (regularization_cfg.voigt_nuisance_weight * voigt_nuisance_prior if regularization_enabled else 0.0)
                )

                running += loss.item() * inp.size(0)
                running_data += data_loss.item() * inp.size(0)
                running_shift += shift_prior.item() * inp.size(0)
                running_fwhm += fwhm_prior.item() * inp.size(0)
                running_kernel += kernel_prior.item() * inp.size(0)
                running_baseline += baseline_prior.item() * inp.size(0)
                running_voigt_nuisance += voigt_nuisance_prior.item() * inp.size(0)
                running_mse += plain_mse.item() * inp.size(0)
                running_sigma_loss += sigma_loss_metric.item() * inp.size(0)
                running_sigma_median += sigma_metric.item() * inp.size(0)
                if regularization_enabled:
                    shift_squares, fwhm_squares, parameter_voxels = (
                        physics_parameter_squared_sums(
                            model_output, regularization_mask, model.spectral_axis
                        )
                    )
                    if parameter_voxels:
                        val_shift_squares = (
                            shift_squares if val_shift_squares is None
                            else val_shift_squares + shift_squares
                        )
                        val_fwhm_squares = (
                            fwhm_squares if val_fwhm_squares is None
                            else val_fwhm_squares + fwhm_squares
                        )
                        val_parameter_voxels += parameter_voxels

        avg_val = running / len(val_loader.dataset)
        avg_val_data = running_data / len(val_loader.dataset)
        avg_val_shift = running_shift / len(val_loader.dataset)
        avg_val_fwhm = running_fwhm / len(val_loader.dataset)
        avg_val_kernel = running_kernel / len(val_loader.dataset)
        avg_val_baseline = running_baseline / len(val_loader.dataset)
        avg_val_voigt_nuisance = running_voigt_nuisance / len(val_loader.dataset)
        avg_val_mse = running_mse / len(val_loader.dataset)
        avg_val_sigma_loss = running_sigma_loss / len(val_loader.dataset)
        avg_val_sigma_median = running_sigma_median / len(val_loader.dataset)
        avg_val_shift_weighted = shift_weight * avg_val_shift
        avg_val_fwhm_weighted = fwhm_weight * avg_val_fwhm
        avg_val_kernel_weighted = kernel_weight * avg_val_kernel
        avg_val_baseline_weighted = baseline_weight * avg_val_baseline
        avg_val_voigt_nuisance_weighted = voigt_nuisance_weight * avg_val_voigt_nuisance

        epsilon = 1e-12
        train_shift_percent = 100.0 * avg_train_shift_weighted / max(avg_train_data, epsilon)
        train_fwhm_percent = 100.0 * avg_train_fwhm_weighted / max(avg_train_data, epsilon)
        train_kernel_percent = 100.0 * avg_train_kernel_weighted / max(avg_train_data, epsilon)
        train_baseline_percent = 100.0 * avg_train_baseline_weighted / max(avg_train_data, epsilon)
        val_shift_percent = 100.0 * avg_val_shift_weighted / max(avg_val_data, epsilon)
        val_fwhm_percent = 100.0 * avg_val_fwhm_weighted / max(avg_val_data, epsilon)
        val_kernel_percent = 100.0 * avg_val_kernel_weighted / max(avg_val_data, epsilon)
        val_baseline_percent = 100.0 * avg_val_baseline_weighted / max(avg_val_data, epsilon)
        train_voigt_nuisance_percent = 100.0 * avg_train_voigt_nuisance_weighted / max(avg_train_data, epsilon)
        val_voigt_nuisance_percent = 100.0 * avg_val_voigt_nuisance_weighted / max(avg_val_data, epsilon)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        sigma_logging_enabled = bool(
            physics_data_loss_cfg is not None
            and physics_data_loss_cfg.residual_variance_scaling
        )
        log_parts = [
            f"Epoch {epoch:03d}",
            f"train_mse={avg_train_mse:.4e}",
        ]
        if sigma_logging_enabled:
            log_parts.extend(
                [
                    f"train_sigma_loss={avg_train_sigma_loss:.4e}",
                    f"train_sigma_median={avg_train_sigma_median:.4e}",
                ]
            )
        log_parts.append(f"train_total={avg_train:.4e}")
        if shift_weight > 0:
            log_parts.append(
                f"train_shift_reg={avg_train_shift_weighted:.4e} "
                f"({train_shift_percent:.2f}%)"
            )
        if fwhm_weight > 0:
            log_parts.append(
                f"train_fwhm_reg={avg_train_fwhm_weighted:.4e} "
                f"({train_fwhm_percent:.2f}%)"
            )
        if kernel_weight > 0:
            log_parts.append(
                f"train_kernel_reg={avg_train_kernel_weighted:.4e} "
                f"({train_kernel_percent:.2f}%)"
            )
        if baseline_weight > 0:
            log_parts.append(
                f"train_baseline_reg={avg_train_baseline_weighted:.4e} "
                f"({train_baseline_percent:.2f}%)"
            )
        if voigt_nuisance_weight > 0:
            log_parts.append(
                f"train_voigt_nuisance_reg={avg_train_voigt_nuisance_weighted:.4e} "
                f"({train_voigt_nuisance_percent:.2f}%)"
            )
        log_parts.append(f"val_mse={avg_val_mse:.4e}")
        if sigma_logging_enabled:
            log_parts.extend(
                [
                    f"val_sigma_loss={avg_val_sigma_loss:.4e}",
                    f"val_sigma_median={avg_val_sigma_median:.4e}",
                ]
            )
        log_parts.append(f"val_total={avg_val:.4e}")
        if shift_weight > 0:
            log_parts.append(
                f"val_shift_reg={avg_val_shift_weighted:.4e} "
                f"({val_shift_percent:.2f}%)"
            )
        if fwhm_weight > 0:
            log_parts.append(
                f"val_fwhm_reg={avg_val_fwhm_weighted:.4e} "
                f"({val_fwhm_percent:.2f}%)"
            )
        if kernel_weight > 0:
            log_parts.append(
                f"val_kernel_reg={avg_val_kernel_weighted:.4e} "
                f"({val_kernel_percent:.2f}%)"
            )
        if baseline_weight > 0:
            log_parts.append(
                f"val_baseline_reg={avg_val_baseline_weighted:.4e} "
                f"({val_baseline_percent:.2f}%)"
            )
        if voigt_nuisance_weight > 0:
            log_parts.append(
                f"val_voigt_nuisance_reg={avg_val_voigt_nuisance_weighted:.4e} "
                f"({val_voigt_nuisance_percent:.2f}%)"
            )
        log_parts.append(f"lr={current_lr:.2e}")
        epoch_message = " · ".join(log_parts)
        logger.info(epoch_message)

        if (
            regularization_enabled
            and val_parameter_voxels
            and (shift_weight > 0 or fwhm_weight > 0)
        ):
            train_shift_rms = (train_shift_squares / train_parameter_voxels).sqrt()
            train_fwhm_rms = (train_fwhm_squares / train_parameter_voxels).sqrt()
            val_shift_rms = (val_shift_squares / val_parameter_voxels).sqrt()
            val_fwhm_rms = (val_fwhm_squares / val_parameter_voxels).sqrt()
            parameter_parts = []
            for i, name in enumerate(model.basis_names):
                values = []
                if shift_weight > 0:
                    values.append(
                        f"shift train/val={train_shift_rms[i]:.3f}/"
                        f"{val_shift_rms[i]:.3f}"
                    )
                if fwhm_weight > 0:
                    values.append(
                        f"fwhm train/val={train_fwhm_rms[i]:.3f}/"
                        f"{val_fwhm_rms[i]:.3f}"
                    )
                parameter_parts.append(f"{name}[{', '.join(values)}]")
            logger.info(
                "Epoch %03d · per_metabolite_RMS_Hz · %s",
                epoch,
                " · ".join(parameter_parts),
            )
        print(epoch_message)
        rng_state = get_rng_state()

        # ---- Checkpoints ----
        # Keep model selection comparable to pure N2S runs: priors do not
        # decide which checkpoint is best.
        if avg_val_mse < best_val:
            best_val = avg_val_mse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": avg_val_mse,
                    "val_sigma_loss": avg_val_sigma_loss,
                    "val_total_loss": avg_val,
                    "best_val": best_val,
                    "rng_state": rng_state,
                    "self_supervised_mode": self_mode,
                },
                best_ckpt,
            )
            logger.info("   ↳ NEW BEST")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": avg_val_mse,
                "val_sigma_loss": avg_val_sigma_loss,
                "val_total_loss": avg_val,
                "best_val": best_val,
                "rng_state": rng_state,
                "self_supervised_mode": self_mode,
            },
            last_ckpt,
        )

    logger.info(f"Training fertig · best_val={best_val:.4e}")
    print(f"Training fertig · best_val={best_val:.4e}")


if __name__ == "__main__":
    raise SystemExit(
        "Please run training via scripts/train.py so configuration and run directories are handled consistently."
    )

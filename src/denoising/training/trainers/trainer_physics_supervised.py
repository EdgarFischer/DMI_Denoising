"""Plain-MSE supervised pretraining from classical-forD parameter maps."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from denoising.data.physics_supervised_dataset import PhysicsSupervisedDataset
from denoising.models.factory import build_model


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _parameter_channels(output) -> torch.Tensor:
    p = output.parameters
    amplitudes = p.amplitudes.movedim(-1, 1)
    nuisance = torch.stack(
        (
            p.frequency_shift_hz,
            p.lorentzian_fwhm_hz,
            p.gaussian_fwhm_hz,
            p.zero_order_phase_radians,
            p.first_order_phase_rad_per_hz,
        ),
        dim=1,
    )
    return torch.cat((amplitudes, nuisance), dim=1)


def _masked_parameter_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    weights = mask[:, None].to(dtype=prediction.dtype, device=prediction.device)
    difference = (prediction - target).square() * weights
    denominator = weights.sum() * prediction.shape[1]
    return difference.sum() / denominator.clamp_min(1.0)


def _standardized_supervised_loss(model, output, target, mask):
    if output.standardized_parameter_maps is None:
        raise RuntimeError(
            "Supervised physics training requires parameter_statistics_path."
        )
    target_model_units = (
        model.parameterization.convert_teacher_physical_channels(target)
    )
    target_z = model.parameterization.standardize_physical_channels(
        target_model_units
    )
    return _masked_parameter_mse(
        output.standardized_parameter_maps, target_z, mask
    )


def train(*, cfg, run_dir: str, checkpoint_dir: str, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    _seed(cfg.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.data.target_dirname is None or cfg.data.spatial_mask_filename is None:
        raise ValueError(
            "Supervised pretraining requires data.target_dirname and "
            "data.spatial_mask_filename."
        )
    names = cfg.model.physics.basis_components
    if names is None:
        raise ValueError(
            "Supervised pretraining requires explicit model.physics.basis_components."
        )
    spatial_patch = tuple(
        int(value) for value in cfg.patching.patch_sizes[:2]
    )
    dataset_kwargs = dict(
        base_dir=cfg.data.base_dir,
        data_filename=cfg.data.data_filename,
        mask_filename=cfg.data.spatial_mask_filename,
        target_dirname=cfg.data.target_dirname,
        metabolite_names=names,
        patch_size=spatial_patch,
    )
    train_dataset = PhysicsSupervisedDataset(
        subjects=list(cfg.data.train),
        num_samples=cfg.data.num_samples,
        **dataset_kwargs,
    )
    validation_dataset = PhysicsSupervisedDataset(
        subjects=list(cfg.data.val),
        num_samples=cfg.data.val_samples,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.optim.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=cfg.optim.num_workers,
        pin_memory=device.type == "cuda",
    )

    sample_input, sample_target, _ = train_dataset[0]
    model = build_model(cfg, tuple(sample_input.shape)).to(device)
    expected_parameters = len(names) + 5
    if sample_target.shape[0] != expected_parameters:
        raise RuntimeError(
            f"Expected {expected_parameters} target channels, "
            f"found {sample_target.shape[0]}."
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.factor
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    logger.info("Start PhysicsConv3D supervised parameter pretraining")
    logger.info("Metabolites (%d): %s", len(names), ", ".join(names))
    logger.info("Input shape: %s; target shape: %s", sample_input.shape, sample_target.shape)

    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        train_sum = 0.0
        for inputs, targets, mask in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            output = model(inputs, return_parameters=True)
            loss = _standardized_supervised_loss(
                model, output, targets, mask
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * inputs.shape[0]
        train_loss = train_sum / len(train_dataset)

        model.eval()
        validation_sum = 0.0
        with torch.inference_mode():
            for inputs, targets, mask in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                output = model(inputs, return_parameters=True)
                loss = _standardized_supervised_loss(
                    model, output, targets, mask
                )
                validation_sum += loss.item() * inputs.shape[0]
        validation_loss = validation_sum / len(validation_dataset)
        scheduler.step()

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": min(best_loss, validation_loss),
            "val_loss": validation_loss,
            "metabolite_names": tuple(names),
            "training_mode": "physics_supervised",
            "parameter_mean": model.parameterization.parameter_mean.detach().cpu(),
            "parameter_std": model.parameterization.parameter_std.detach().cpu(),
            "teacher_to_model_amplitude_scale": (
                model.parameterization.teacher_to_model_amplitude_scale.detach().cpu()
            ),
        }
        torch.save(state, os.path.join(checkpoint_dir, "last.pt"))
        if validation_loss < best_loss:
            best_loss = validation_loss
            state["best_val"] = best_loss
            torch.save(state, os.path.join(checkpoint_dir, "best.pt"))
        logger.info(
            "Epoch %03d · train=%.6e · val=%.6e · lr=%.3e",
            epoch,
            train_loss,
            validation_loss,
            optimizer.param_groups[0]["lr"],
        )
        print(
            f"[Ep {epoch:03d}] train={train_loss:.6e} "
            f"val={validation_loss:.6e}",
            flush=True,
        )
    print(f"Supervised pretraining finished · best_val={best_loss:.6e}", flush=True)

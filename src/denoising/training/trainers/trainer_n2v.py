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

from denoising.data.data_utils import load_and_preprocess_data
from denoising.data.mrsi_nd_dataset import MRSiNDataset
from denoising.losses.n2v_loss import masked_mse_loss
from denoising.models.unet2d import UNet2D
from denoising.models.unet3d import UNet3D

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
    fourier_axes,
    image_axes,
    channel_axis,
    masked_axes,
    fixed_indices,
    augmentation,
    normalization: bool,
    patching_enabled: bool,
    patch_sizes,
):
    data = load_and_preprocess_data(
        folder_names=folders,
        base_path=base_path,
        fourier_axes=list(fourier_axes),
        normalization=normalization,
    )

    return MRSiNDataset(
        data=data,
        image_axes=tuple(image_axes),
        channel_axis=channel_axis,
        masked_axes=tuple(masked_axes),
        fixed_indices=fixed_indices,
        transform=transform,
        num_samples=num_samples,
        augmentation=augmentation,
        patching_enabled=patching_enabled,
        patch_sizes=tuple(patch_sizes),
    )


def sample_n2s_mask(shape, p=0.03, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.rand(shape, device=device) < p).float()

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

    self_mode = getattr(cfg, "self_supervised_mode", None)
    if self_mode is None:
        self_mode = getattr(cfg.run, "self_supervised_mode", None)
    if self_mode is None:
        self_mode = "n2v"

    assert self_mode in ("n2v", "n2s"), f"Unsupported self_supervised_mode: {self_mode}"

    logger.info(f"Start Training ({self_mode.upper()}) – Seed {seed}")

    # ----- Dataset params -----
    base_path = "datasets"
    image_axes = cfg.data.image_axes
    channel_axis = cfg.data.channel_axis
    fourier_axes = cfg.data.fourier_axes
    masked_axes = cfg.mask.masked_axes

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
        fourier_axes=fourier_axes,
        image_axes=image_axes,
        channel_axis=channel_axis,
        masked_axes=masked_axes,
        fixed_indices=fixed_indices,
        augmentation=cfg.augmentation,
        normalization=do_norm,
        patching_enabled=patching_enabled,
        patch_sizes=patch_sizes,
    )

    val_ds = prepare_dataset(
        folders=list(cfg.data.val),
        transform=transform_val,
        num_samples=val_samples,
        base_path=base_path,
        fourier_axes=fourier_axes,
        image_axes=image_axes,
        channel_axis=channel_axis,
        masked_axes=masked_axes,
        fixed_indices=fixed_indices,
        augmentation=None,
        normalization=do_norm,
        patching_enabled=patching_enabled,
        patch_sizes=patch_sizes,
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
            logger.info(
                f"[DEBUG] N2V mask shape {tuple(mask_dbg.shape)}, mean {mask_dbg.float().mean():.4f}"
            )
        except StopIteration:
            logger.warning("[DEBUG] Train Loader leer – keine Maske.")
        except Exception as e:
            logger.warning(f"[DEBUG] Mask debug failed: {e}")

    # ----- Model / optim / scheduler -----
    if spatial_dim == 2:
        model = UNet2D(in_channels, out_channels, cfg.model.features).to(device)
        logger.info("[model] Using UNet2D")
    elif spatial_dim == 3:
        if UNet3D is None:
            raise ImportError(
                "spatial_dim==3 detected, but denoising.models.unet3d.UNet3D could not be imported."
            )
        model = UNet3D(in_channels, out_channels, cfg.model.features).to(device)
        logger.info("[model] Using UNet3D")
    else:
        raise ValueError(f"Unsupported spatial_dim={spatial_dim}. Expected 2 or 3.")

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

    if ckpt_path and os.path.isfile(ckpt_path):
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

        for inp, tgt, mask_n2v in train_loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            mask_n2v = mask_n2v.to(device, non_blocking=True)

            if self_mode == "n2s":
                mask = sample_n2s_mask(inp[:, :1].shape, p=0.03, device=device)
                inp_masked = inp * (1 - mask)
                loss = masked_mse_loss(model(inp_masked), tgt, mask)
            else:
                loss = masked_mse_loss(model(inp), tgt, mask_n2v)

            optim.zero_grad()
            loss.backward()
            tnn_utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += loss.item() * inp.size(0)

        avg_train = running / len(train_loader.dataset)

        # ---- VALID ----
        model.eval()
        running = 0.0
        with torch.no_grad():
            for inp, tgt, mask_n2v in val_loader:
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                mask_n2v = mask_n2v.to(device, non_blocking=True)

                if self_mode == "n2s":
                    mask = sample_n2s_mask(inp[:, :1].shape, p=0.03, device=device)
                    inp_masked = inp * (1 - mask)
                    loss = masked_mse_loss(model(inp_masked), tgt, mask)
                else:
                    loss = masked_mse_loss(model(inp), tgt, mask_n2v)

                running += loss.item() * inp.size(0)

        avg_val = running / len(val_loader.dataset)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e} · lr={current_lr:.2e}"
        )
        print(f"[Ep {epoch:03d}] train={avg_train:.4e} val={avg_val:.4e}")
        rng_state = get_rng_state()

        # ---- Checkpoints ----
        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": avg_val,
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
                "val_loss": avg_val,
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
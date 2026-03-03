# ───────────────────────── denoising/training/trainers/trainer_n2v.py ─────────────────────────
# Noise2Void / Noise2Self, 2-D (refactored: no global config.py, no import side effects)

import os
import random
import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.utils as tnn_utils
from torch.utils.data import DataLoader

from denoising.data.data_utils import load_and_preprocess_data
from denoising.data.mrsi_2d_dataset import MRSiNDataset
from denoising.losses.n2v_loss import masked_mse_loss
from denoising.models.unet2d import UNet2D


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
    fixed_indices,
    phase_prob: float,
    normalization: bool,
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
        fixed_indices=fixed_indices,
        transform=transform,
        num_samples=num_samples,
        phase_prob=phase_prob,
    )


def sample_n2s_mask(shape, p=0.03, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.rand(shape, device=device) < p).float()


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
    """
    Args:
        cfg: typed config object (from YAML). Expected fields:
            cfg.run.seed, cfg.data.train/val, cfg.data.image_axes, cfg.data.fourier_axes,
            cfg.data.num_samples/val_samples, cfg.model.*, cfg.optim.*
            Optional: cfg.data.fixed_indices, cfg.data.phase_prob, cfg.run.self_supervised_mode
        run_dir/checkpoint_dir/log_dir: output paths
        transform_train/transform_val: already-built masking transforms (recommended)
    """
    logger = setup_logger(log_dir)

    # ----- Repro / mode -----
    seed = cfg.run.seed
    set_seed(seed)

    # If you keep a mode in YAML, use it; otherwise default to n2v
    self_mode = getattr(cfg, "self_supervised_mode", None)
    if self_mode is None:
        # common place if you keep it under run or masking
        self_mode = getattr(cfg.run, "self_supervised_mode", None)
    if self_mode is None:
        self_mode = "n2v"

    assert self_mode in ("n2v", "n2s"), f"Unsupported self_supervised_mode: {self_mode}"

    logger.info(f"Start Training 2D ({self_mode.upper()}) – Seed {seed}")

    # ----- Dataset params (handle optional fields cleanly) -----
    base_path = "datasets"
    image_axes = cfg.data.image_axes
    fourier_axes = cfg.data.fourier_axes
    num_samples = cfg.data.num_samples
    val_samples = cfg.data.val_samples

    fixed_indices = getattr(cfg.data, "fixed_indices", None)
    phase_prob = getattr(cfg.data, "phase_prob", 1.0)

    # ----- Build datasets -----
    do_norm = getattr(cfg.data, "normalization", True)

    train_ds = prepare_dataset(
        folders=list(cfg.data.train),
        transform=transform_train,
        num_samples=num_samples,
        base_path=base_path,
        fourier_axes=fourier_axes,
        image_axes=image_axes,
        fixed_indices=fixed_indices,
        phase_prob=phase_prob,
        normalization=do_norm, 
    )
    val_ds = prepare_dataset(
        folders=list(cfg.data.val),
        transform=transform_val,
        num_samples=val_samples,
        base_path=base_path,
        fourier_axes=fourier_axes,
        image_axes=image_axes,
        fixed_indices=fixed_indices,
        phase_prob=phase_prob,
        normalization=do_norm,
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

    # ----- Debug mask -----
    if self_mode == "n2v":
        try:
            _, _, mask_dbg = next(iter(train_loader))
            logger.info(
                f"[DEBUG] N2V-Maske shape {tuple(mask_dbg.shape)}, mean {mask_dbg.float().mean():.4f}"
            )
        except StopIteration:
            logger.warning("[DEBUG] Train Loader leer – keine Maske.")
        except Exception as e:
            logger.warning(f"[DEBUG] Mask debug failed: {e}")

    # ----- Model / optim / scheduler -----
    model = UNet2D(2, 2, cfg.model.features).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    def lr_lambda(epoch: int):
        lr = cfg.optim.lr * (cfg.optim.factor ** (epoch // cfg.optim.step_size))
        lr = max(lr, cfg.optim.min_lr)
        return lr / cfg.optim.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ----- Optional pretrained -----
    ckpt_path = getattr(cfg, "pretrained_ckpt", "")
    if ckpt_path == "":
        ckpt_path = getattr(cfg.optim, "pretrained_ckpt", "") if hasattr(cfg, "optim") else ""
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        strict_flag = getattr(cfg, "pretrained_strict", True)
        model.load_state_dict(state_dict, strict=strict_flag)
        logger.info(f"Gewichte geladen aus {ckpt_path}")
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
    best_val = float("inf")

    epochs = cfg.optim.epochs

    # ----- Training loop -----
    for epoch in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        running = 0.0

        for inp, tgt, mask_n2v in train_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask_n2v = mask_n2v.to(device)

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
                inp = inp.to(device)
                tgt = tgt.to(device)
                mask_n2v = mask_n2v.to(device)

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

        # ---- Checkpoints ----
        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "val_loss": best_val,
                },
                best_ckpt,
            )
            logger.info("   ↳ NEW BEST")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "val_loss": avg_val,
            },
            last_ckpt,
        )

    logger.info(f"Training fertig · best_val={best_val:.4e}")
    print(f"Training fertig · best_val={best_val:.4e}")


if __name__ == "__main__":
    raise SystemExit(
        "Please run training via scripts/train.py so configuration and run directories are handled consistently."
    )
# train/trainer_unet_lowrank.py
# -*- coding: utf-8 -*-
"""
Supervised Pretraining: U-Net lernt Noisy -> Low-Rank.

Dieses Script ist eine abgespeckte, ASCII-only, syntaktisch saubere Version
ohne Sonderzeichen (keine Mittelpunkte, En-Dashes etc.), um Copy/Paste-
Probleme zu vermeiden.

Usage:
    python train/trainer_unet_lowrank.py

Voraussetzungen:
    - config.py definiert: GPU_NUMBER, log_dir, checkpoint_dir, seed,
      train_data, val_data, transform_train, transform_val, num_samples,
      val_samples, image_axes, fixed_indices, in_channels, out_channels,
      features, lr, epochs, num_workers, pin_memory, lowrank_rank.
    - data.load_noisy_and_lowrank_data() existiert und liefert
      (noisy_array, lowrank_array) gleicher Shape (complex od. real).
    - data.supervised_to_lowrank_dataset.SupervisedToLowRankDataset vorhanden.
    - losses.supervised_to_lowrank_loss.supervised_mse_loss vorhanden.
    - models.unet2d.UNet2D vorhanden.
"""

import os
import sys
import random
import logging
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Projekt-Root einhängen ------------------------------------------------------
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_NUMBER)

import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Projekt-Module --------------------------------------------------------------
# -----------------------------------------------------------------------------
from data.data_utils import load_noisy_and_lowrank_data
from data.supervised_to_lowrank_dataset import SupervisedToLowRankDataset
from losses.supervised_to_lowrank_loss import supervised_mse_loss
from models.unet2d import UNet2D

# -----------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# -----------------------------------------------------------------------------
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train_unet_lowrank.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Hilfsfunktionen ------------------------------------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset(
    folders: List[str],
    image_axes: Tuple[int, int],
    transform,
    num_samples: int,
):
    noisy, lowrank = load_noisy_and_lowrank_data(
        folder_names=folders,
        base_path="datasets",
        fourier_axes=config.fourier_transform_axes,
        normalize=True,
        rank=config.lowrank_rank,
    )
    return SupervisedToLowRankDataset(
        noisy_data=noisy,
        lowrank_data=lowrank,
        image_axes=image_axes,
        fixed_indices=config.fixed_indices,
        transform=transform,
        num_samples=num_samples,
    )


# -----------------------------------------------------------------------------
# Haupt-Trainer --------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_unet_lowrank():
    set_seed(config.seed)
    logger.info("Start Supervised-to-LowRank Training (U-Net)")

    # ----- Datasets ---------------------------------------------------------
    train_ds = prepare_dataset(
        config.train_data, config.image_axes, None, config.num_samples
    )
    val_ds = prepare_dataset(
        config.val_data, config.image_axes, None, config.val_samples
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # ----- Modell / Optimizer ----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(
        in_channels=config.in_channels,  # typ. 2 (real+imag)
        out_channels=config.out_channels,
        features=config.features,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ----- Checkpoints ------------------------------------------------------
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best_unet_lowrank.pt")
    last_ckpt = os.path.join(config.checkpoint_dir, "last_unet_lowrank.pt")
    best_val = float("inf")

    # ----- Training Loop ----------------------------------------------------
    for epoch in range(1, config.epochs + 1):
        # ---- TRAIN ---------------------------------------------------------
        model.train()
        running = 0.0
        for inp_noisy, tgt_lr in train_loader:
            inp_noisy = inp_noisy.to(device)
            tgt_lr = tgt_lr.to(device)

            pred = model(inp_noisy)
            loss = supervised_mse_loss(pred, tgt_lr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * inp_noisy.size(0)

        avg_train = running / len(train_loader.dataset)

        # ---- VALID ---------------------------------------------------------
        model.eval()
        running = 0.0
        with torch.no_grad():
            for inp_noisy, tgt_lr in val_loader:
                inp_noisy = inp_noisy.to(device)
                tgt_lr = tgt_lr.to(device)
                pred = model(inp_noisy)
                running += supervised_mse_loss(pred, tgt_lr).item() * inp_noisy.size(0)
        avg_val = running / len(val_loader.dataset)

        # ---- Logging -------------------------------------------------------
        msg = f"Epoch {epoch:03d} | train={avg_train:.4e} | val={avg_val:.4e}"
        logger.info(msg)
        print(msg)

        # ---- Checkpoints ---------------------------------------------------
        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss": best_val,
                },
                best_ckpt,
            )
            logger.info(f"NEW BEST (val={best_val:.4e}) saved -> {best_ckpt}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss": avg_val,
            },
            last_ckpt,
        )

    logger.info(f"Training finished | best_val={best_val:.4e}")
    print(f"Training finished | best_val={best_val:.4e}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_unet_lowrank()




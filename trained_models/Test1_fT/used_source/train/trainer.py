
# train/trainer.py

import os
import random
import logging
from typing import List

import sys

# 1) bestimme den Pfad zum Projekt-Root (eine Ebene höher)
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# 2) füge ihn vor allen anderen Einträgen ein
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_NUMBER  

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─ Projekt-Module ────────────────────────────────────────────────────────────
from data.data_utils import load_and_preprocess_data
from data.mrsi_2d_dataset import MRSiNDataset
from data.transforms import StratifiedPixelSelection
from losses.n2v_loss import masked_mse_loss
from models.unet2d import UNet2D

# ─ Logging ──────────────────────────────────────────────────────────────────
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─ Hilfsfunktionen ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare_dataset(folders: List[str], transform, num_samples: int):
    data = load_and_preprocess_data(
        folder_names=folders,
        base_path="datasets",
        fourier_axes=config.fourier_transform_axes,
        normalize=True,
    )
    return MRSiNDataset(
        data=data,
        image_axes=config.image_axes,
        fixed_indices=config.fixed_indices,
        transform=transform,
        num_samples=num_samples,
    )

# ─ Hauptfunktion ────────────────────────────────────────────────────────────
def train():
    set_seed(config.seed)
    logger.info(f"Start Training – Seed {config.seed}")

    # Datensätze & Loader
    train_ds = prepare_dataset(config.train_data, config.transform_train, config.num_samples)
    val_ds   = prepare_dataset(config.val_data,   config.transform_val,   config.val_samples)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=config.pin_memory)

    # Device & Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Using device: {device}")
    model = UNet2D(config.in_channels, config.out_channels, config.features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Checkpoint-Pfade
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best.pt")
    last_ckpt_path = os.path.join(config.checkpoint_dir, "last.pt")
    best_val_loss  = float("inf")

    # Training-Loop
    for epoch in range(1, config.epochs + 1):
        # --- Training ---
        model.train(); train_loss = 0.0
        for inp, tgt, mask in train_loader:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = masked_mse_loss(model(inp), tgt, mask)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * inp.size(0)
        avg_train = train_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, mask in val_loader:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                val_loss += masked_mse_loss(model(inp), tgt, mask).item() * inp.size(0)
        avg_val = val_loss / len(val_loader.dataset)

        # --- Logging ---
        logger.info(f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e}")

        # --- Best-Checkpoint ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {"epoch": epoch,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "val_loss": best_val_loss},
                best_ckpt_path)
            logger.info(f"   ↳ NEW BEST (val={best_val_loss:.4e}) → {best_ckpt_path}")

        # --- Last-Checkpoint (überschreibt jedes Mal) ---
        torch.save(
            {"epoch": epoch,
             "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "val_loss": avg_val},
            last_ckpt_path)

    logger.info(f"Training fertig · best_val={best_val_loss:.4e}")
    logger.info(f"   ↳ Finales Modell (last.pt) gespeichert")

# Wenn Datei direkt ausgeführt
if __name__ == "__main__":
    train()

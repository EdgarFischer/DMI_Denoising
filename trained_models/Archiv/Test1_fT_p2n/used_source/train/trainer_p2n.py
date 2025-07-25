
# train/trainer_p2n.py
# -----------------------------------------------------------------------------
# Finetuning- / Training-Skript für Positive2Negative (P2N)
# -----------------------------------------------------------------------------
# Dieses Skript basiert auf deinem bisherigen Noise2Void‑Trainer. Es nutzt dasselbe
# Dataset & Modell, tauscht aber:
#   • den Loss (p2n_loss statt masked_mse_loss)
#   • den Trainings‑Step (RDC + DCS anstelle der Maskierung)
# Masken und Targets werden zwar nach wie vor vom Dataset geliefert, aber hier
# ignoriert.
# -----------------------------------------------------------------------------

import os
import random
import logging
from typing import List
import sys

# 1) Projekt‑Root finden und einhängen
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # <- enthält deine Hyperparameter

# Begrenze sichtbare GPU (wie beim alten Skript)
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_NUMBER

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─ Projekt‑Module ────────────────────────────────────────────────────────────
from data.data_utils import load_and_preprocess_data
from data.mrsi_2d_dataset import MRSiNDataset
from losses.p2n_loss import p2n_loss
from models.unet2d import UNet2D

# ─ Logging ───────────────────────────────────────────────────────────────────
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train_p2n.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─ Hilfsfunktionen ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

# -----------------------------------------------------------------------------
# Haupt‑Trainingsfunktion
# -----------------------------------------------------------------------------

def train_p2n():
    """Fein‑ oder Volltraining mit Positive2Negative-Konsistenz‑Loss."""

    set_seed(config.seed)
    logger.info(f"Start P2N‑Training – Seed {config.seed}")

    # Datensätze & Loader
    train_ds = prepare_dataset(
        config.train_data, config.transform_train, config.num_samples
    )
    val_ds = prepare_dataset(
        config.val_data, config.transform_val, config.val_samples
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

    # ─ Debug‑Check Masken (nur einmal) ──────────────────────────────────────
    first = next(iter(train_loader))
    if len(first) == 3:
        _, _, mask_dbg = first
        logger.info(
            f"[DEBUG] Dataset liefert Masken‑Shape {mask_dbg.shape} – wird ignoriert."
        )

    # ─ Device & Modell ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer‑P2N] Using device: {device}")

    model = UNet2D(config.in_channels, config.out_channels, config.features).to(device)

    # Lade optional N2V‑Weights zum Finetuning
    if hasattr(config, "pretrained_ckpt") and os.path.isfile(config.pretrained_ckpt):
        ckpt = torch.load(config.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Geladene Vortrainings‑Gewichte: {config.pretrained_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Checkpoints
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best.pt")
    last_ckpt_path = os.path.join(config.checkpoint_dir, "last.pt")
    best_val_loss = float("inf")

    sigma_std = getattr(config, "sigma_std", 0.1)  # Streuung um 1.0 herum

    # ─ Epoch‑Loop ───────────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Dataset liefert (inp, tgt, mask) – mask wird ignoriert
            inp = batch[0].to(device)  # Form (B,C,H,W)

            # --- RDC ---
            x_hat = model(inp)           # erster Forward
            n_hat = inp - x_hat          # geschätztes Rauschen

            sigma_p = torch.randn_like(n_hat) * sigma_std + 1.0
            sigma_n = torch.randn_like(n_hat) * sigma_std + 1.0

            y_p = x_hat +  sigma_p * n_hat
            y_n = x_hat -  sigma_n * n_hat

            # --- DCS ---
            x_p = model(y_p)
            x_n = model(y_n)

            loss = p2n_loss(x_p, x_n)    # reiner L2‑Loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inp.size(0)

        avg_train = train_loss / len(train_loader.dataset)

        # ------------------- VALIDATION -------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inp_val = batch[0].to(device)

                # gleiche RDC+DCS‑Abfolge
                x_hat_v = model(inp_val)
                n_hat_v = inp_val - x_hat_v
                sigma_p_v = torch.randn_like(n_hat_v) * sigma_std + 1.0
                sigma_n_v = torch.randn_like(n_hat_v) * sigma_std + 1.0
                y_p_v = x_hat_v + sigma_p_v * n_hat_v
                y_n_v = x_hat_v - sigma_n_v * n_hat_v
                x_p_v = model(y_p_v)
                x_n_v = model(y_n_v)
                val_loss += p2n_loss(x_p_v, x_n_v).item() * inp_val.size(0)

        avg_val = val_loss / len(val_loader.dataset)

        # ------------------- Logging & Checkpoints -------------------
        logger.info(f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                best_ckpt_path,
            )
            logger.info(f"   ↳ NEW BEST (val={best_val_loss:.4e}) → {best_ckpt_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val,
            },
            last_ckpt_path,
        )

    logger.info(f"P2N‑Training fertig · best_val={best_val_loss:.4e}")
    logger.info("   ↳ Finales Modell gespeichert (last_p2n.pt)")


# ----------------------------------------------------------------------------
# Wenn diese Datei direkt ausgeführt wird
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    train_p2n()

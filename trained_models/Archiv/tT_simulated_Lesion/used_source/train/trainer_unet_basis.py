# ───────────────────────── train/trainer_unet_basis.py ─────────────────────────
import os, sys, random, logging
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Projekt-Root & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_NUMBER

# ---------------------------------------------------------------------------
# Self‐supervision umschalten wie gehabt
# ---------------------------------------------------------------------------
SELF_SUPERVISED_MODE = getattr(config, "SELF_SUPERVISED_MODE", "n2v")
assert SELF_SUPERVISED_MODE in ("n2v", "n2s"), "n2v oder n2s wählen!"

# ---------------------------------------------------------------------------
# Projekt-Module
# ---------------------------------------------------------------------------
from data.data_utils       import load_and_preprocess_data
from data.mrsi_2d_dataset  import MRSiNDataset
from data.transforms       import StratifiedPixelSelection
from losses.n2v_loss       import masked_mse_loss
from models.basis_unet2d_coeff import BasisCoeffUNet2D
from train.trainer_n2v    import set_seed, prepare_dataset, sample_n2s_mask

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train_basis.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Haupt‐Trainer
# ---------------------------------------------------------------------------
def train_unet_basis():
    set_seed(config.seed)
    logger.info(f"Start Basis‐Training – Seed {config.seed}")

    # ─ Daten ─────────────────────────────────────────────────────────────────
    train_ds = prepare_dataset(
        config.train_data, config.transform_train, config.num_samples
    )
    val_ds = prepare_dataset(
        config.val_data,   config.transform_val,   config.val_samples
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )

    # ─ Basis laden ────────────────────────────────────────────────────────────
    # Erzeuge V_r einmal extern:
    #   >>> V_r, _ = build_basis(noisy_subject0, rank=config.lowrank_rank)
    #   >>> np.save(config.basis_file, V_r)
    #
    logger.info(f"Lade Basis-Rank{config.lowrank_rank} aus {config.basis_file}")
    V_r = np.load(config.basis_file)  # Form: (f, r), komplex
    
    # ─ Modell / Optimizer ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BasisCoeffUNet2D(basis=V_r, features=config.features).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ─ (Optional) Pretrained wie gehabt ──────────────────────────────────────
    ckpt_path = getattr(config, "pretrained_ckpt", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd   = ckpt.get("model_state", ckpt)
        model.load_state_dict(sd, strict=getattr(config, "pretrained_strict", True))
        logger.info(f"Geladene Gewichte aus {ckpt_path}")

    # ─ Checkpoints setup ─────────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best_basis.pt")
    last_ckpt = os.path.join(config.checkpoint_dir, "last_basis.pt")
    best_val  = float("inf")

    # ─ Training-Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        # ----- TRAIN ----- 
        model.train()
        train_loss = 0.0
        for inp, tgt, mask in train_loader:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)

            if SELF_SUPERVISED_MODE == "n2s":
                m = sample_n2s_mask(inp[:, :1].shape)
                inp_masked = inp * (1 - m)
                loss = ((model(inp_masked) - tgt)**2 * m).sum() / m.sum()
            else:
                loss = masked_mse_loss(model(inp), tgt, mask)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * inp.size(0)
        avg_train = train_loss / len(train_loader.dataset)

        # ----- VALID -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, mask in val_loader:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                if SELF_SUPERVISED_MODE == "n2s":
                    m = sample_n2s_mask(inp[:, :1].shape)
                    inp_masked = inp * (1 - m)
                    loss = ((model(inp_masked) - tgt)**2 * m).sum() / m.sum()
                else:
                    loss = masked_mse_loss(model(inp), tgt, mask)
                val_loss += loss.item() * inp.size(0)
        avg_val = val_loss / len(val_loader.dataset)

        logger.info(f"[Epoch {epoch:03d}] train={avg_train:.4e} val={avg_val:.4e}")
        print(f"[Basis Ep {epoch:03d}] train={avg_train:.4e} val={avg_val:.4e}")

        # ----- Checkpoints -----
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(),
                        "val_loss": best_val},
                       best_ckpt)
            logger.info(f"NEW BEST → {best_ckpt}")
        torch.save({"epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "val_loss": avg_val},
                   last_ckpt)

    logger.info(f"Training abgeschlossen · best_val={best_val:.4e}")
    print(f"Training abgeschlossen · best_val={best_val:.4e}")

if __name__ == "__main__":
    train_unet_basis()



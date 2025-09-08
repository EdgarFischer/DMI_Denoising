# ───────────────────────── train/trainer.py (NEU, mit Pretrain) ──────────────────────────
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
# Self-supervision umschalten:  "n2v"  oder  "n2s"
# ---------------------------------------------------------------------------
SELF_SUPERVISED_MODE = getattr(config, "SELF_SUPERVISED_MODE", "n2v")
assert SELF_SUPERVISED_MODE in ("n2v", "n2s"), "n2v oder n2s wählen!"

# ---------------------------------------------------------------------------
# Projekt-Module
# ---------------------------------------------------------------------------
from data.data_utils       import load_and_preprocess_data
from data.mrsi_2d_dataset  import MRSiNDataset
from data.transforms       import StratifiedPixelSelection          # N2V-Maske
from losses.n2v_loss       import masked_mse_loss
from models.unet2d         import UNet2D

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare_dataset(folders: List[str], transform, num_samples: int):
    data = load_and_preprocess_data(
        folder_names = folders,
        base_path    = "datasets",
        fourier_axes = config.fourier_transform_axes,
        normalize    = True,
    )
    return MRSiNDataset(
        data          = data,
        image_axes    = config.image_axes,
        fixed_indices = config.fixed_indices,
        transform     = transform,
        num_samples   = num_samples,
        phase_prob    = config.phase_prob,   #  ← NEU
    )

# ---------------------------------------------------------------------------
# Noise2Self: zufällige Maske on-the-fly
# ---------------------------------------------------------------------------
def sample_n2s_mask(shape, p=0.03):
    """Bernoulli-Maske (B,1,H,W) – 1 = für Loss benutzen & Input auf 0 setzen."""
    return (torch.rand(shape, device="cuda" if torch.cuda.is_available() else "cpu") < p).float()

# ---------------------------------------------------------------------------
# Haupt-Trainer
# ---------------------------------------------------------------------------
def train():
    set_seed(config.seed)
    logger.info(f"Start Training  ({SELF_SUPERVISED_MODE.upper()})  – Seed {config.seed}")

    # ─ Daten ───────────────────────────────────────────────────────────────
    train_ds = prepare_dataset(config.train_data, config.transform_train, config.num_samples)
    val_ds   = prepare_dataset(config.val_data  , config.transform_val  , config.val_samples)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader   = DataLoader(
        val_ds  , batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory)

    # ─ Debug-Ausgabe Maske ────────────────────────────────────────────────
    if SELF_SUPERVISED_MODE == "n2v":
        try:
            _, _, mask_dbg = next(iter(train_loader))
            logger.info(f"[DEBUG] N2V-Maske shape {mask_dbg.shape}, mean {mask_dbg.float().mean():.4f}")
        except StopIteration:
            logger.warning("[DEBUG] Train Loader leer – keine Maske zum Loggen.")

    # ─ Modell / Optimizer ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet2D(config.in_channels, config.out_channels, config.features).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ─ Optional: Vortrainiertes Modell laden ───────────────────────────────
    # Steuerung in config.py:
    #   pretrained_ckpt = "path/to/checkpoint.pt"  ("" oder None = nichts laden)
    #   pretrained_strict = True/False             (Default: True)
    #   load_optimizer_from_pretrained = False     (nur wenn wirklich weitertrainieren)
    ckpt_path = getattr(config, "pretrained_ckpt", "")
    if ckpt_path:
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("model_state", ckpt)
            strict_flag = getattr(config, "pretrained_strict", True)
            missing, unexpected = model.load_state_dict(state_dict, strict=strict_flag)
            logger.info(f"Geladene Gewichte aus {ckpt_path}")
            if not strict_flag:
                if missing:   logger.info(f"  -> missing keys: {missing}")
                if unexpected:logger.info(f"  -> unexpected keys: {unexpected}")
            if getattr(config, "load_optimizer_from_pretrained", False) and "optimizer_state" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optimizer_state"])
                    logger.info("Optimizer-State mitgeladen.")
                except Exception as e:
                    logger.warning(f"Optimizer-State konnte nicht geladen werden: {e}")
        else:
            logger.warning(f"pretrained_ckpt nicht gefunden: {ckpt_path}")

    # ─ Checkpoints ────────────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best.pt")
    last_ckpt = os.path.join(config.checkpoint_dir, "last.pt")
    best_val  = float("inf")

    # ─ Training-Loop ──────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):

        # ---- TRAIN -------------------------------------------------------
        model.train(); running = 0.0
        for inp, tgt, mask_n2v in train_loader:
            inp, tgt, mask_n2v = inp.to(device), tgt.to(device), mask_n2v.to(device)

            # ----- Noise2Self: on-the-fly Zero-Mask ----------------------
            if SELF_SUPERVISED_MODE == "n2s":
                mask = sample_n2s_mask(inp[:, :1].shape, p=0.03)  # (B,1,H,W)
                inp_masked = inp * (1 - mask)                     # Pixel → 0
                loss = ((model(inp_masked) - tgt) ** 2 * mask).sum() / mask.sum()

            # ----- Noise2Void: klassische Mask --------------------------
            else:   # "n2v"
                loss = masked_mse_loss(model(inp), tgt, mask_n2v)

            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item() * inp.size(0)

        avg_train = running / len(train_loader.dataset)

        # ---- VALID ------------------------------------------------------
        model.eval(); running = 0.0
        with torch.no_grad():
            for inp, tgt, mask_n2v in val_loader:
                inp, tgt, mask_n2v = inp.to(device), tgt.to(device), mask_n2v.to(device)

                if SELF_SUPERVISED_MODE == "n2s":
                    mask = sample_n2s_mask(inp[:, :1].shape, p=0.03)
                    inp_masked = inp * (1 - mask)
                    loss = ((model(inp_masked) - tgt) ** 2 * mask).sum() / mask.sum()
                else:
                    loss = masked_mse_loss(model(inp), tgt, mask_n2v)

                running += loss.item() * inp.size(0)

        avg_val = running / len(val_loader.dataset)
        logger.info(f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e}")
        print(f"[Ep {epoch:03d}] train={avg_train:.4e} val={avg_val:.4e}")

        # ---- Checkpoints -------------------------------------------------
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(), "val_loss": best_val}, best_ckpt)
            logger.info(f"   ↳ NEW BEST → {best_ckpt}")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(), "val_loss": avg_val}, last_ckpt)

    logger.info(f"Training fertig · best_val={best_val:.4e}")
    print(f"Training fertig · best_val={best_val:.4e}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()


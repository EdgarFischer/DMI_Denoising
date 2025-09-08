# train/trainer3d.py
import os, sys, random, logging
from typing import List

# ─ Projekt-Root in sys.path ────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_NUMBER

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─ Projekt-Module ──────────────────────────────────────────────────────────
from data.data_utils      import load_and_preprocess_data
from data.mrsi_3d_dataset import MRSiN3DDataset
from data.transforms_3d   import StratifiedVoxelSelection
from models.unet3d        import UNet3D
from losses.n2v_loss      import masked_mse_loss

# ─ Logging ────────────────────────────────────────────────────────────────
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train3d.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─ Hilfsfunktionen ────────────────────────────────────────────────────────
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
    return MRSiN3DDataset(
        data          = data,
        volume_axes   = config.volume_axes,
        fixed_indices = config.fixed_indices,
        transform     = transform,
        num_samples   = num_samples,
    )

# ─ Hauptfunktion ──────────────────────────────────────────────────────────
def train_n2v_3d():
    set_seed(config.seed)
    logger.info(f"Start 3-D-Training – Seed {config.seed}")

    # Datensätze & Loader
    train_ds = prepare_dataset(config.train_data,
                               config.transform_train,
                               config.num_samples)
    val_ds   = prepare_dataset(config.val_data,
                               config.transform_val,
                               config.val_samples)

    train_loader = DataLoader(train_ds,
                              batch_size  = config.batch_size,
                              shuffle     = True,
                              num_workers = config.num_workers,
                              pin_memory  = config.pin_memory)
    val_loader   = DataLoader(val_ds,
                              batch_size  = config.batch_size,
                              shuffle     = False,
                              num_workers = config.num_workers,
                              pin_memory  = config.pin_memory)

    # ─ Debug: Maske checken ────────────────────────────────────────────────
    inp_dbg, _, mask_dbg = next(iter(train_loader))
    print(f"[DEBUG] mask.shape       = {mask_dbg.shape}")   # (B,1,D,H,W)
    print(f"[DEBUG] mask.sum pro Vol = {mask_dbg[0].sum().item()}")

    # ─ Device & Modell ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Using device: {device}")

    model     = UNet3D(config.in_channels,
                       config.out_channels,
                       config.features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Checkpoints
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best3d.pt")
    last_ckpt = os.path.join(config.checkpoint_dir, "last3d.pt")
    best_val  = float("inf")

    # ─ Training-Loop ───────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        # --- Training -----------------------------------------------
        model.train(); running = 0.0
        for inp, tgt, mask in train_loader:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            optimizer.zero_grad()

            pred  = model(inp)                       # ← kein mask-Arg
            loss  = masked_mse_loss(pred, tgt, mask)

            loss.backward(); optimizer.step()
            running += loss.item() * inp.size(0)
        avg_train = running / len(train_loader.dataset)

        # --- Validation ---------------------------------------------
        model.eval(); running = 0.0
        with torch.no_grad():
            for inp, tgt, mask in val_loader:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                pred  = model(inp)
                running += masked_mse_loss(pred, tgt, mask).item() * inp.size(0)
        avg_val = running / len(val_loader.dataset)

        # --- Logging & Checkpoints ----------------------------------
        logger.info(f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {"epoch": epoch,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "val_loss": best_val},
                best_ckpt)
            logger.info(f"   ↳ NEW BEST → {best_ckpt}")

        torch.save(
            {"epoch": epoch,
             "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "val_loss": avg_val},
            last_ckpt)

    logger.info(f"Training fertig · best_val={best_val:.4e}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_n2v_3d()


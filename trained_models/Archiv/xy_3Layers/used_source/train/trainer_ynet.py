# train/trainer_ynet.py
# -----------------------------------------------------------------------------
# Training‑Script für zweipfadiges Y‑Net (Noisy + Low‑Rank) im Noise2Void‑Setup.
# Dieses File ist eine *vollständig eigenständige* Kopie des alten Trainers –
# lediglich Daten‑Loading, Dataset, Modell und Loss‑Aufruf wurden auf Y‑Net
# umgestellt. Alles andere (Logging, Checkpoints, Config‑Variablen) bleibt
# unverändert, sodass eure bestehende Infrastruktur weiter funktioniert.
# -----------------------------------------------------------------------------

import os, random, logging, sys
from typing import List

# ─ Projekt‑Root einhängen (eine Ebene höher) ---------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_NUMBER  # z. B. "0"

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─ Projekt‑Module -------------------------------------------------------------
from data.data_utils        import load_noisy_and_lowrank_data
from data.mrsi_y_dataset    import MRSiYDataset
from data.transforms        import StratifiedPixelSelection
from losses.n2v_loss        import masked_mse_loss
from models.ynet2d          import YNet2D

# ─ Logging --------------------------------------------------------------------
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train_ynet.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─ Hilfsfunktionen ------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_y_dataset(folders: List[str], transform, num_samples: int):
    noisy, lowrank = load_noisy_and_lowrank_data(
        folder_names = folders,
        base_path    = "datasets",
        fourier_axes = config.fourier_transform_axes,
        normalize    = True,
        rank         = config.lowrank_rank,
    )
    return MRSiYDataset(
        noisy_data    = noisy,
        lowrank_data  = lowrank,
        image_axes    = config.image_axes,
        fixed_indices = config.fixed_indices,
        transform     = transform,
        num_samples   = num_samples,
    )

# ─ Haupt‑Funktion -------------------------------------------------------------

def train_ynet_n2v():
    set_seed(config.seed)
    logger.info(f"Start Training Y‑Net – Seed {config.seed}")

    # ---------- Datensätze & Loader -------------------------------------
    train_ds = prepare_y_dataset(config.train_data, config.transform_train, config.num_samples)
    val_ds   = prepare_y_dataset(config.val_data,   config.transform_val,   config.val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size = config.batch_size,
        shuffle    = True,
        num_workers= config.num_workers,
        pin_memory = config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = config.batch_size,
        shuffle    = False,
        num_workers= config.num_workers,
        pin_memory = config.pin_memory,
    )

    # ---------- Debug: Mask‑Shapes --------------------------------------
    noisy_dbg, lr_dbg, _, mask_dbg = next(iter(train_loader))
    logger.info(f"[DEBUG] first mask batch shape = {mask_dbg.shape}")

    # ---------- Device & Modell ----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer‑Y] Using device: {device}")
    model = YNet2D(
        in_ch_noisy = config.in_channels_noisy,
        in_ch_lr    = config.in_channels_lr,
        out_channels= config.out_channels,
        features    = config.features,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ---------- Checkpoints‑Ordner --------------------------------------
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_ynet.pt")
    last_ckpt_path = os.path.join(config.checkpoint_dir, "last_ynet.pt")

    best_val_loss = float("inf")

    # ---------- Training‑Loop ------------------------------------------
    for epoch in range(1, config.epochs + 1):
        # --- Training ------------------------------------------------
        model.train(); running_loss = 0.0
        for noisy, lowrank, tgt, mask in train_loader:
            noisy   = noisy.to(device)
            lowrank = lowrank.to(device)
            tgt     = tgt.to(device)
            mask    = mask.to(device)

            optimizer.zero_grad()
            pred = model(noisy, lowrank)
            loss = masked_mse_loss(pred, tgt, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)
        avg_train = running_loss / len(train_loader.dataset)

        # --- Validation ----------------------------------------------
        model.eval(); running_val = 0.0
        with torch.no_grad():
            for noisy, lowrank, tgt, mask in val_loader:
                noisy   = noisy.to(device)
                lowrank = lowrank.to(device)
                tgt     = tgt.to(device)
                mask    = mask.to(device)

                pred = model(noisy, lowrank)
                running_val += masked_mse_loss(pred, tgt, mask).item() * noisy.size(0)
        avg_val = running_val / len(val_loader.dataset)

        # --- Logging --------------------------------------------------
        logger.info(f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e}")
        print (f"Epoch {epoch:03d} | train {avg_train:.3e} | val {avg_val:.3e}")

        # --- Best Checkpoint -----------------------------------------
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch"        : epoch,
                "model_state"  : model.state_dict(),
                "optim_state"  : optimizer.state_dict(),
                "val_loss"     : best_val_loss,
            }, best_ckpt_path)
            logger.info(f"   ↳ NEW BEST (val={best_val_loss:.4e}) → {best_ckpt_path}")

        # --- Last Checkpoint -----------------------------------------
        torch.save({
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "val_loss"    : avg_val,
        }, last_ckpt_path)

    logger.info(f"Training beendet · best_val={best_val_loss:.4e}")
    logger.info(f"   ↳ Finales Modell (last_ynet.pt) gespeichert")


if __name__ == "__main__":
    train_ynet_n2v()

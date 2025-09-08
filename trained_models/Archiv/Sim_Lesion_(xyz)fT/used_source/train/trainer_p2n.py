
# train/trainer_p2n.py
# -----------------------------------------------------------------------------
# Positive2Negative (P2N) Trainer mit optionaler Data-Consistency (MSE).
# Basierend auf deinem Noise2Void-Setup; Dataset liefert (inp, tgt, mask),
# wir verwenden nur inp.
# -----------------------------------------------------------------------------

import os
import sys
import random
import logging
from typing import List

# Projekt-Root ins PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # enthält Hyperparameter

# Sichtbare GPU setzen (wie gewohnt)
os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(config, "GPU_NUMBER", 0))

import numpy as np
import torch
from torch.utils.data import DataLoader

# Projekt-Module
from data.data_utils import load_and_preprocess_data
from data.mrsi_2d_dataset import MRSiNDataset
from losses.p2n_loss import p2n_total_loss
from models.unet2d import UNet2D


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train_p2n.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset(folders: List[str], transform, num_samples: int):
    data = load_and_preprocess_data(
        folder_names=folders,
        base_path="datasets",
        fourier_axes=getattr(config, "fourier_transform_axes", None),
        normalize=getattr(config, "normalize", True),
    )
    return MRSiNDataset(
        data=data,
        image_axes=getattr(config, "image_axes", None),
        fixed_indices=getattr(config, "fixed_indices", None),
        transform=transform,
        num_samples=num_samples,
    )


def make_sigmas(shape, sigma_mean=1.0, sigma_std=0.1, device=None):
    """RDC-Skalierungsfaktoren."""
    return torch.randn(shape, device=device) * sigma_std + sigma_mean


# -----------------------------------------------------------------------------
# Haupt-Trainingsfunktion
# -----------------------------------------------------------------------------
def train_p2n():
    set_seed(getattr(config, "seed", 0))
    logger.info(f"Start P2N-Training – Seed {getattr(config, 'seed', 0)}")

    # Datasets / Loader -------------------------------------------------------
    train_ds = prepare_dataset(
        getattr(config, "train_data", []),
        getattr(config, "transform_train", None),
        getattr(config, "num_samples", None),
    )
    val_ds = prepare_dataset(
        getattr(config, "val_data", []),
        getattr(config, "transform_val", None),
        getattr(config, "val_samples", None),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=getattr(config, "batch_size", 4),
        shuffle=True,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=getattr(config, "pin_memory", False),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=getattr(config, "batch_size", 4),
        shuffle=False,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=getattr(config, "pin_memory", False),
    )

    # Debug: Mask-Shape loggen ------------------------------------------------
    try:
        first = next(iter(train_loader))
        if len(first) == 3:
            _, _, mask_dbg = first
            logger.info(f"[DEBUG] Dataset liefert Masken-Shape {tuple(mask_dbg.shape)} – wird ignoriert.")
    except StopIteration:
        logger.warning("Train Loader leer? Keine Batches verfügbar.")

    # Device & Modell ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer-P2N] Using device: {device}")

    model = UNet2D(
        getattr(config, "in_channels", 1),
        getattr(config, "out_channels", 1),
        getattr(config, "features", [64, 128, 256, 512]),
    ).to(device)

    # Optional: Pretrained (z.B. N2V)
    if getattr(config, "pretrained_ckpt", None):
        if os.path.isfile(config.pretrained_ckpt):
            ckpt = torch.load(config.pretrained_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Geladene Vortrainings-Gewichte: {config.pretrained_ckpt}")
        else:
            logger.warning(f"Pretrained Checkpoint nicht gefunden: {config.pretrained_ckpt}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(config, "lr", 1e-4),
        weight_decay=getattr(config, "weight_decay", 0.0),
    )

    # Checkpoints --------------------------------------------------------------
    ckpt_dir = getattr(config, "checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
    best_val_loss = float("inf")

    # Hyperparameter -----------------------------------------------------------
    sigma_mean = getattr(config, "sigma_mean", 1.0)
    sigma_std  = getattr(config, "sigma_std", 0.1)
    lambda_dc  = getattr(config, "lambda_dc", 0.0)     # Gewicht Data Consistency (MSE)
    lambda_dcs = getattr(config, "lambda_dcs", 1.0)    # Gewicht P2N/DCS
    dcs_p      = getattr(config, "dcs_p", 1.5)         # p-Norm in DCS; 2.0 => L2
    use_data_consistency = getattr(config, "use_data_consistency", False)
    dc_mode    = "mse"  # fest, da du MSE willst; ignoriert config.dc_mode falls vorhanden
    detach_rdc = getattr(config, "detach_rdc", True)   # Grad-Cut vor RDC

    # Epoch-Loop ---------------------------------------------------------------
    for epoch in range(1, getattr(config, "epochs", 1) + 1):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0
        train_loss_dc = 0.0
        train_loss_dcs = 0.0

        for batch in train_loader:
            inp = batch[0].to(device)  # (B,C,H,W)

            # 1) Grundschätzung
            x_hat = model(inp)
            n_hat = inp - x_hat

            if detach_rdc:
                x_hat_det = x_hat.detach()
                n_hat_det = n_hat.detach()
            else:
                x_hat_det = x_hat
                n_hat_det = n_hat

            # 2) RDC
            sigma_p = make_sigmas(n_hat_det.shape, sigma_mean, sigma_std, device=device)
            sigma_n = make_sigmas(n_hat_det.shape, sigma_mean, sigma_std, device=device)
            y_p = x_hat_det + sigma_p * n_hat_det
            y_n = x_hat_det - sigma_n * n_hat_det

            # 3) Konsistenz-Forward
            x_p = model(y_p)
            x_n = model(y_n)

            # 4) Loss
            total, loss_dc, loss_dcs = p2n_total_loss(
                x_pos=x_p,
                x_neg=x_n,
                x_hat=x_hat,
                inp=inp,
                lambda_dc=(lambda_dc if use_data_consistency else 0.0),
                lambda_dcs=lambda_dcs,
                p=dcs_p,
                mode=dc_mode,  # MSE
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            bs = inp.size(0)
            train_loss     += total.item()   * bs
            train_loss_dc  += loss_dc.item() * bs
            train_loss_dcs += loss_dcs.item()* bs

        n_train = len(train_loader.dataset)
        avg_train     = train_loss     / n_train
        avg_train_dc  = train_loss_dc  / n_train
        avg_train_dcs = train_loss_dcs / n_train

        # ------------------- VALIDATION -------------------
        model.eval()
        val_loss = 0.0
        val_loss_dc = 0.0
        val_loss_dcs = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inp_val = batch[0].to(device)

                x_hat_v = model(inp_val)
                n_hat_v = inp_val - x_hat_v

                if detach_rdc:
                    x_hat_vd = x_hat_v.detach()
                    n_hat_vd = n_hat_v.detach()
                else:
                    x_hat_vd = x_hat_v
                    n_hat_vd = n_hat_v

                sigma_p_v = make_sigmas(n_hat_vd.shape, sigma_mean, sigma_std, device=device)
                sigma_n_v = make_sigmas(n_hat_vd.shape, sigma_mean, sigma_std, device=device)
                y_p_v = x_hat_vd + sigma_p_v * n_hat_vd
                y_n_v = x_hat_vd - sigma_n_v * n_hat_vd

                x_p_v = model(y_p_v)
                x_n_v = model(y_n_v)

                total_v, loss_dc_v, loss_dcs_v = p2n_total_loss(
                    x_pos=x_p_v,
                    x_neg=x_n_v,
                    x_hat=x_hat_v,
                    inp=inp_val,
                    lambda_dc=(lambda_dc if use_data_consistency else 0.0),
                    lambda_dcs=lambda_dcs,
                    p=dcs_p,
                    mode=dc_mode,  # MSE
                )

                bs = inp_val.size(0)
                val_loss     += total_v.item()    * bs
                val_loss_dc  += loss_dc_v.item()  * bs
                val_loss_dcs += loss_dcs_v.item() * bs

        n_val = len(val_loader.dataset)
        avg_val     = val_loss     / n_val
        avg_val_dc  = val_loss_dc  / n_val
        avg_val_dcs = val_loss_dcs / n_val

        # Logging -------------------------------------------------------------
        logger.info(
            f"Epoch {epoch:03d} · "
            f"train={avg_train:.4e} (dc={avg_train_dc:.4e}, dcs={avg_train_dcs:.4e}) · "
            f"val={avg_val:.4e} (dc={avg_val_dc:.4e}, dcs={avg_val_dcs:.4e})"
        )
        print(
            f"[Ep {epoch:03d}] train {avg_train:.4e} | val {avg_val:.4e} "
            f"(dc {avg_val_dc:.4e} dcs {avg_val_dcs:.4e})"
        )

        # Checkpoints ---------------------------------------------------------
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

    logger.info(f"P2N-Training fertig · best_val={best_val_loss:.4e}")
    print(f"Training done. Best val loss: {best_val_loss:.4e}")


# -----------------------------------------------------------------------------
# Script Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_p2n()

# ─────────────────────── train/trainer3d.py  (Noise2Void / Noise2Self, 3D) ───────────────────────
import os, sys, random, logging
from typing import List
import numpy as np
import torch
import torch.nn.utils as tnn_utils
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Projekt-Root & Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_NUMBER

# -----------------------------------------------------------------------------
# Self-supervision umschalten  ("n2v" | "n2s")
# -----------------------------------------------------------------------------
SELF_SUPERVISED_MODE = getattr(config, "SELF_SUPERVISED_MODE", "n2v")
assert SELF_SUPERVISED_MODE in ("n2v", "n2s"), "SELF_SUPERVISED_MODE muss 'n2v' oder 'n2s' sein"

# -----------------------------------------------------------------------------
# Projekt-Module (3D)
# -----------------------------------------------------------------------------
from data.data_utils         import load_and_preprocess_data
from data.mrsi_3d_dataset    import MRSi3DDataset
from data.transforms_3d      import StratifiedPixelSelection3D   # N2V-Maske
from losses.n2v_loss         import masked_mse_loss
from models.unet3d           import UNet3D                       # 3D-UNet

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train3d.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare_dataset(folders: List[str], transform, num_samples: int):
    data = load_and_preprocess_data(
        folder_names = folders,
        base_path    = "datasets",
        fourier_axes = config.fourier_transform_axes
    )
    return MRSi3DDataset(
        data          = data,
        image_axes    = config.image_axes_3d,      # z,f,t – typ. (2,3,4)
        fixed_indices = config.fixed_indices,
        transform     = transform,
        num_samples   = num_samples,
        phase_prob    = config.phase_prob,
    )

# -----------------------------------------------------------------------------
# Noise2Self: Bernoulli-Maske on-the-fly (für 3D-Tensoren)
# -----------------------------------------------------------------------------
def sample_n2s_mask(shape, p=0.03):
    """
    shape  = (B,1,Z,F,T); 1 = Loss-Pixel, 0 = Input-Pixel
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.rand(shape, device=device) < p).float()

# -----------------------------------------------------------------------------
# Haupt-Trainer
# -----------------------------------------------------------------------------
def train():
    set_seed(config.seed)
    logger.info(f"Start Training 3D ({SELF_SUPERVISED_MODE.upper()}) – Seed {config.seed}")

    # ─ Daten ──────────────────────────────────────────────────────────────
    train_ds = prepare_dataset(config.train_data, config.transform_train_3d, config.num_samples)
    val_ds   = prepare_dataset(config.val_data  , config.transform_val_3d  , config.val_samples)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader   = DataLoader(
        val_ds  , batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory)

    # ─ Debug-Maske ───────────────────────────────────────────────────────
    if SELF_SUPERVISED_MODE == "n2v":
        try:
            _, _, mask_dbg = next(iter(train_loader))
            logger.info(f"[DEBUG] N2V-Maske shape {tuple(mask_dbg.shape)}, mean {mask_dbg.float().mean():.4f}")
        except StopIteration:
            logger.warning("[DEBUG] Train Loader leer – keine Maske geloggt.")

    # ─ Modell / Optimizer / Scheduler ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(config.in_channels, config.out_channels, config.features_3d).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    def lr_lambda(epoch):
        lr = config.init_lr * (config.factor ** (epoch // config.step_size))
        return max(lr, config.min_lr) / config.init_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ─ Optional: Vortrainierte Gewichte ──────────────────────────────────
    ckpt_path = getattr(config, "pretrained_ckpt", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        strict_flag = getattr(config, "pretrained_strict", True)
        model.load_state_dict(state_dict, strict=strict_flag)
        logger.info(f"Gewichte geladen aus {ckpt_path}")
        if getattr(config, "load_optimizer_from_pretrained", False) and "optimizer_state" in ckpt:
            try:
                optim.load_state_dict(ckpt["optimizer_state"])
                logger.info("Optimizer-State mitgeladen.")
            except Exception as e:
                logger.warning(f"Optimizer-State konnte nicht geladen werden: {e}")

    # ─ Lactat-Maske (einmalig) ───────────────────────────────────────────
    F_dim            = train_ds[0][0].shape[2]
    l_low, l_high    = config.lact_bins
    lactat_mask      = torch.zeros(1, 1, 1, F_dim, 1, device=device)
    lactat_mask[..., l_low:l_high, :] = 1  # 1 auf Lactat-Bins, sonst 0

    # ─ Checkpoints ───────────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best3d.pt")
    last_ckpt = os.path.join(config.checkpoint_dir, "last3d.pt")
    best_val  = float("inf")

    # ─ Training-Loop ─────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        # Gewicht dieser Epoche
        w = config.lact_weight(epoch)
        weight_map = 1 + (w - 1) * lactat_mask      # (1,1,1,F,1)

        # ---- TRAIN ------------------------------------------------------
        model.train(); running = 0.0
        for inp, tgt, mask_n2v in train_loader:
            inp, tgt, mask_n2v = inp.to(device), tgt.to(device), mask_n2v.to(device)

            if SELF_SUPERVISED_MODE == "n2s":
                mask = sample_n2s_mask(inp[:, :1].shape, p=0.03)
                inp_masked = inp * (1 - mask)
                loss = masked_mse_loss(
                    model(inp_masked), tgt, mask,
                    weight=weight_map.expand_as(tgt))
            else:  # n2v
                loss = masked_mse_loss(
                    model(inp), tgt, mask_n2v,
                    weight=weight_map.expand_as(tgt))

            optim.zero_grad()
            loss.backward()
            tnn_utils.clip_grad_norm_(model.parameters(), 1.0)  # Safety
            optim.step()
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
                    loss = masked_mse_loss(
                        model(inp_masked), tgt, mask,
                        weight=weight_map.expand_as(tgt))
                else:
                    loss = masked_mse_loss(
                        model(inp), tgt, mask_n2v,
                        weight=weight_map.expand_as(tgt))

                running += loss.item() * inp.size(0)

        avg_val = running / len(val_loader.dataset)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e} · "
                    f"lr={current_lr:.2e} · w={w}")
        print(f"[Ep {epoch:03d}] train={avg_train:.4e} val={avg_val:.4e}")

        # ---- Checkpoints -----------------------------------------------
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(), "val_loss": best_val}, best_ckpt)
            logger.info("   ↳ NEW BEST")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(), "val_loss": avg_val}, last_ckpt)

    logger.info(f"Training fertig · best_val={best_val:.4e}")
    print(f"Training fertig · best_val={best_val:.4e}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train()




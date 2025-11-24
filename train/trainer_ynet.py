# ───────────────────────── train/trainer_ynet.py  (Y-Net, 2-D) ─────────────────────────
import os, sys, random, logging
from typing import List
import numpy as np
import torch
import torch.nn.utils as tnn_utils
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
# Self-supervision  ("n2v" | "n2s")
# ---------------------------------------------------------------------------
SELF_SUPERVISED_MODE = getattr(config, "SELF_SUPERVISED_MODE", "n2v")
assert SELF_SUPERVISED_MODE in ("n2v", "n2s")

# ---------------------------------------------------------------------------
# Projekt-Module
# ---------------------------------------------------------------------------
from data.data_utils        import load_and_preprocess_data
from data.mrsi_y_dataset    import MRSiYDataset
from data.transforms        import StratifiedPixelSelection  # falls genutzt
from losses.n2v_loss        import masked_mse_loss, combined_loss_simple
from models.ynet2d          import YNet2D

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.log_dir, "train_ynet.log"),
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

def _load_array_from_folders(folders: List[str]):
    """Lädt EIN Array aus einer Liste von Foldernamen (wie im U-Net-Trainer)."""
    return load_and_preprocess_data(
        folder_names = folders,
        base_path    = "datasets",
        fourier_axes = config.fourier_transform_axes
    )

def prepare_y_dataset(
    noisy_folders: List[str],
    lr_folders: List[str],
    transform,
    num_samples: int
):
    """Lädt Noisy und LowRank aus je EIGENEN Pfadlisten und baut MRSiYDataset."""
    noisy   = _load_array_from_folders(noisy_folders)
    lowrank = _load_array_from_folders(lr_folders)

    assert noisy.shape == lowrank.shape, (
        f"Noisy/LR Shapes ungleich: {noisy.shape} vs {lowrank.shape}"
    )

    return MRSiYDataset(
        noisy_data    = noisy,
        lowrank_data  = lowrank,
        image_axes    = config.image_axes,       # z.B. (f, T)
        fixed_indices = config.fixed_indices,
        transform     = transform,
        num_samples   = num_samples,
        phase_prob    = getattr(config, "phase_prob", 0.0),
    )

# ---------------------------------------------------------------------------
# Noise2Self-Maske (nur falls n2s benutzt wird)
# ---------------------------------------------------------------------------
def sample_n2s_mask(shape, p=0.03):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.rand(shape, device=device) < p).float()

# ---------------------------------------------------------------------------
# Haupt-Trainer
# ---------------------------------------------------------------------------
def train_ynet():
    set_seed(config.seed)
    logger.info(f"Start Training Y-Net 2D ({SELF_SUPERVISED_MODE.upper()}) – Seed {config.seed}")

    # ─ Daten ───────────────────────────────────────────────────────────
    train_ds = prepare_y_dataset(
        noisy_folders = config.train_data_noisy,
        lr_folders    = config.train_data_lr,
        transform     = config.transform_train,
        num_samples   = config.num_samples
    )
    val_ds = prepare_y_dataset(
        noisy_folders = config.val_data_noisy,
        lr_folders    = config.val_data_lr,
        transform     = config.transform_val,
        num_samples   = config.val_samples
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader   = DataLoader(
        val_ds  , batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory)

    print(f"[Y-Net] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Train samples (virtual) = {len(train_ds)} · Val samples (virtual) = {len(val_ds)}")
    logger.info(f"Batch size = {config.batch_size} · num_workers = {config.num_workers}")

    # ─ Modell / Optimizer / Scheduler ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = YNet2D(
        in_ch_noisy = getattr(config, "in_channels_noisy", 2),
        in_ch_lr    = getattr(config, "in_channels_lr", 2),
        out_channels= config.out_channels,
        features    = config.features,
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optim  = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    def lr_lambda(epoch):
        lr = config.init_lr * (config.factor ** (epoch // config.step_size))
        return max(lr, config.min_lr) / config.init_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ─ Optional: Preload aus config ────────────────────────────────────
    ckpt_path = getattr(config, "pretrained_ckpt", None)
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            strict = getattr(config, "pretrained_strict", True)
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict)
            logger.info(
                f"[Preload] Loaded '{ckpt_path}' "
                f"(epoch={ckpt.get('epoch','?')}, val={ckpt.get('val_loss','?')}) "
                f"missing={missing}, unexpected={unexpected}, strict={strict}"
            )

            if getattr(config, "load_optimizer_from_pretrained", False) and "optimizer_state" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optimizer_state"])
                    for state in optim.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    logger.info("[Preload] Optimizer state loaded.")
                except Exception as e:
                    logger.warning(f"[Preload] Could not load optimizer state: {e}")
        except Exception as e:
            logger.exception(f"[Preload] Failed to load checkpoint '{ckpt_path}': {e}")
    else:
        logger.info("[Preload] No valid pretrained_ckpt found; skipping.")

    # ─ Sanity-Fetch + Dry-Run (mit 5-Tupel) ────────────────────────────
    DO_DRY_RUN = True
    if DO_DRY_RUN:
        try:
            b_noisy, b_lr, b_tgt, b_mask, b_lr_tgt = next(iter(train_loader))
            logger.info(f"[Sanity] First batch shapes: "
                        f"noisy={tuple(b_noisy.shape)}, lr={tuple(b_lr.shape)}, "
                        f"tgt_noisy={tuple(b_tgt.shape)}, mask={tuple(b_mask.shape)}, "
                        f"tgt_lr={tuple(b_lr_tgt.shape)}")

            model.train()
            b_noisy  = b_noisy.to(device)
            b_lr     = b_lr.to(device)
            b_tgt    = b_tgt.to(device)
            b_mask   = b_mask.to(device)
            b_lr_tgt = b_lr_tgt.to(device)

            pred = model(b_noisy, b_lr)
            loss = combined_loss_simple(
                y_hat    = pred,
                x_raw    = b_tgt,      # UNmaskiertes noisy-Target
                x_tmppca = b_lr_tgt,   # UNmaskiertes lowrank-Target
                B        = b_mask,
                alpha    = getattr(config, "alpha_tmppca", 1.0)
            )
            optim.zero_grad(); loss.backward(); optim.step()
            logger.info(f"[Dry-Run] ok · loss={loss.item():.3e} · batch={b_noisy.size(0)}")
        except Exception as e:
            logger.exception("[Dry-Run] Fehler")
            raise

    # ─ Checkpoints ─────────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best_ynet.pt")
    last_ckpt = os.path.join(config.checkpoint_dir, "last_ynet.pt")
    best_val  = float("inf")

    # ─ Training-Loop ───────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        try:
            # ---- TRAIN ------------------------------------------------
            model.train(); running = 0.0
            for noisy, lowrank, tgt, mask_n2v, tgt_lr in train_loader:
                noisy    = noisy.to(device)
                lowrank  = lowrank.to(device)
                tgt      = tgt.to(device)
                tgt_lr   = tgt_lr.to(device)   # wichtig
                mask_n2v = mask_n2v.to(device)

                if SELF_SUPERVISED_MODE == "n2s":
                    n2s_mask   = sample_n2s_mask(noisy[:, :1].shape, p=getattr(config, "n2s_p", 0.03))
                    noisy_mskd = noisy * (1 - n2s_mask)
                    pred = model(noisy_mskd, lowrank)
                    loss = masked_mse_loss(pred, tgt, n2s_mask)
                else:
                    pred = model(noisy, lowrank)
                    loss = combined_loss_simple(
                        y_hat    = pred,
                        x_raw    = tgt,       # UNmaskiertes noisy-Target
                        x_tmppca = tgt_lr,    # UNmaskiertes lowrank-Target
                        B        = mask_n2v,
                        alpha    = getattr(config, "alpha_tmppca", 1.0)
                    )

                optim.zero_grad(); loss.backward()
                tnn_utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                running += loss.item() * noisy.size(0)

            avg_train = running / len(train_loader.dataset)

            # ---- VALID ------------------------------------------------
            model.eval(); running = 0.0
            with torch.no_grad():
                for noisy, lowrank, tgt, mask_n2v, tgt_lr in val_loader:
                    noisy    = noisy.to(device)
                    lowrank  = lowrank.to(device)
                    tgt      = tgt.to(device)
                    tgt_lr   = tgt_lr.to(device)   # wichtig
                    mask_n2v = mask_n2v.to(device)

                    if SELF_SUPERVISED_MODE == "n2s":
                        n2s_mask   = sample_n2s_mask(noisy[:, :1].shape, p=getattr(config, "n2s_p", 0.03))
                        noisy_mskd = noisy * (1 - n2s_mask)
                        pred = model(noisy_mskd, lowrank)
                        loss = masked_mse_loss(pred, tgt, n2s_mask)
                    else:
                        pred = model(noisy, lowrank)
                        loss = combined_loss_simple(
                            y_hat    = pred,
                            x_raw    = tgt,       # UNmaskiertes noisy-Target
                            x_tmppca = tgt_lr,    # UNmaskiertes lowrank-Target
                            B        = mask_n2v,
                            alpha    = getattr(config, "alpha_tmppca", 1.0)
                        )

                    running += loss.item() * noisy.size(0)

            avg_val = running / len(val_loader.dataset)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            logger.info(f"Epoch {epoch:03d} · train={avg_train:.4e} · val={avg_val:.4e} · lr={current_lr:.2e}")
            print(f"[Ep {epoch:03d}] train={avg_train:.4e} val={avg_val:.4e}")

            # ---- Checkpoints -----------------------------------------
            if avg_val < best_val:
                best_val = avg_val
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optimizer_state": optim.state_dict(), "val_loss": best_val}, best_ckpt)
                logger.info("   ↳ NEW BEST")

            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(), "val_loss": avg_val}, last_ckpt)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[Epoch {epoch}] CUDA OOM – batch_size={config.batch_size}. "
                             "Bitte batch_size reduzieren oder features verkleinern.")
            else:
                logger.exception(f"[Epoch {epoch}] Fehler")
            raise

    logger.info(f"Training fertig · best_val={best_val:.4e}")
    print(f"Training fertig · best_val={best_val:.4e}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_ynet()





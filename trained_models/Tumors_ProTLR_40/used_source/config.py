# ───────────────────────── config.py (ergänzt für 3-D) ─────────────────────────
import os
# ---------------------------------------------------------------------------
# 0) Welche Trainings-Variante?
# ---------------------------------------------------------------------------
SELF_SUPERVISED_MODE = "n2v"          #  n2v | n2s

# ---------------------------------------------------------------------------
# 1) Architektur / Trainer
# ---------------------------------------------------------------------------
UNET_DIM     = "3d"                   #  2d | 3d
TRAIN_METHOD = "n2v"                  #  n2v | n2s | …

lowrank_rank = 20
basis_file   = f"datasets/basis_rank{lowrank_rank}.npy"

SWAP_MODE    = "both"
supervised_to_lowrank = False

phase_prob = 1.0  # data augmentation probabiliy with random global phase 

# ---------------------------------------------------------------------------
# 1a) Trainer-Mapping  (NEU: 3-D-Einträge)
# ---------------------------------------------------------------------------
_TRAINER_MAP = {
    ("n2v", "2d"): ("train.trainer_n2v", "train"),
    ("n2s", "2d"): ("train.trainer_n2v", "train"),
    ("n2v", "3d"): ("train.trainer3d",   "train"),   # ← NEU
    ("n2s", "3d"): ("train.trainer3d",   "train"),   # ← NEU
    # weitere Einträge unverändert …
}

TRAINER_MODULE, TRAIN_FUNC = _TRAINER_MAP[(TRAIN_METHOD, UNET_DIM)]

# ---------------------------------------------------------------------------
# 2) GPU & Ordner (unverändert)
# ---------------------------------------------------------------------------
GPU_NUMBER = "1"
RUN_NAME   = "Tumors_ProTLR_40"
BASE_RUN_DIR  = "trained_models"

run_dir        = os.path.join(BASE_RUN_DIR, RUN_NAME)
checkpoint_dir = os.path.join(run_dir, "checkpoints")
log_dir        = os.path.join(run_dir, "logs")

# ---------------------------------------------------------------------------
# 3) Daten-Setup
# ---------------------------------------------------------------------------
seed       = 42
train_data = ["Tumor_1_normalized_proTLR40"]
val_data   = ["Tumor_1_normalized_proTLR40"]

# --- Achsen‐Definition ------------------------------------------------------
if UNET_DIM == "2d":
    image_axes  = (3, 4)          # (f , T)
    volume_axes = None
else:                             # 3-D-Netz
    image_axes_3d = (2, 3, 4)     # (Z , f , T)  ← von trainer3d genutzt
    volume_axes   = image_axes_3d

fourier_transform_axes = [3]      # FFT über FID-Achse (t)
fixed_indices = None
num_samples   = 2000
val_samples   = 400

# ---------------------------------------------------------------------------
# 4) Maskierung
# ---------------------------------------------------------------------------
if SELF_SUPERVISED_MODE == "n2v":
    if UNET_DIM == "2d":
        from data.transforms import StratifiedPixelSelection
        transform_train = StratifiedPixelSelection(
            num_masked_pixels=20,
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        transform_val = StratifiedPixelSelection(
            num_masked_pixels=20,
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        # Platzhalter, damit Trainer immer ein Attribut findet
        transform_train_3d = transform_val_3d = None
    else:  # 3-D
        from data.transforms_3d import StratifiedPixelSelection3D
        transform_train_3d = StratifiedPixelSelection3D(
            num_masked_pixels=1,            # nur (f,T)-Pixels – werden über Z dupliziert
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        transform_val_3d = StratifiedPixelSelection3D(
            num_masked_pixels=1,
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        # Platzhalter für 2-D-Trainer
        transform_train = transform_val = None
else:
    transform_train = transform_val = None
    transform_train_3d = transform_val_3d = None

# ---------------------------------------------------------------------------
# 5) Netzwerk-Hyper­parameter
# ---------------------------------------------------------------------------
batch_size  = 120  #160
num_workers = 0
pin_memory  = False

in_channels   = 2
out_channels  = 2
features      = (32, 64, 128, 256, 512)             # 2-D-UNet
features_3d   = (32, 64, 128, 256, 512)              # 3-D-UNet  ← neu

# ---- Hyper­parameter ------------------------------------
init_lr   = 1e-3          # Anfangs learning rate
factor    = 0.5          # alle 150 Epochen /4
step_size = 50           # so oft wird die learning rate angepasst 
min_lr    = 1e-3          # learnign rate wir nie niedriger als das

# ───────────────────────────────
# Laktat-Gewicht (Curriculum)
# ───────────────────────────────
def lact_weight(epoch: int) -> float:
    if epoch < 50: return 1
    elif epoch < 100: return 1
    elif epoch < 150: return 1
    elif epoch < 200: return 1
    elif epoch < 250: return 1
    elif epoch < 300: return 1

    else: return 1.0

# ----------------- Frequenz-Bins (inklusive Endindex) ------
lact_bins = (95, 120)       # 12..16   (an Daten anpassen!)

epochs = 500

# Vortrainiertes Modell (optional)
pretrained_ckpt = ""   # path/to/ckpt.pt
pretrained_strict = True
load_optimizer_from_pretrained = False





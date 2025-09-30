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
IMAGE_MODE = "xyf"          #  "zft"  → (Z,f,T)  |  "xyf" → (X,Y,f) nur für 3D relevant

lowrank_rank = 20
basis_file   = f"datasets/basis_rank{lowrank_rank}.npy"

SWAP_MODE    = "both"
supervised_to_lowrank = False
phase_prob = 1.0  # data augmentation probabiliy with random global phase 

# ---------------------------------------------------------------------------
# 2) GPU & Ordner (unverändert)
# ---------------------------------------------------------------------------
GPU_NUMBER = "0"
RUN_NAME   = "xyf"
BASE_RUN_DIR  = "trained_models"

run_dir        = os.path.join(BASE_RUN_DIR, RUN_NAME)
checkpoint_dir = os.path.join(run_dir, "checkpoints")
log_dir        = os.path.join(run_dir, "logs")

# ---------------------------------------------------------------------------
# 3) Daten-Setup
# ---------------------------------------------------------------------------
seed       = 42
train_data = [f"Simulated_Lesion_{i}" for i in range(1, 6)]
val_data   = ["Simulated_Lesion_6"]

# --- Achsen‐Definition ------------------------------------------------------
if UNET_DIM == "2d":
    image_axes  = (3, 4)          # (f , T)
    volume_axes = None
else:                             # 3-D-Netz
    if IMAGE_MODE == "zft":
        image_axes_3d = (2, 3, 4)     # (Z , f , T)  ← von trainer3d genutzt
    else: # 'xyf
        image_axes_3d = (0, 1, 3) 
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
            num_masked_pixels=12,
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        transform_val = StratifiedPixelSelection(
            num_masked_pixels=12,
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        # Platzhalter, damit Trainer immer ein Attribut findet
        transform_train_3d = transform_val_3d = None
    else:  # 3-D
        if IMAGE_MODE == "zft":                # 2-D-Maske (f,T) → über Z repliziert
            from data.transforms_3d import StratifiedPixelSelection3D as _Mask3D
            mask_kwargs = dict(
                num_masked_pixels=12, window_size=3,
                random_mask_low_rank=False, random_mask_noisy=False,
                swap_mode=SWAP_MODE,
            )
        else:                                  # "xyf" → 1-D-Maske in f
            from data.transforms_3d import StratifiedFreqSelection3D as _Mask3D
            mask_kwargs = dict(num_masked_freq=2)

        transform_train_3d = _Mask3D(**mask_kwargs)
        transform_val_3d   = _Mask3D(**mask_kwargs)
        # Platzhalter für 2-D-Trainer
        transform_train = transform_val = None
else:
    transform_train = transform_val = None
    transform_train_3d = transform_val_3d = None

# ---------------------------------------------------------------------------
# 5) Netzwerk-Hyper­parameter
# ---------------------------------------------------------------------------
batch_size  = 125
num_workers = 0
pin_memory  = False

in_channels   = 2
out_channels  = 2
features      = (32, 64, 128, 256, 512)             # 2-D-UNet
features_3d   = (32, 64, 128, 256, 512)              # 3-D-UNet  ← neu

lr     = 1e-4
epochs = 1000

# --- Kaskaden-Schalter --------------------------------------------------
# trainier xyf - fT Netz in tandem!
# ---------------------------------------------------------------------------
TRAIN_MODE          = "cascade"          # "single" | "cascade" so kann man ein "doppel netz " trainieren xyf + fT
STAGE1_PRE_EPOCHS   = 150                # Vortrain für Net1
CYCLE_EPOCHS_NET1   = 5                  # Fine-Tune-Blöcke Net1
CYCLE_EPOCHS_NET2   = 5                  # Trainingsblöcke Net2
CACHE_DIR           = os.path.join(run_dir, "pred_cache")
CACHE_TRAIN         = os.path.join(CACHE_DIR, "net1_preds_train.npy")
CACHE_VAL           = os.path.join(CACHE_DIR, "net1_preds_val.npy")
NET1_BEST           = os.path.join(checkpoint_dir, "best3d.pt")
NET2_SUFFIX         = "_net2"            # hängt an RUN_NAME bei Kaskaden-Lauf

# Vortrainiertes Modell (optional)
pretrained_ckpt = ""   # path/to/ckpt.pt
pretrained_strict = True
load_optimizer_from_pretrained = False

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




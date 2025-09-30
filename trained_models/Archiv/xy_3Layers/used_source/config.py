import os
# ───────────────────────── config.py (NEU) ─────────────────────────
# ------------------------------------------------------------------
# Modell / Training
# ------------------------------------------------------------------
TRAIN_METHOD  = "n2v"        #  n2v | p2n | ynet
UNET_DIM      = "2d"         #  2d  | 3d          ←  NEU

# Mappe Dimensionalität → Trainer‐Modul + Funktion
_TRAINER_MAP = {
    ("n2v", "2d"): ("train.trainer_n2v",  "train_n2v"),
    ("n2v", "3d"): ("train.trainer_n2v_3D",    "train_n2v_3d"),
    ("p2n", "2d"): ("train.trainer_p2n",  "train_p2n"),      # Beispiel
    ("ynet","2d"): ("train.trainer_ynet", "train_ynet_n2v"),
}
TRAINER_MODULE, TRAIN_FUNC = _TRAINER_MAP[(TRAIN_METHOD, UNET_DIM)]

# ── GPU & Run-Ordner ───────────────────────────────────────────────
GPU_NUMBER = "2"
RUN_NAME   = "xy_3Layers"
BASE_RUN_DIR = "trained_models"

run_dir        = os.path.join(BASE_RUN_DIR, RUN_NAME)
checkpoint_dir = os.path.join(run_dir, "checkpoints")
log_dir        = os.path.join(run_dir, "logs")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir,        exist_ok=True)

# ── Vortrainiertes Modell (optional) ───────────────────────────────
pretrained_ckpt = False   # z. B. "/…/best.pt"

# ── Allgemeines Training-Setup ─────────────────────────────────────
seed = 42
train_data = ["P03","P04","P05","P06","P07"]
val_data   = ["P08"]

image_axes  = (0, 1)          # (Spektrum, Zeit) 
volume_axes  = (2, 3, 4)          # (Spektrum, Zeit) 
fixed_indices   = None
fourier_transform_axes = [3]  # FFT über Zeit
num_samples = 10000
val_samples = 2000


if UNET_DIM == '2d':
    from data.transforms import StratifiedPixelSelection
    transform_train = StratifiedPixelSelection(num_masked_pixels=8, window_size=3)
    transform_val   = StratifiedPixelSelection(num_masked_pixels=8, window_size=3)
else:
    from data.transforms_3d import StratifiedVoxelSelection as StratifiedTransform
    transform_train = StratifiedTransform(num_masked_voxels=264, window_size=3)
    transform_val   = StratifiedTransform(num_masked_voxels=264, window_size=3)  # oder ebenfalls StratifiedTransform

batch_size  = 500
num_workers = 0
pin_memory  = False

# ── Architektur-Parameter ──────────────────────────────────────────
# Klassisch (U-Net) – bleiben erhalten
in_channels  = 2
out_channels = 2
features     = (32, 64, 128)

# Y-Net-spezifisch (werden nur verwendet, wenn TRAIN_METHOD == \"ynet\")
in_channels_noisy = 2      # Real + Imag
in_channels_lr    = 2      # Real + Imag
lowrank_rank      = 3     # Truncation-Rank für SVD

# ── Optimierung ────────────────────────────────────────────────────
lr     = 2e-3
epochs = 1000



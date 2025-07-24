# ───────────────────────── config.py (NEU) ─────────────────────────
import os

# ── Trainingsmethode ───────────────────────────────────────────────
#  • "n2v"  -> klassisches Noise2Void (1-Encoder U-Net)
#  • "ynet" -> zweipfadiges Y-Net (Noisy + Low-Rank) n2v
#  • "p2n"  -> Positive2Negative (unet)
TRAIN_METHOD = "ynet"

# ── GPU & Run-Ordner ───────────────────────────────────────────────
GPU_NUMBER = "0"
RUN_NAME   = "fT_3Layers_ynetRank5"
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

image_axes  = (3, 4)          # (Spektrum, Zeit)
fixed_indices   = None
fourier_transform_axes = [3]  # FFT über Zeit
num_samples = 10000
val_samples = 2000

from data.transforms import StratifiedPixelSelection
transform_train = StratifiedPixelSelection(num_masked_pixels=12, window_size=3)
transform_val   = StratifiedPixelSelection(num_masked_pixels=12, window_size=3)

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
lowrank_rank      = 5     # Truncation-Rank für SVD

# ── Optimierung ────────────────────────────────────────────────────
lr     = 2e-5
epochs = 1000



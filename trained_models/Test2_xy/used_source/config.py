import os

# ── welche GPU nutzen ────────────────────────────────────────────────────────
GPU_NUMBER = "3"  

# ── Run-Identifier mit Datum/Uhrzeit ─────────────────────────────────────────
RUN_NAME = 'Test2_xy'

# ── Basis-Ordner für alle Runs ────────────────────────────────────────────────
BASE_RUN_DIR = 'trained_models'

# ── Run-spezifische Ordner ───────────────────────────────────────────────────
run_dir        = os.path.join(BASE_RUN_DIR, RUN_NAME)
checkpoint_dir = os.path.join(run_dir, 'checkpoints')
log_dir        = os.path.join(run_dir, 'logs')

# Ordner anlegen
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir,        exist_ok=True)

# ── Rest deiner Config ───────────────────────────────────────────────────────
seed = 42
train_data = ['P03','P04','P05','P06','P07']
val_data   = ['P08']
image_axes = (0, 1)
fixed_indices   = None
fourier_transform_axes = []
num_samples     = 10000
val_samples     = 2000

from data.transforms import StratifiedPixelSelection
transform_train = StratifiedPixelSelection(num_masked_pixels=12, window_size=21)
transform_val   = StratifiedPixelSelection(num_masked_pixels=12, window_size=21)

batch_size  = 500
num_workers = 0
pin_memory  = False

in_channels   = 2
out_channels  = 2
features      = (32, 64, 128)

lr           = 1e-3
epochs       = 1000


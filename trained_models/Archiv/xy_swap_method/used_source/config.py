import os

# ── Trainingsmethode: entweder "n2v" für Noise2Void oder "p2n" für Positive2Negative
TRAIN_METHOD = "n2v"    # <-- setze hier "n2v" oder "p2n"

# ── welche GPU nutzen ────────────────────────────────────────────────────────
GPU_NUMBER = "0"  

# ── Run-Identifier mit Datum/Uhrzeit ─────────────────────────────────────────
RUN_NAME = 'xy_swap_method'

# ── Basis-Ordner für alle Runs ────────────────────────────────────────────────
BASE_RUN_DIR = 'trained_models'

# ── Run-spezifische Ordner ───────────────────────────────────────────────────
run_dir        = os.path.join(BASE_RUN_DIR, RUN_NAME)
checkpoint_dir = os.path.join(run_dir, 'checkpoints')
log_dir        = os.path.join(run_dir, 'logs')

# Ordner anlegen
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir,        exist_ok=True)

# ── Vorheriges Model laden ───────────────────────────────────────────────────

pretrained_ckpt = False #"/workspace/Deuterium_Denosing/trained_models/Test1_fT/checkpoints/best.pt" # kann einfach auf False gesetzt werden 

# ── Rest deiner Config ───────────────────────────────────────────────────────
seed = 42
train_data = ['P03_xy_tT_swapped','P04_xy_tT_swapped','P05_xy_tT_swapped','P06_xy_tT_swapped','P07_xy_tT_swapped']
val_data   = ['P08_xy_tT_swapped']
image_axes = (3, 4)
fixed_indices   = None
fourier_transform_axes = []
num_samples     = 10000
val_samples     = 2000

from data.transforms import StratifiedPixelSelection
transform_train = StratifiedPixelSelection(num_masked_pixels=12, window_size=3)
transform_val   = StratifiedPixelSelection(num_masked_pixels=12, window_size=3)

batch_size  = 500
num_workers = 0
pin_memory  = False

in_channels   = 2
out_channels  = 2
features      = (32, 64, 128)

lr           = 2e-5
epochs       = 500


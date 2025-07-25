import os

# ── Trainingsmethode: entweder "n2v" für Noise2Void oder "p2n" für Positive2Negative
TRAIN_METHOD = "n2v"    # <-- setze hier "n2v" oder "p2n"

# ── welche GPU nutzen ────────────────────────────────────────────────────────
GPU_NUMBER = "1"  

# ── Run-Identifier mit Datum/Uhrzeit ─────────────────────────────────────────
RUN_NAME = 'xy_90p'

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
train_data = ['P03','P04','P05','P06','P07']
val_data   = ['P08']
image_axes = (0, 1)
fixed_indices   = None
fourier_transform_axes = []
num_samples     = 10000
val_samples     = 2000

from data.transforms import StratifiedPixelSelection
transform_train = StratifiedPixelSelection(num_masked_pixels=436, window_size=21)
transform_val   = StratifiedPixelSelection(num_masked_pixels=436, window_size=21)

batch_size  = 4000
num_workers = 0
pin_memory  = False

in_channels   = 2
out_channels  = 2
features      = (32, 64, 128)

lr           = 2e-3
epochs       = 700


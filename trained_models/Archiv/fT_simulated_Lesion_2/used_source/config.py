# ───────────────────────── config.py (aufgeräumt) ──────────────────────────
import os

# ---------------------------------------------------------------------------
# 0) Welche Trainings-Variante?
#    - SELF_SUPERVISED_MODE = "n2v"  … klassische Noise2Void
#    - SELF_SUPERVISED_MODE = "n2s"  … Zero-Mask Noise2Self (kein Pixeltausch)
# ---------------------------------------------------------------------------
SELF_SUPERVISED_MODE = "n2v"        #  n2v | n2s

# ---------------------------------------------------------------------------
# 1) Architektur / Trainer
# ---------------------------------------------------------------------------
UNET_DIM     = "2d"                 #  2d  | 3d
TRAIN_METHOD = "n2v"                #  n2v = unet | p2n (auch unet) | ynet_n2v (noise 2 void) | sup_lowrank

# Mögliche Werte: "both" (2D‑Fenster), "time" (nur Zeitachse), "freq" (nur Frequenzachse)
SWAP_MODE = "both"  

supervised_to_lowrank = False

# Mapping TRAIN_METHOD/UNET_DIM -> (module, func)
_TRAINER_MAP = {
    ("n2v",        "2d"): ("train.trainer_n2v",              "train"),
    ("n2s",        "2d"): ("train.trainer_n2v",              "train"),
    ("ynet_n2v",   "2d"): ("train.trainer_ynet",         "train_ynet_n2v"),
    ("sup_lowrank","2d"): ("train.trainer_unet_lowrank", "train_unet_lowrank"),  # <-- neu
}

# Ableiten (wirft KeyError, wenn Kombination fehlt -> gut zum Debuggen)
TRAINER_MODULE, TRAIN_FUNC = _TRAINER_MAP[(TRAIN_METHOD, UNET_DIM)]

# ---------------------------------------------------------------------------
# 2) GPU & Ordner
# ---------------------------------------------------------------------------
GPU_NUMBER = "0"
RUN_NAME   = "fT_simulated_Lesion_2"
BASE_RUN_DIR = "trained_models"

run_dir        = os.path.join(BASE_RUN_DIR, RUN_NAME)
checkpoint_dir = os.path.join(run_dir, "checkpoints")
log_dir        = os.path.join(run_dir, "logs")

# ---------------------------------------------------------------------------
# 3) Daten-Setup
# ---------------------------------------------------------------------------
seed       = 42
train_data = ["Simulated_Lesion_1","Simulated_Lesion_2","Simulated_Lesion_3","Simulated_Lesion_4","Simulated_Lesion_5"]
val_data   = ["Simulated_Lesion_6"]

# -> Welche Achsen bilden das Netz-Input?
if UNET_DIM == "2d":
    image_axes  = (3, 4)          # (f , T)  – nur relevant für 2-D
    volume_axes = None
else:                             # 3-D-Netz
    image_axes  = None
    volume_axes = (2, 3, 4)       # (Z , f , T)

fourier_transform_axes = [3]      # FFT über FID-Achse (t)
fixed_indices   = None
num_samples = 10000
val_samples = 2000

# ---------------------------------------------------------------------------
# 4) Kanal-Maskierung: Zufälliges Maskieren pro Branch
# ---------------------------------------------------------------------------
# Für jeden Branch (low-rank vs. noisy) wird pro Sample zufällig einer der Kanäle (real/imag) maskiert.
MASK_RANDOM_LOW_RANK = False   # low-rank branch: zufällig real oder imag maskieren
MASK_RANDOM_NOISY    = False   # noisy branch: zufällig real oder imag maskieren
# Falls auf False gesetzt wird werden immer gleichzeitig beide Kanäle maskiert

# 5) Maskierung (abhängig von UNET_DIM & SELF_SUPERVISED_MODE)
# ---------------------------------------------------------------------------
if SELF_SUPERVISED_MODE == "n2v":
    if UNET_DIM == "2d":
        from data.transforms import StratifiedPixelSelection
        transform_train = StratifiedPixelSelection(
            num_masked_pixels=15,
            window_size=3,
            random_mask_low_rank=MASK_RANDOM_LOW_RANK,
            random_mask_noisy=MASK_RANDOM_NOISY,
            swap_mode=SWAP_MODE,
        )
        transform_val = StratifiedPixelSelection(
            num_masked_pixels=15,
            window_size=3,
            random_mask_low_rank=MASK_RANDOM_LOW_RANK,
            random_mask_noisy=MASK_RANDOM_NOISY,
            swap_mode=SWAP_MODE,
        )
    else:
        from data.transforms_3d import StratifiedVoxelSelection as StratifiedTransform
        transform_train = StratifiedTransform(
            num_masked_voxels=8 * (21*96)//2,
            window_size=3,
            random_mask_low_rank=MASK_RANDOM_LOW_RANK,
            random_mask_noisy=MASK_RANDOM_NOISY,
        )
        transform_val = StratifiedTransform(
            num_masked_voxels=8 * (21*96)//2,
            window_size=3,
            random_mask_low_rank=MASK_RANDOM_LOW_RANK,
            random_mask_noisy=MASK_RANDOM_NOISY,
        )
else:
    transform_train = transform_val = None

# --- P2N Hyperparams ---
use_data_consistency = True   # False => reines P2N (kein DC)
lambda_dc  = 1.0              # Gewicht Data Consistency (MSE gegen inp)
lambda_dcs = 0.0              # Gewicht P2N/DCS; setz höher, wenn Konsistenz stärker zählen soll
dcs_p      = 2.0              # p-Norm in P2N (2.0 = L2)
sigma_mean = 1.0              # RDC: Mittelwert der Skalierung
sigma_std  = 0.1              # RDC: Streuung
detach_rdc = True             # stabiler; kein Grad zurück durch RDC

# ---------------------------------------------------------------------------
# 6) Training-Hyperparameter
# ---------------------------------------------------------------------------
batch_size  = 500
num_workers = 0
pin_memory  = False

in_channels  = 2
out_channels = 2
features     = (32, 64, 128, 256, 512)

#  Y-Net-Spezifisches (ignoriert bei n2v)
in_channels_noisy = 2
in_channels_lr    = 2
lowrank_rank      = 8

lr     = 2e-5
epochs = 1000

pretrained_ckpt = "" # choose path for pretrained model, must match mnodel dimensions





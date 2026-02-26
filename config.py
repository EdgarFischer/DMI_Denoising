# ───────────────────────── config.py (ergänzt für 3-D) ─────────────────────────
import os
# ---------------------------------------------------------------------------
# 0) Welche Trainings-Variante?
# ---------------------------------------------------------------------------
SELF_SUPERVISED_MODE = "n2v"          #  n2v | n2s | 

# ---------------------------------------------------------------------------
# 1) Architektur / Trainer test
# ---------------------------------------------------------------------------
UNET_DIM     = "2d"                   #  2d | 3d
TRAIN_METHOD = "n2v"                  #  n2v | n2s | ynet

lowrank_rank = 20
basis_file   = f"datasets/basis_rank{lowrank_rank}.npy"

SWAP_MODE    = "both"
supervised_to_lowrank = False

phase_prob = 1.0  # data augmentation probabiliy with random global phase 

# ---------------------------------------------------------------------------
# 1a) Trainer-Mapping  (NEU: 3-D-Einträge)
# ---------------------------------------------------------------------------
_TRAINER_MAP = {
    ("n2v", "2d"): ("denoising.training.trainers.trainer_n2v", "train"),
    ("n2s", "2d"): ("denoising.training.trainers.trainer_n2v", "train"),
    ("ynet", "2d"): ("denoising.training.trainers.trainer_ynet", "train_ynet"),
    ("n2v", "3d"): ("denoising.training.trainers.trainer3d", "train"),
    ("n2s", "3d"): ("denoising.training.trainers.trainer3d", "train"),
    # weitere Einträge unverändert …
}

TRAINER_MODULE, TRAIN_FUNC = _TRAINER_MAP[(TRAIN_METHOD, UNET_DIM)]

# ---------------------------------------------------------------------------
# 2) GPU & Ordner (unverändert)
# ---------------------------------------------------------------------------
GPU_NUMBER = "2"
RUN_NAME   = "ABC"
BASE_RUN_DIR  = "trained_models"

run_dir        = os.path.join(BASE_RUN_DIR, RUN_NAME)
checkpoint_dir = os.path.join(run_dir, "checkpoints")
log_dir        = os.path.join(run_dir, "logs")

# ---------------------------------------------------------------------------
# 3) Daten-Setup
# ---------------------------------------------------------------------------
seed       = 42
train_data = ["sf_brain_DMI_HC_pilot_normalized"]
val_data   = ["sf_brain_DMI_HC_pilot_normalized"]

# --- Achsen‐Definition ------------------------------------------------------
if UNET_DIM == "2d":
    image_axes  = (3, 4) #(3, 4)          # (f , T)
    volume_axes = None
else:                             # 3-D-Netz
    image_axes_3d = (2, 3, 4)     # (Z , f , T)  ← von trainer3d genutzt
    volume_axes   = image_axes_3d

fourier_transform_axes = [3]      # FFT über FID-Achse (t)
fixed_indices = None
num_samples   = 10000
val_samples   = 2000

# ---------------------------------------------------------------------------
# 4) Maskierung
# ---------------------------------------------------------------------------
if SELF_SUPERVISED_MODE == "n2v":
    if UNET_DIM == "2d":
        from data.transforms import StratifiedPixelSelection
        transform_train = StratifiedPixelSelection(
            num_masked_pixels=1, #61
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        transform_val = StratifiedPixelSelection(
            num_masked_pixels=1,
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
            num_masked_pixels=1017,            # nur (f,T)-Pixels – werden über Z dupliziert
            window_size=3,
            random_mask_low_rank=False,
            random_mask_noisy=False,
            swap_mode=SWAP_MODE,
        )
        transform_val_3d = StratifiedPixelSelection3D(
            num_masked_pixels=1017,
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
batch_size  = 600  #160 , 600
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
min_lr    = 2e-5          # learnign rate wir nie niedriger als das

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

##### YNET TEST

# --- Y-Net: separate Pfade für Noisy & LowRank -------------------------------
# Wenn du erst mal dieselben Ordner wie oben verwenden willst, kannst du sie kopieren:
# train_data_noisy = ["Tumor_1_normalized", "Tumor_1_normalized_only_tumor_voxels"]
# train_data_lr    = ["Tumor_1_normalized_tMPPCA_5D_On_All_Reps", "Tumor_1_normalized_only_tumor_voxels_tMPPCA_5D"]  # <-- anpassen!
# val_data_noisy   = ["Tumor_1_normalized"]
# val_data_lr      = ["Tumor_1_normalized_tMPPCA_5D_On_All_Reps"]  # <-- anpassen!

# Optional (falls vom Default 2 abweichend):
in_channels_noisy = 2
in_channels_lr    = 2

# train_data = ["P03_normalized_tMPPCA_5D", "P04_normalized_tMPPCA_5D", "P05_normalized_tMPPCA_5D", "P06_normalized_tMPPCA_5D", "P07_normalized_tMPPCA_5D"]
# val_data   = ["P08_normalized_tMPPCA_5D"]
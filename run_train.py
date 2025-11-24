# ───────────────────────── run_train.py ──────────────────────────
import os
import sys
import shutil
import importlib

# 1) Arbeitsverzeichnis auf Projekt-Root wechseln
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 2) Config laden
import config

os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.log_dir,        exist_ok=True)

# 3) Source-Snapshot anlegen
used_src_dir = os.path.join(config.run_dir, "used_source")
os.makedirs(used_src_dir, exist_ok=True)

items_to_snapshot = [
    "config.py",
    "run_train.py",
    "train",      # enthält alle Trainer
    "models",
    "losses",
    "data",
]
for item in items_to_snapshot:
    src = os.path.join(SCRIPT_DIR, item)
    dst = os.path.join(used_src_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif os.path.isfile(src):
        shutil.copy2(src, dst)

# 4) Schreibrechte (optional, z. B. auf HPC nötig)
for d in (config.run_dir, config.checkpoint_dir, config.log_dir, used_src_dir):
    try:
        os.chmod(d, 0o777)
    except PermissionError:
        pass

# 5) Trainer dynamisch importieren
#    -> config.TRAINER_MODULE und config.TRAIN_FUNC kommen entweder
#       aus _TRAINER_MAP in config.py ODER du überschreibst sie dort manuell.
try:
    module_name = config.TRAINER_MODULE
    func_name   = config.TRAIN_FUNC
except AttributeError as e:
    raise RuntimeError(
        "TRAINER_MODULE/ TRAIN_FUNC fehlen in config.py – "
        "prüfe _TRAINER_MAP und deine TRAIN_METHOD-/UNET_DIM-Einstellungen."
    ) from e

trainer_mod = importlib.import_module(module_name)
train_func  = getattr(trainer_mod, func_name)

print(f"[run_train] Starte {module_name}.{func_name} "
      f"(TRAIN_METHOD={config.TRAIN_METHOD}, UNET_DIM={config.UNET_DIM})")

# 6) Training starten
if __name__ == "__main__":
    train_func()

# ───────────────────────── run_train.py (NEU) ──────────────────────
import os, sys, shutil

# 1) Arbeitsverzeichnis auf Projekt-Root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 2) Config laden
import config

# 3) Source-Snapshot anlegen
used_src_dir = os.path.join(config.run_dir, "used_source")
os.makedirs(used_src_dir, exist_ok=True)

items_to_snapshot = [
    "config.py",
    "run_train.py",
    "train",          # enthält jetzt auch trainer_ynet.py
    "models",         # enthält ynet2d.py
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

# 4) Schreibrechte (optional)
for d in (config.run_dir, config.checkpoint_dir, config.log_dir, used_src_dir):
    try:
        os.chmod(d, 0o777)
    except PermissionError:
        pass  # falls nicht erlaubt (z. B. auf HPC-System)

# 5) Trainer auswählen
if   config.TRAIN_METHOD == "n2v":
    from train.trainer_n2v  import train_n2v        as train_func
elif config.TRAIN_METHOD == "p2n":
    from train.trainer_p2n  import train_p2n        as train_func
elif config.TRAIN_METHOD == "ynet":
    from train.trainer_ynet import train_ynet_n2v   as train_func
else:
    raise ValueError(f"Unknown TRAIN_METHOD {config.TRAIN_METHOD!r}. "
                 "Choose 'n2v', 'ynet' or 'p2n'.")


# 6) Start!
if __name__ == "__main__":
    train_func()



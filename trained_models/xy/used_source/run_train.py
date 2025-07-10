import os
import sys
import shutil

# ── 1) Stelle sicher, dass wir im Projektverzeichnis arbeiten ───────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── 2) Config laden (legt run_dir, checkpoint_dir & log_dir an) ────────────
import config

# ── 3) Quellcode-Snapshot anfertigen ────────────────────────────────────────
used_src_dir = os.path.join(config.run_dir, 'used_source')
os.makedirs(used_src_dir, exist_ok=True)

items_to_snapshot = [
    'config.py',
    'run_train.py',
    'train',
    'models',
    'losses',
    'data',
]
for item in items_to_snapshot:
    src = os.path.join(SCRIPT_DIR, item)
    dst = os.path.join(used_src_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    elif os.path.isfile(src):
        shutil.copy2(src, dst)

# ── 4) Volle Rechte setzen (777) ─────────────────────────────────────────────
for d in (
    config.run_dir,
    config.checkpoint_dir,
    config.log_dir,
    used_src_dir,
):
    os.chmod(d, 0o777)

# ── 5) Training starten ──────────────────────────────────────────────────────
if config.TRAIN_METHOD == "n2v":
    from train.trainer_n2v import train_n2v as train_func
elif config.TRAIN_METHOD == "p2n":
    from train.trainer_p2n import train_p2n as train_func
else:
    raise ValueError(f"Unknown TRAIN_METHOD {config.TRAIN_METHOD!r}, must be 'n2v' or 'p2n'")

if __name__ == "__main__":
    train_func()



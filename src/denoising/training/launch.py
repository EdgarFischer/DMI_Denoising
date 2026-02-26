# src/denoising/training/launch.py
import os
import sys
import shutil
import importlib
from pathlib import Path


def main():
    # 1) Switch working directory to repository root
    REPO_ROOT = Path(__file__).resolve().parents[3]  # .../repo
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # 2) Load config (keep as-is for now)
    import config

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 3) Create source snapshot
        # 3) Create source snapshot
    used_src_dir = os.path.join(config.run_dir, "used_source")
    os.makedirs(used_src_dir, exist_ok=True)

    items_to_snapshot = [
        "config.py",          
        "scripts",           
        "src/denoising",     
    ]

    for item in items_to_snapshot:
        src = os.path.join(REPO_ROOT, item)
        dst = os.path.join(used_src_dir, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        else:
            print(f"[snapshot] WARNING: not found: {src}")

    # 4) Permissions (optional, e.g. needed on HPC)
    for d in (config.run_dir, config.checkpoint_dir, config.log_dir, used_src_dir):
        try:
            os.chmod(d, 0o777)
        except PermissionError:
            pass

    # 5) Dynamically import trainer
    try:
        module_name = config.TRAINER_MODULE
        func_name = config.TRAIN_FUNC
    except AttributeError as e:
        raise RuntimeError(
            "TRAINER_MODULE / TRAIN_FUNC missing in config.py – "
            "check _TRAINER_MAP and your TRAIN_METHOD / UNET_DIM settings."
        ) from e

    trainer_mod = importlib.import_module(module_name)
    train_func = getattr(trainer_mod, func_name)

    print(
        f"[train] Starting {module_name}.{func_name} "
        f"(TRAIN_METHOD={config.TRAIN_METHOD}, UNET_DIM={config.UNET_DIM})"
    )

    # 6) Start training
    train_func()
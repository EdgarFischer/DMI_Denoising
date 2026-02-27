# src/denoising/training/launch.py
import os
import sys
import shutil
from pathlib import Path

def main(cfg, config_path: str | None = None):
    # 1) Switch working directory to repository root
    REPO_ROOT = Path(__file__).resolve().parents[3]
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # 2) Resolve run dirs
    run_dir = REPO_ROOT / cfg.run.base_dir / cfg.run.name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3) Create source snapshot
    used_src_dir = run_dir / "used_source"
    used_src_dir.mkdir(parents=True, exist_ok=True)

    items_to_snapshot = [
        "scripts",
        "src/denoising",
        "configs",
    ]

    for item in items_to_snapshot:
        src = REPO_ROOT / item
        dst = used_src_dir / item

        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            print(f"[snapshot] WARNING: not found: {src}")

    # Optional: copy exact config file
    if config_path is not None:
        cfg_src = Path(config_path)
        if cfg_src.exists():
            shutil.copy2(cfg_src, run_dir / cfg_src.name)

    # 4) Permissions (optional, e.g. needed on HPC)
    for d in (run_dir, checkpoint_dir, log_dir, used_src_dir):
        try:
            os.chmod(d, 0o777)
        except PermissionError:
            pass

    # ------------------------------------------------------------------
    # 5) Build N2V transforms based on cfg.mask.type
    # ------------------------------------------------------------------
    from denoising.data.transforms import (
        StratifiedPixelSelection2D,
        StratifiedPixelSelectionTime1D,
    )

    m = cfg.mask
    print(f"[mask] type={m.type} num_pixels={m.num_pixels} window_size={m.window_size}")

    if m.type == "time1d":
        transform_train = StratifiedPixelSelectionTime1D(
            num_masked_pixels=m.num_pixels,
            window_size=m.window_size,
        )
        transform_val = StratifiedPixelSelectionTime1D(
            num_masked_pixels=m.num_pixels,
            window_size=m.window_size,
        )
    elif m.type == "2d":
        transform_train = StratifiedPixelSelection2D(
            num_masked_pixels=m.num_pixels,
            window_size=m.window_size,
        )
        transform_val = StratifiedPixelSelection2D(
            num_masked_pixels=m.num_pixels,
            window_size=m.window_size,
        )
    else:
        raise ValueError(f"Unknown mask.type: {m.type} (expected 'time1d' or '2d')")

    # 6) Import fixed trainer (2D n2v only)
    from denoising.training.trainers.trainer_n2v import train as train_func

    print("[train] Starting denoising.training.trainers.trainer_n2v.train (2D n2v-only)")

    # 7) Start training
    train_func(
        cfg=cfg,
        run_dir=str(run_dir),
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        transform_train=transform_train,
        transform_val=transform_val,
    )
import subprocess
import sys
import tempfile
from pathlib import Path
import copy

import numpy as np
import yaml


def make_synthetic_dataset(out_dir: Path, shape=(5, 5, 5, 5, 10), noise_std=0.1, seed=0):
    rng = np.random.default_rng(seed)

    base = np.ones(shape, dtype=np.float32)
    noise_real = noise_std * rng.standard_normal(shape, dtype=np.float32)
    noise_imag = noise_std * rng.standard_normal(shape, dtype=np.float32)

    arr = (base + noise_real + 1j * noise_imag).astype(np.complex64)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data.npy", arr)


def build_sanity_config(original_cfg: dict, train_dir: Path, val_dir: Path, run_dir: Path) -> dict:
    cfg = copy.deepcopy(original_cfg)

    cfg["run"]["name"] = "sanity_check"
    cfg["run"]["base_dir"] = str(run_dir)
    cfg["run"]["seed"] = 0

    cfg["data"]["train"] = [str(train_dir)]
    cfg["data"]["val"] = [str(val_dir)]

    # keep training very small and fast
    cfg["data"]["num_samples"] = 16
    cfg["data"]["val_samples"] = 8

    cfg["optim"]["epochs"] = 2
    cfg["optim"]["batch_size"] = 2
    cfg["optim"]["num_workers"] = 0

    # small model for speed
    cfg["model"]["features"] = [4, 8]

    # disable patching for tiny synthetic data
    cfg["patching"]["enabled"] = False

    # disable augmentation for sanity check
    if "augmentation" in cfg:
        cfg["augmentation"]["enabled"] = False

    return cfg


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "train.yaml"
    train_script = repo_root / "scripts" / "train.py"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    with open(cfg_path, "r") as f:
        original_cfg = yaml.safe_load(f)

    original_cfg_text = cfg_path.read_text()

    with tempfile.TemporaryDirectory(prefix="sanity_check_") as tmp:
        tmp = Path(tmp)

        train_dir = tmp / "train_case"
        val_dir = tmp / "val_case"
        run_dir = tmp / "runs"

        print("Creating synthetic dataset...")
        make_synthetic_dataset(train_dir, seed=0)
        make_synthetic_dataset(val_dir, seed=1)

        print("Building temporary sanity config...")
        sanity_cfg = build_sanity_config(
            original_cfg=original_cfg,
            train_dir=train_dir,
            val_dir=val_dir,
            run_dir=run_dir,
        )

        try:
            print("Writing temporary config to configs/train.yaml ...")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(sanity_cfg, f, sort_keys=False)

            print("\nStarting training via scripts/train.py\n")

            result = subprocess.run(
                [sys.executable, str(train_script)],
                cwd=repo_root,
            )

            print()

            if result.returncode != 0:
                raise SystemExit(f"Sanity check failed with exit code {result.returncode}")

            print("Sanity check finished successfully.")

        finally:
            print("Restoring original configs/train.yaml ...")
            cfg_path.write_text(original_cfg_text)


if __name__ == "__main__":
    main()
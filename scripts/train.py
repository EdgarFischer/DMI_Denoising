#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import os
import shutil


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "train.yaml"),
        help="Path to training config YAML.",
    )
    return p.parse_args()


def load_yaml_minimal(path: str) -> dict:
    # Minimal loader to avoid importing project modules (and torch) before env is set
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()

    # ------------------------------------------------------------
    # Phase 1: read GPU from YAML WITHOUT importing your project
    # ------------------------------------------------------------
    raw0 = load_yaml_minimal(args.config)
    gpu = str(raw0["run"]["gpu"])

    # Make GPU indices match `nvidia-smi` order (recommended)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # Hard guarantee: restart once so that CUDA env is applied before any torch import
    if os.environ.get("DENOISING_GPU_ENV_READY") != "1":
        os.environ["DENOISING_GPU_ENV_READY"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # ------------------------------------------------------------
    # Phase 2: safe to import project now
    # ------------------------------------------------------------
    sys.path.insert(0, str(SRC))

    from denoising.config.load import load_yaml
    from denoising.config.build import build_config
    from denoising.training.launch import main as launch_main

    raw = load_yaml(args.config)
    cfg = build_config(raw)

    run_dir = ROOT / cfg.run.base_dir / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, run_dir / Path(args.config).name)

    print("[env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    launch_main(cfg, config_path=args.config)
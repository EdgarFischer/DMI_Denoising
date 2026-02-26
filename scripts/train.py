#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # ok als Übergang, später via pip install -e .

from denoising.config.load import load_yaml
from denoising.config.build import build_config
from denoising.training.launch import main as launch_main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "train.yaml"),
        help="Path to training config YAML.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    raw = load_yaml(args.config)
    cfg = build_config(raw)

    run_dir = Path(cfg.run.base_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Repro: exakt verwendete Config sichern
    shutil.copy(args.config, run_dir / Path(args.config).name)

    # Jetzt: bestehende Pipeline starten (minimal-invasive Änderung)
    launch_main(cfg)
#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def parse_args():
    p = argparse.ArgumentParser(description="Run denoising inference on a single FID .npy file.")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--input", type=str, required=True, help="Path to input FID .npy file")
    p.add_argument("--output", type=str, default="", help="Optional output .npy file path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu", type=str, default=None, help="GPU id (e.g. 0,1,2). Optional (overrides YAML).")
    return p.parse_args()


def load_yaml_minimal(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()

    # ---------------------------------------------------------
    # GPU selection (must happen before importing torch)
    # ---------------------------------------------------------
    raw0 = load_yaml_minimal(args.config)
    gpu = args.gpu if args.gpu is not None else str(raw0.get("run", {}).get("gpu", ""))

    if gpu and gpu.lower() != "none":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # ---------------------------------------------------------
    # Project imports
    # ---------------------------------------------------------
    sys.path.insert(0, str(SRC))

    from denoising.config.load import load_yaml
    from denoising.config.build import build_config
    from denoising.inference.api import infer

    cfg = build_config(load_yaml(args.config))

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_denoised.npy")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run inference
    y_fid, meta = infer(
        cfg=cfg,
        ckpt_path=args.ckpt,
        input_path=input_path,
        output_path=output_path,
        batch_size=args.batch_size,
    )

    print("[infer] done")
    print(f"[infer] input:  {input_path}")
    print(f"[infer] output: {output_path}")
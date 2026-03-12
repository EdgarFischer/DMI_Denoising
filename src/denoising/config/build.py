# src/denoising/config/build.py
from .schema import Config, RunCfg, DataCfg, MaskCfg, ModelCfg, OptimCfg

def build_config(raw: dict) -> Config:
    # --- run ---
    run = RunCfg(**raw["run"])

    # --- data ---
    data_raw = raw["data"]
    data = DataCfg(
        train=list(data_raw["train"]),
        val=list(data_raw["val"]),
        image_axes=tuple(data_raw["image_axes"]),
        channel_axis=(
            None if data_raw.get("channel_axis", None) is None
            else int(data_raw["channel_axis"])
        ),
        fourier_axes=tuple(data_raw["fourier_axes"]),
        num_samples=int(data_raw["num_samples"]),
        val_samples=int(data_raw["val_samples"]),
        normalization=bool(data_raw.get("normalization", True)),
    )

    # --- masking ---
    mask_raw = raw["masking"]
    mask = MaskCfg(
        masked_axes=tuple(mask_raw["masked_axes"]),
        num_pixels=int(mask_raw["num_pixels"]),
        window_size=int(mask_raw["window_size"]),
    )

    # --- model ---
    model_raw = raw["model"]
    model = ModelCfg(
        features=tuple(model_raw["features"]),
    )

    # --- optim ---
    optim_raw = raw["optim"]
    optim = OptimCfg(
        lr=float(optim_raw["lr"]),
        factor=float(optim_raw["factor"]),
        step_size=int(optim_raw["step_size"]),
        min_lr=float(optim_raw["min_lr"]),
        epochs=int(optim_raw["epochs"]),
        batch_size=int(optim_raw["batch_size"]),
        num_workers=int(optim_raw["num_workers"]),
    )

    return Config(
        run=run,
        data=data,
        mask=mask,
        model=model,
        optim=optim,
    )
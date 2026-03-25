# src/denoising/config/build.py
from .schema import Config, RunCfg, DataCfg, PatchingCfg, MaskCfg, ModelCfg, OptimCfg

def validate_config(cfg: Config) -> None:
    # --- patching ---
    if cfg.patching.enabled:
        num_axes = len(cfg.data.image_axes) + (
            1 if cfg.data.channel_axis is not None else 0
        )

        if len(cfg.patching.patch_sizes) != num_axes:
            raise ValueError(
                f"patch_sizes must have length {num_axes} "
                f"(image_axes + optional channel_axis), "
                f"but got {len(cfg.patching.patch_sizes)}."
            )

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

    # --- patching ---
    patch_raw = raw.get("patching", {})
    patching = PatchingCfg(
        enabled=bool(patch_raw.get("enabled", False)),
        patch_sizes=tuple(
            None if p is None else int(p)
            for p in patch_raw.get("patch_sizes", [])
        ),
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

    cfg = Config(
        run=run,
        data=data,
        patching=patching,
        mask=mask,
        model=model,
        optim=optim,
    )

    validate_config(cfg)
    return cfg

# src/denoising/config/build.py
from .schema import (
    Config,
    RunCfg,
    ViewSamplingCfg,
    DataCfg,
    GlobalPhaseAugCfg,
    PermutationAugCfg,
    InversionAugCfg,
    GlobalScaleAugCfg,
    AugmentationCfg,
    PatchingCfg,
    MaskCfg,
    ModelCfg,
    OptimCfg,
    InferenceCfg,
)

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
    # --- inference ---
    if cfg.inference is not None:
        num_axes = len(cfg.data.image_axes) + (
            1 if cfg.data.channel_axis is not None else 0
        )

        if len(cfg.inference.patch_strides) != num_axes:
            raise ValueError(
                f"patch_strides must have length {num_axes} "
                f"(image_axes + optional channel_axis), "
                f"but got {len(cfg.inference.patch_strides)}."
            )
    
    # --- masking ---
    if not (0.0 < cfg.mask.mask_fraction <= 1.0):
        raise ValueError("mask.mask_fraction must be in (0, 1].")

    # --- augmentation ---
    if cfg.augmentation is not None:
        if not (0.0 <= cfg.augmentation.global_phase.p <= 1.0):
            raise ValueError("augmentation.global_phase.p must be in [0, 1].")

        if not (0.0 <= cfg.augmentation.permutation.p <= 1.0):
            raise ValueError("augmentation.permutation.p must be in [0, 1].")

        if not (0.0 <= cfg.augmentation.inversion.p <= 1.0):
            raise ValueError("augmentation.inversion.p must be in [0, 1].")

        if not (0.0 <= cfg.augmentation.global_scale.p <= 1.0):
            raise ValueError("augmentation.global_scale.p must be in [0, 1].")

        if cfg.augmentation.global_scale.min > cfg.augmentation.global_scale.max:
            raise ValueError(
                "augmentation.global_scale.min must be <= augmentation.global_scale.max."
            )

    # --- view sampling ---
    if cfg.data.view_sampling is not None and cfg.data.view_sampling.enabled:
        if len(cfg.data.view_sampling.views) == 0:
            raise ValueError("data.view_sampling.views must not be empty when view_sampling is enabled.")

        for view in cfg.data.view_sampling.views:
            if len(view) != len(cfg.data.image_axes):
                raise ValueError(
                    f"Each view in data.view_sampling.views must have length {len(cfg.data.image_axes)}, "
                    f"but got view {view}."
                )

            if len(set(view)) != len(view):
                raise ValueError(f"View {view} contains duplicate axes.")

            if cfg.data.channel_axis is not None and cfg.data.channel_axis in view:
                raise ValueError(
                    f"View {view} must not contain channel_axis {cfg.data.channel_axis}."
                )
            
def build_config(raw: dict) -> Config:
    # --- run ---
    run = RunCfg(**raw["run"])

    # --- data ---
    data_raw = raw["data"]
    vs_raw = data_raw.get("view_sampling", None)
    view_sampling = None
    if vs_raw is not None:
        view_sampling = ViewSamplingCfg(
            enabled=bool(vs_raw.get("enabled", False)),
            views=tuple(
                tuple(int(ax) for ax in view)
                for view in vs_raw.get("views", [])
            ),
        )
    data = DataCfg(
        base_dir=str(data_raw.get("base_dir", "")),
        data_filename=str(data_raw.get("data_filename", "data.npy")),
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
        view_sampling=view_sampling,
    )

    # --- augmentation ---
    aug_raw = raw.get("augmentation", None)
    augmentation = None
    if aug_raw is not None:
        gp_raw = aug_raw.get("global_phase", {})
        perm_raw = aug_raw.get("permutation", {})
        inv_raw = aug_raw.get("inversion", {})
        scale_raw = aug_raw.get("global_scale", {})

        augmentation = AugmentationCfg(
            enabled=bool(aug_raw.get("enabled", True)),
            global_phase=GlobalPhaseAugCfg(
                enabled=bool(gp_raw.get("enabled", False)),
                p=float(gp_raw.get("p", 1.0)),
            ),
            permutation=PermutationAugCfg(
                enabled=bool(perm_raw.get("enabled", False)),
                p=float(perm_raw.get("p", 0.0)),
                axes=tuple(int(ax) for ax in perm_raw.get("axes", [])),
            ),
            inversion=InversionAugCfg(
                enabled=bool(inv_raw.get("enabled", False)),
                p=float(inv_raw.get("p", 0.0)),
                axes=tuple(int(ax) for ax in inv_raw.get("axes", [])),
            ),
            global_scale=GlobalScaleAugCfg(
                enabled=bool(scale_raw.get("enabled", False)),
                p=float(scale_raw.get("p", 0.0)),
                min=float(scale_raw.get("min", 1.0)),
                max=float(scale_raw.get("max", 1.0)),
            ),
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
    mask_raw = raw.get("masking", {})

    mask = MaskCfg(
        masked_axes=tuple(mask_raw.get("masked_axes", [])),
        mask_fraction=float(mask_raw.get("mask_fraction", 0.1)),
        window_size=int(mask_raw.get("window_size", 1)),
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

    # --- inference ---
    inf_raw = raw.get("inference", None)
    inference = None
    if inf_raw is not None:
        inference = InferenceCfg(
            patch_strides=tuple(
                None if p is None else int(p)
                for p in inf_raw.get("patch_strides", [])
            ),
            weight_mode=str(inf_raw.get("weight_mode", "hann")),
        )

    cfg = Config(
        run=run,
        data=data,
        augmentation=augmentation,
        patching=patching,
        mask=mask,
        model=model,
        optim=optim,
        inference=inference,
        resume_training=bool(raw.get("resume_training", False)),
        resume_ckpt=str(raw.get("resume_ckpt", "")),
    )

    validate_config(cfg)
    return cfg

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
    PhysicsCfg,
    OptimCfg,
    InferenceCfg,
    PhysicsDataLossCfg,
)

def validate_config(cfg: Config) -> None:
    if cfg.model.architecture == "physics_conv3d":
        if cfg.model.physics is None:
            raise ValueError(
                "model.physics is required for architecture='physics_conv3d'."
            )
        if len(cfg.data.image_axes) != 3:
            raise ValueError("physics_conv3d requires exactly three image_axes.")
        if cfg.data.channel_axis is not None:
            raise ValueError("physics_conv3d currently requires channel_axis: null.")
        if cfg.data.spectral_axis not in (0, 1, 2):
            raise ValueError(
                "data.spectral_axis must be 0, 1, or 2 for physics_conv3d."
            )
        if len(cfg.model.features) != len(cfg.model.physics.spectral_strides):
            raise ValueError(
                "model.features and model.physics.spectral_strides must have "
                "the same length."
            )
        if any(value < 1 for value in cfg.model.physics.spectral_strides):
            raise ValueError("All spectral_strides must be >= 1.")
        if cfg.model.physics.spectral_kernel_size % 2 != 1:
            raise ValueError("spectral_kernel_size must be odd.")
        if cfg.model.physics.spatial_kernel_size % 2 != 1:
            raise ValueError("spatial_kernel_size must be odd.")
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
    if not cfg.phive_mode and not (0.0 < cfg.mask.mask_fraction <= 1.0):
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
        spectral_axis=(
            None if data_raw.get("spectral_axis") is None
            else int(data_raw["spectral_axis"])
        ),
        spatial_mask_filename=(
            None if data_raw.get("spatial_mask_filename") is None
            else str(data_raw["spatial_mask_filename"])
        ),
        target_dirname=(
            None if data_raw.get("target_dirname") is None
            else str(data_raw["target_dirname"])
        ),
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
    physics_raw = model_raw.get("physics")
    physics = None
    if physics_raw is not None:
        physics = PhysicsCfg(
            simulation_config=str(physics_raw["simulation_config"]),
            basis_dataset=str(physics_raw.get("basis_dataset", "clean_fid")),
            active_metabolites_only=bool(
                physics_raw.get("active_metabolites_only", True)
            ),
            basis_components=(
                None if physics_raw.get("basis_components") is None
                else tuple(str(name) for name in physics_raw["basis_components"])
            ),
            parameter_statistics_path=(
                None if physics_raw.get("parameter_statistics_path") is None
                else str(physics_raw["parameter_statistics_path"])
            ),
            denoising_ppm_range=(
                None if physics_raw.get("denoising_ppm_range") is None
                else tuple(
                    float(value)
                    for value in physics_raw["denoising_ppm_range"]
                )
            ),
            ppm_reference=float(physics_raw.get("ppm_reference", 4.65)),
            hz_per_ppm=(
                None if physics_raw.get("hz_per_ppm") is None
                else float(physics_raw["hz_per_ppm"])
            ),
            spectral_strides=tuple(
                int(value) for value in physics_raw.get(
                    "spectral_strides", (2, 2, 2, 2, 2, 2)
                )
            ),
            spectral_kernel_size=int(
                physics_raw.get("spectral_kernel_size", 5)
            ),
            spatial_kernel_size=int(physics_raw.get("spatial_kernel_size", 3)),
            parameter_head_hidden_channels=int(
                physics_raw.get("parameter_head_hidden_channels", 256)
            ),
            initial_reconstruction_rms=float(
                physics_raw.get("initial_reconstruction_rms", 0.025)
            ),
            initial_lorentzian_fwhm_hz=float(
                physics_raw.get("initial_lorentzian_fwhm_hz", 5.0)
            ),
            initial_gaussian_fwhm_hz=float(
                physics_raw.get("initial_gaussian_fwhm_hz", 3.0)
            ),
            parameter_head_weight_std=float(
                physics_raw.get("parameter_head_weight_std", 1e-3)
            ),
            lineshape_model=str(
                physics_raw.get("lineshape_model", "global_voigt")
            ),
            lineshape_kernel_size=int(
                physics_raw.get("lineshape_kernel_size", 23)
            ),
            metabolite_shift_mean_hz=float(
                physics_raw.get("metabolite_shift_mean_hz", 0.0)
            ),
            metabolite_shift_std_hz=float(
                physics_raw.get("metabolite_shift_std_hz", 1.0)
            ),
            metabolite_fwhm_mean_hz=float(
                physics_raw.get("metabolite_fwhm_mean_hz", 5.0)
            ),
            metabolite_fwhm_std_hz=float(
                physics_raw.get("metabolite_fwhm_std_hz", 2.5)
            ),
            baseline_n_splines=int(
                physics_raw.get("baseline_n_splines", 0)
            ),
            baseline_ppm_range=(
                None if physics_raw.get("baseline_ppm_range") is None
                else tuple(float(value) for value in physics_raw["baseline_ppm_range"])
            ),
            baseline_conjugate_subject_signals=bool(
                physics_raw.get("baseline_conjugate_subject_signals", False)
            ),
            baseline_ford_to_model_scale=float(
                physics_raw.get("baseline_ford_to_model_scale", 1.0)
            ),
            baseline_coefficient_statistics_path=(
                None
                if physics_raw.get("baseline_coefficient_statistics_path") is None
                else str(physics_raw["baseline_coefficient_statistics_path"])
            ),
        )
    model = ModelCfg(
        features=tuple(model_raw["features"]),
        architecture=str(model_raw.get("architecture", "auto_unet")),
        physics=physics,
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

    regularization_raw = raw.get("parameter_regularization")
    parameter_regularization = None
    if regularization_raw is not None:
        from denoising.config.schema import ParameterRegularizationCfg

        parameter_regularization = ParameterRegularizationCfg(
            enabled=bool(regularization_raw.get("enabled", False)),
            shift_weight=float(regularization_raw.get("shift_weight", 0.0)),
            shift_mean_hz=float(regularization_raw.get("shift_mean_hz", 0.0)),
            shift_std_hz=float(regularization_raw.get("shift_std_hz", 1.0)),
            fwhm_weight=float(regularization_raw.get("fwhm_weight", 0.0)),
            fwhm_mean_hz=float(regularization_raw.get("fwhm_mean_hz", 5.0)),
            fwhm_std_hz=float(regularization_raw.get("fwhm_std_hz", 2.5)),
            kernel_curvature_weight=float(
                regularization_raw.get("kernel_curvature_weight", 0.0)
            ),
            baseline_curvature_weight=float(
                regularization_raw.get("baseline_curvature_weight", 0.0)
            ),
            voigt_nuisance_weight=float(
                regularization_raw.get("voigt_nuisance_weight", 0.0)
            ),
        )
        if parameter_regularization.shift_std_hz <= 0:
            raise ValueError("parameter_regularization.shift_std_hz must be > 0")
        if parameter_regularization.fwhm_std_hz <= 0:
            raise ValueError("parameter_regularization.fwhm_std_hz must be > 0")
        if (
            parameter_regularization.shift_weight < 0
            or parameter_regularization.fwhm_weight < 0
            or parameter_regularization.kernel_curvature_weight < 0
            or parameter_regularization.baseline_curvature_weight < 0
            or parameter_regularization.voigt_nuisance_weight < 0
        ):
            raise ValueError("parameter regularization weights must be >= 0")

    physics_data_loss_raw = raw.get("physics_data_loss")
    physics_data_loss = None
    if physics_data_loss_raw is not None:
        physics_data_loss = PhysicsDataLossCfg(
            residual_variance_scaling=bool(
                physics_data_loss_raw.get("residual_variance_scaling", False)
            ),
            residual_std_epsilon=float(
                physics_data_loss_raw.get("residual_std_epsilon", 1e-8)
            ),
            residual_variance_warmup_epochs=int(
                physics_data_loss_raw.get("residual_variance_warmup_epochs", 0)
            ),
        )
        if physics_data_loss.residual_std_epsilon <= 0:
            raise ValueError("physics_data_loss.residual_std_epsilon must be > 0")
        if physics_data_loss.residual_variance_warmup_epochs < 0:
            raise ValueError(
                "physics_data_loss.residual_variance_warmup_epochs must be >= 0"
            )
        if (
            physics_data_loss.residual_variance_scaling
            and model.architecture != "physics_conv3d"
        ):
            raise ValueError(
                "Residual-variance scaling is only supported by physics_conv3d."
            )

    cfg = Config(
        run=run,
        data=data,
        augmentation=augmentation,
        patching=patching,
        mask=mask,
        model=model,
        optim=optim,
        parameter_regularization=parameter_regularization,
        physics_data_loss=physics_data_loss,
        inference=inference,
        resume_training=bool(raw.get("resume_training", False)),
        resume_ckpt=str(raw.get("resume_ckpt", "")),
        training_mode=str(raw.get("training_mode", "n2v")),
        phive_mode=bool(raw.get("phive_mode", False)),
    )

    validate_config(cfg)
    return cfg

from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass(frozen=True)
class RunCfg:
    name: str
    base_dir: str
    gpu: str
    seed: int

@dataclass(frozen=True)
class ViewSamplingCfg:
    enabled: bool
    views: Tuple[Tuple[int, ...], ...]

@dataclass(frozen=True)
class DataCfg:
    base_dir: str
    data_filename: str
    train: List[str]
    val: List[str]
    image_axes: Tuple[int, ...]
    channel_axis: Optional[int]
    fourier_axes: Tuple[int, ...]
    num_samples: int
    val_samples: int
    normalization: bool
    view_sampling: Optional[ViewSamplingCfg] = None
    spectral_axis: Optional[int] = None
    spatial_mask_filename: Optional[str] = None
    target_dirname: Optional[str] = None

@dataclass(frozen=True)
class GlobalPhaseAugCfg:
    enabled: bool
    p: float


@dataclass(frozen=True)
class PermutationAugCfg:
    enabled: bool
    p: float
    axes: Tuple[int, ...]


@dataclass(frozen=True)
class InversionAugCfg:
    enabled: bool
    p: float
    axes: Tuple[int, ...]


@dataclass(frozen=True)
class GlobalScaleAugCfg:
    enabled: bool
    p: float
    min: float
    max: float


@dataclass(frozen=True)
class AugmentationCfg:
    enabled: bool
    global_phase: GlobalPhaseAugCfg
    permutation: PermutationAugCfg
    inversion: InversionAugCfg
    global_scale: GlobalScaleAugCfg

@dataclass(frozen=True)
class PatchingCfg:
    enabled: bool
    patch_sizes: Tuple[Optional[int], ...]

@dataclass(frozen=True)
class MaskCfg:
    masked_axes: Tuple[int, ...]   # global axes that will be masked, can be 1D oder 2D
    mask_fraction: float
    window_size: int


@dataclass(frozen=True)
class PhysicsCfg:
    simulation_config: str
    basis_dataset: str = "clean_fid"
    active_metabolites_only: bool = True
    basis_components: Optional[Tuple[str, ...]] = None
    parameter_statistics_path: Optional[str] = None
    denoising_ppm_range: Optional[Tuple[float, float]] = None
    ppm_reference: float = 4.65
    hz_per_ppm: Optional[float] = None
    spectral_strides: Tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    spectral_kernel_size: int = 5
    spatial_kernel_size: int = 3
    parameter_head_hidden_channels: int = 256
    initial_reconstruction_rms: float = 0.025
    initial_lorentzian_fwhm_hz: float = 5.0
    initial_gaussian_fwhm_hz: float = 3.0
    parameter_head_weight_std: float = 1e-3
    lineshape_model: str = "global_voigt"
    lineshape_kernel_size: int = 23
    metabolite_shift_mean_hz: float = 0.0
    metabolite_shift_std_hz: float = 1.0
    metabolite_fwhm_mean_hz: float = 5.0
    metabolite_fwhm_std_hz: float = 2.5
    baseline_n_splines: int = 0
    baseline_ppm_range: Optional[Tuple[float, float]] = None
    baseline_conjugate_subject_signals: bool = False
    baseline_ford_to_model_scale: float = 1.0
    baseline_coefficient_statistics_path: Optional[str] = None


@dataclass(frozen=True)
class ModelCfg:
    features: Tuple[int, ...]
    architecture: str = "auto_unet"
    physics: Optional[PhysicsCfg] = None


@dataclass(frozen=True)
class OptimCfg:
    lr: float
    factor: float
    step_size: int
    min_lr: float
    epochs: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class ParameterRegularizationCfg:
    enabled: bool = False
    shift_weight: float = 0.0
    shift_mean_hz: float = 0.0
    shift_std_hz: float = 1.0
    fwhm_weight: float = 0.0
    fwhm_mean_hz: float = 5.0
    fwhm_std_hz: float = 2.5
    kernel_curvature_weight: float = 0.0
    baseline_curvature_weight: float = 0.0
    voigt_nuisance_weight: float = 0.0


@dataclass(frozen=True)
class PhysicsDataLossCfg:
    residual_variance_scaling: bool = False
    residual_std_epsilon: float = 1e-8
    residual_variance_warmup_epochs: int = 0


@dataclass(frozen=True)
class InferenceCfg:
    patch_strides: Tuple[Optional[int], ...]
    weight_mode: str = "hann"

@dataclass(frozen=True)
class Config:
    run: RunCfg
    data: DataCfg
    augmentation: Optional[AugmentationCfg]
    patching: PatchingCfg
    mask: MaskCfg
    model: ModelCfg
    optim: OptimCfg
    parameter_regularization: Optional[ParameterRegularizationCfg] = None
    physics_data_loss: Optional[PhysicsDataLossCfg] = None
    inference: Optional[InferenceCfg] = None
    resume_training: bool = False
    resume_ckpt: str = ""
    training_mode: str = "n2v"
    phive_mode: bool = False

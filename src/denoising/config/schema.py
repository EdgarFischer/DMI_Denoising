from dataclasses import dataclass
from typing import Tuple, List


@dataclass(frozen=True)
class RunCfg:
    name: str
    base_dir: str
    gpu: str
    seed: int


@dataclass(frozen=True)
class DataCfg:
    train: List[str]
    val: List[str]
    image_axes: Tuple[int, int]
    fourier_axes: Tuple[int, ...]
    num_samples: int
    val_samples: int
    normalization: bool


@dataclass(frozen=True)
class MaskCfg:
    type: str              # "time1d" | "2d"
    num_pixels: int
    window_size: int


@dataclass(frozen=True)
class ModelCfg:
    features: Tuple[int, ...]


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
class Config:
    run: RunCfg
    data: DataCfg
    mask: MaskCfg
    model: ModelCfg
    optim: OptimCfg
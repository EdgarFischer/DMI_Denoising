"""Physics-informed, baseline-free spectral models."""

from .basis_decoder import BaselineFreeBasisDecoder
from .parameters import SpectralParameters
from .parameterization import (
    LCModelKernelParameterization,
    StandardizedLCModelKernelParameterization,
    StandardizedVoigtBaselineParameterization,
    MinimalPhysicalParameterization,
)
from .physics_conv3d import PhysicsConv3D, PhysicsModelOutput

__all__ = [
    "BaselineFreeBasisDecoder",
    "MinimalPhysicalParameterization",
    "LCModelKernelParameterization",
    "StandardizedLCModelKernelParameterization",
    "StandardizedVoigtBaselineParameterization",
    "PhysicsConv3D",
    "PhysicsModelOutput",
    "SpectralParameters",
]

# data/get_transform.py
from transforms_3d import StratifiedPixelSelection3D, StratifiedFreqSelection3D

def build_transform(cfg):
    if cfg["mask_type"] == "2d":
        return StratifiedPixelSelection3D(
            num_masked_pixels = cfg["n_mask"],
            window_size       = cfg["window"],
        )
    elif cfg["mask_type"] == "1d":
        return StratifiedFreqSelection3D(
            num_masked_freq   = cfg["n_mask"],
        )
    else:
        raise ValueError("mask_type muss '1d' oder '2d' sein")

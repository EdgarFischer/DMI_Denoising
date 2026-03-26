import numpy as np
from types import SimpleNamespace

from denoising.config.build import build_config
from denoising.data.mrsi_nd_dataset import MRSiNDataset


def make_complex_data(shape=(2, 3, 4, 5, 6)):
    real = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    imag = real + 1.0
    return (real + 1j * imag).astype(np.complex64)


def test_dataset_applies_augmentation_from_config_global_scale():
    raw_cfg = {
        "run": {"name": "t", "base_dir": "x", "gpu": "0", "seed": 0},
        "data": {
            "train": ["dummy"],
            "val": ["dummy"],
            "image_axes": [3, 4],
            "channel_axis": None,
            "fourier_axes": [],
            "num_samples": 1,
            "val_samples": 1,
            "normalization": True,
        },
        "augmentation": {
            "enabled": True,
            "global_phase": {"enabled": False, "p": 0.0},
            "permutation": {"enabled": False, "p": 0.0, "axes": []},
            "inversion": {"enabled": False, "p": 0.0, "axes": []},
            "global_scale": {"enabled": True, "p": 1.0, "min": 2.0, "max": 2.0},
        },
        "patching": {"enabled": False, "patch_sizes": []},
        "masking": {"masked_axes": [4], "num_pixels": 1, "window_size": 3},
        "model": {"features": [8, 16]},
        "optim": {
            "lr": 1e-3,
            "factor": 0.5,
            "step_size": 10,
            "min_lr": 1e-5,
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
        },
        "inference": {"patch_strides": [None, None], "weight_mode": "average"},
    }

    cfg = build_config(raw_cfg)

    data = make_complex_data()

    fixed_indices = {0: 0, 1: 0, 2: 0}  # keep only axes 3,4 visible

    ds_aug = MRSiNDataset(
        data=data,
        image_axes=tuple(cfg.data.image_axes),
        channel_axis=cfg.data.channel_axis,
        masked_axes=tuple(cfg.mask.masked_axes),
        fixed_indices=fixed_indices,
        transform=None,
        num_samples=1,
        augmentation=cfg.augmentation,
        patching_enabled=False,
        patch_sizes=(),
    )

    ds_noaug = MRSiNDataset(
        data=data,
        image_axes=tuple(cfg.data.image_axes),
        channel_axis=cfg.data.channel_axis,
        masked_axes=tuple(cfg.mask.masked_axes),
        fixed_indices=fixed_indices,
        transform=None,
        num_samples=1,
        augmentation=None,
        patching_enabled=False,
        patch_sizes=(),
    )

    inp_aug, tgt_aug, mask_aug = ds_aug[0]
    inp_ref, tgt_ref, mask_ref = ds_noaug[0]

    np.testing.assert_allclose(inp_aug.numpy(), 2.0 * inp_ref.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tgt_aug.numpy(), 2.0 * tgt_ref.numpy(), rtol=1e-6, atol=1e-6)

    # mask should stay unchanged because transform=None creates ones_like(img)
    np.testing.assert_array_equal(mask_aug.numpy(), mask_ref.numpy())


def test_dataset_does_not_apply_augmentation_if_top_level_disabled():
    raw_cfg = {
        "run": {"name": "t", "base_dir": "x", "gpu": "0", "seed": 0},
        "data": {
            "train": ["dummy"],
            "val": ["dummy"],
            "image_axes": [3, 4],
            "channel_axis": None,
            "fourier_axes": [],
            "num_samples": 1,
            "val_samples": 1,
            "normalization": True,
        },
        "augmentation": {
            "enabled": False,
            "global_phase": {"enabled": False, "p": 0.0},
            "permutation": {"enabled": False, "p": 0.0, "axes": []},
            "inversion": {"enabled": False, "p": 0.0, "axes": []},
            "global_scale": {"enabled": True, "p": 1.0, "min": 2.0, "max": 2.0},
        },
        "patching": {"enabled": False, "patch_sizes": []},
        "masking": {"masked_axes": [4], "num_pixels": 1, "window_size": 3},
        "model": {"features": [8, 16]},
        "optim": {
            "lr": 1e-3,
            "factor": 0.5,
            "step_size": 10,
            "min_lr": 1e-5,
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
        },
        "inference": {"patch_strides": [None, None], "weight_mode": "average"},
    }

    cfg = build_config(raw_cfg)

    data = make_complex_data()
    fixed_indices = {0: 0, 1: 0, 2: 0}

    ds_disabled = MRSiNDataset(
        data=data,
        image_axes=tuple(cfg.data.image_axes),
        channel_axis=cfg.data.channel_axis,
        masked_axes=tuple(cfg.mask.masked_axes),
        fixed_indices=fixed_indices,
        transform=None,
        num_samples=1,
        augmentation=cfg.augmentation,
        patching_enabled=False,
        patch_sizes=(),
    )

    ds_ref = MRSiNDataset(
        data=data,
        image_axes=tuple(cfg.data.image_axes),
        channel_axis=cfg.data.channel_axis,
        masked_axes=tuple(cfg.mask.masked_axes),
        fixed_indices=fixed_indices,
        transform=None,
        num_samples=1,
        augmentation=None,
        patching_enabled=False,
        patch_sizes=(),
    )

    inp_dis, tgt_dis, mask_dis = ds_disabled[0]
    inp_ref, tgt_ref, mask_ref = ds_ref[0]

    np.testing.assert_array_equal(inp_dis.numpy(), inp_ref.numpy())
    np.testing.assert_array_equal(tgt_dis.numpy(), tgt_ref.numpy())
    np.testing.assert_array_equal(mask_dis.numpy(), mask_ref.numpy())
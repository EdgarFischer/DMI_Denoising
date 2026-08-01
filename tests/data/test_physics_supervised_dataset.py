import json
import math

import numpy as np
import torch

from denoising.data.physics_supervised_dataset import PhysicsSupervisedDataset
from denoising.training.trainers.trainer_physics_supervised import (
    _masked_parameter_mse,
)


def test_supervised_dataset_loads_and_converts_ford_targets(tmp_path):
    subject = tmp_path / "subject"
    (subject / "OriginalData").mkdir(parents=True)
    (subject / "masks").mkdir()
    target = subject / "fit"
    (target / "metabolite_maps").mkdir(parents=True)
    (target / "nuisance_parameter_maps").mkdir()

    shape = (6, 7, 2)
    fid = np.ones((*shape, 8), dtype=np.complex64) * 2
    np.save(subject / "OriginalData" / "data.npy", fid)
    np.save(subject / "masks" / "brain.npy", np.ones(shape, dtype=bool))
    amplitudes = np.stack(
        (np.full(shape, 2.0), np.full(shape, 3.0)), axis=-1
    ).astype(np.float32)
    np.save(target / "metabolite_maps" / "metabolite_maps.npy", amplitudes)
    (target / "parameter_maps_metadata.json").write_text(
        json.dumps({"metabolite_axis_order": ["A", "B"]})
    )
    nuisance = target / "nuisance_parameter_maps"
    for name, value in {
        "delta_f": 4.0,
        "delta_phi_0": 0.5,
        "delta_phi_1": 2.0,
        "lorentzian_damping": math.pi * 6.0,
        "gaussian_damping": (math.pi * 7.0) ** 2 / (4 * math.log(2)),
    }.items():
        np.save(nuisance / f"{name}.npy", np.full(shape, value, np.float32))

    dataset = PhysicsSupervisedDataset(
        base_dir=tmp_path,
        subjects=["subject"],
        data_filename="OriginalData/data.npy",
        mask_filename="masks/brain.npy",
        target_dirname="fit",
        metabolite_names=("B", "A"),
        num_samples=1,
        patch_size=(4, 4),
        hz_per_ppm=100.0,
        ppm_reference=4.0,
    )
    network_input, parameters, mask = dataset[0]
    assert network_input.shape == (2, 4, 4, 8)
    assert parameters.shape == (7, 4, 4)
    assert mask.shape == (4, 4)
    torch.testing.assert_close(parameters[:, 0, 0], torch.tensor([
        3.0, 2.0, -4.0, 6.0, 7.0, 8.5, 0.02
    ]))


def test_plain_parameter_mse_uses_only_valid_spatial_positions():
    prediction = torch.tensor([[[[2.0, 100.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[1.0, 0.0]]])
    torch.testing.assert_close(
        _masked_parameter_mse(prediction, target, mask), torch.tensor(4.0)
    )


def test_dataset_rejects_cropped_ford_maps(tmp_path):
    subject = tmp_path / "subject"
    (subject / "OriginalData").mkdir(parents=True)
    (subject / "masks").mkdir()
    target = subject / "fit"
    (target / "metabolite_maps").mkdir(parents=True)
    nuisance = target / "nuisance_parameter_maps"
    nuisance.mkdir()

    fid = np.ones((6, 7, 5, 8), dtype=np.complex64)
    mask = np.zeros(fid.shape[:3], dtype=bool)
    mask[1:5, 2:6, 1:4] = True
    np.save(subject / "OriginalData" / "data.npy", fid)
    np.save(subject / "masks" / "brain.npy", mask)
    np.save(target / "metabolite_maps" / "A.npy", np.ones((4, 4, 3)))
    for name in (
        "delta_f",
        "delta_phi_0",
        "delta_phi_1",
        "lorentzian_damping",
        "gaussian_damping",
    ):
        np.save(nuisance / f"{name}.npy", np.zeros((4, 4, 3)))

    import pytest

    with pytest.raises(ValueError, match="does not match mask"):
        PhysicsSupervisedDataset(
            base_dir=tmp_path,
            subjects=["subject"],
            data_filename="OriginalData/data.npy",
            mask_filename="masks/brain.npy",
            target_dirname="fit",
            metabolite_names=("A",),
            num_samples=1,
            patch_size=(4, 4),
        )

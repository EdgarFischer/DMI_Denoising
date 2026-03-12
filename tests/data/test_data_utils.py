import numpy as np
from pathlib import Path

from denoising.data.data_utils import load_and_preprocess_data


def test_load_4d_to_6d(tmp_path):

    base = tmp_path / "datasets"
    folder = base / "testset"
    folder.mkdir(parents=True)

    # Fake 4D data
    arr = np.random.randn(3,4,5,6).astype(np.float32)

    np.save(folder / "data.npy", arr)

    out = load_and_preprocess_data(
        folder_names=["testset"],
        base_path=str(base),
        fourier_axes=None,
        normalization=False
    )

    assert out.shape == (3,4,5,6,1,1)

def test_load_5d_to_6d(tmp_path):

    base = tmp_path / "datasets"
    folder = base / "testset"
    folder.mkdir(parents=True)

    arr = np.random.randn(3,4,5,6,7).astype(np.float32)

    np.save(folder / "data.npy", arr)

    out = load_and_preprocess_data(
        folder_names=["testset"],
        base_path=str(base),
        fourier_axes=None,
        normalization=False
    )

    assert out.shape == (3,4,5,6,7,1)

def test_multiple_folders_concat(tmp_path):

    base = tmp_path / "datasets"

    for i in range(2):
        folder = base / f"set{i}"
        folder.mkdir(parents=True)

        arr = np.random.randn(2,3,4,5).astype(np.float32)
        np.save(folder / "data.npy", arr)

    out = load_and_preprocess_data(
        folder_names=["set0","set1"],
        base_path=str(base),
        fourier_axes=None,
        normalization=False
    )

    # letzte Dimension ist D
    assert out.shape == (2,3,4,5,1,2)

def test_normalization(tmp_path):

    base = tmp_path / "datasets"
    folder = base / "testset"
    folder.mkdir(parents=True)

    arr = np.ones((2,2,2,2)) * 10

    np.save(folder / "data.npy", arr)

    out = load_and_preprocess_data(
        folder_names=["testset"],
        base_path=str(base),
        fourier_axes=None,
        normalization=True
    )

    assert np.max(np.abs(out)) == 1
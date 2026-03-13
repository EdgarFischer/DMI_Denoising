from types import SimpleNamespace
import numpy as np
import torch

import denoising.inference.api as api


def make_cfg(image_axes, channel_axis=None, fourier_axes=(), normalization=False):
    return SimpleNamespace(
        data=SimpleNamespace(
            image_axes=tuple(image_axes),
            channel_axis=channel_axis,
            fourier_axes=tuple(fourier_axes),
            normalization=normalization,
        ),
        model=SimpleNamespace(
            features=[8, 16]
        ),
    )


def test_infer_uses_unet2d_and_runs_forward(tmp_path, monkeypatch):
    # --------------------------------------------------
    # arrange
    # --------------------------------------------------
    arr = (
        np.random.randn(4, 5, 3, 6, 2)
        + 1j * np.random.randn(4, 5, 3, 6, 2)
    ).astype(np.complex64)

    input_path = tmp_path / "input.npy"
    np.save(input_path, arr)

    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"dummy")

    cfg = make_cfg(
        image_axes=(0, 1),   # -> 2D
        channel_axis=None,
        fourier_axes=(),
        normalization=False,
    )

    calls = {
        "unet2d_init": None,
        "unet3d_init": None,
        "forward_called": 0,
        "load_state_called": 0,
    }

    class DummyModel(torch.nn.Module):
        def __init__(self, in_channels, out_channels, features):
            super().__init__()
            calls["unet2d_init"] = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "features": features,
            }

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):
            calls["load_state_called"] += 1
            return

        def forward(self, x):
            calls["forward_called"] += 1
            return x  # Identität

    class DummyUNet3D:
        def __init__(self, *args, **kwargs):
            calls["unet3d_init"] = True
            raise AssertionError("UNet3D should not be constructed in this test")

    monkeypatch.setattr(api, "UNet2D", DummyModel)
    monkeypatch.setattr(api, "UNet3D", DummyUNet3D)
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: {})

    # --------------------------------------------------
    # act
    # --------------------------------------------------
    y_out, meta = api.infer(
        cfg=cfg,
        ckpt_path=ckpt_path,
        input_path=input_path,
        batch_size=4,
        device="cpu",
    )

    # --------------------------------------------------
    # assert
    # --------------------------------------------------
    assert calls["unet2d_init"] is not None
    assert calls["unet3d_init"] is None
    assert calls["load_state_called"] == 1
    assert calls["forward_called"] > 0

    assert y_out.shape == arr.shape
    assert y_out.dtype == np.complex64

    assert meta["spatial_dim"] == 2
    assert meta["in_channels"] == 2
    assert meta["image_axes"] == [0, 1]
    assert meta["channel_axis"] is None

def test_infer_model_input_shape_without_channel_axis(tmp_path, monkeypatch):
    # --------------------------------------------------
    # arrange
    # --------------------------------------------------
    arr = (
        np.random.randn(4, 5, 3, 6, 2)
        + 1j * np.random.randn(4, 5, 3, 6, 2)
    ).astype(np.complex64)

    input_path = tmp_path / "input.npy"
    np.save(input_path, arr)

    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"dummy")

    cfg = make_cfg(
        image_axes=(0, 1),
        channel_axis=None,
        fourier_axes=(),
        normalization=False,
    )

    seen = {
        "input_shape": None,
    }

    class DummyModel(torch.nn.Module):
        def __init__(self, in_channels, out_channels, features):
            super().__init__()

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):
            return

        def forward(self, x):
            seen["input_shape"] = tuple(x.shape)
            return x

    monkeypatch.setattr(api, "UNet2D", DummyModel)
    monkeypatch.setattr(api, "UNet3D", DummyModel)
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: {})

    # --------------------------------------------------
    # act
    # --------------------------------------------------
    api.infer(
        cfg=cfg,
        ckpt_path=ckpt_path,
        input_path=input_path,
        batch_size=64,
        device="cpu",
    )

    # --------------------------------------------------
    # assert
    # remaining axes = Z,F,T = 3,6,2  -> batch N = 36
    # model input should be (N, 2, X, Y) = (36, 2, 4, 5)
    # --------------------------------------------------
    assert seen["input_shape"] == (36, 2, 4, 5)

def test_infer_uses_unet3d_and_input_shape_without_channel_axis(tmp_path, monkeypatch):
    # --------------------------------------------------
    # arrange
    # --------------------------------------------------
    arr = (
        np.random.randn(4, 5, 3, 6, 2)
        + 1j * np.random.randn(4, 5, 3, 6, 2)
    ).astype(np.complex64)

    input_path = tmp_path / "input.npy"
    np.save(input_path, arr)

    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"dummy")

    cfg = make_cfg(
        image_axes=(1, 3, 0),   # spatial sizes -> (5, 6, 4)
        channel_axis=None,
        fourier_axes=(),
        normalization=False,
    )

    seen = {
        "input_shape": None,
        "unet2d_init": False,
        "unet3d_init": None,
    }

    class DummyUNet3D(torch.nn.Module):
        def __init__(self, in_channels, out_channels, features):
            super().__init__()
            seen["unet3d_init"] = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "features": features,
            }

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):
            return

        def forward(self, x):
            seen["input_shape"] = tuple(x.shape)
            return x

    class DummyUNet2D:
        def __init__(self, *args, **kwargs):
            seen["unet2d_init"] = True
            raise AssertionError("UNet2D should not be constructed in this test")

    monkeypatch.setattr(api, "UNet2D", DummyUNet2D)
    monkeypatch.setattr(api, "UNet3D", DummyUNet3D)
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: {})

    # --------------------------------------------------
    # act
    # --------------------------------------------------
    y_out, meta = api.infer(
        cfg=cfg,
        ckpt_path=ckpt_path,
        input_path=input_path,
        batch_size=64,
        device="cpu",
    )

    # --------------------------------------------------
    # assert
    # image_axes = (1,3,0) -> spatial = (5,6,4)
    # remaining axes = (2,4) -> (3,2) -> N = 6
    # expected model input: (6, 2, 5, 6, 4)
    # --------------------------------------------------
    assert seen["unet2d_init"] is False
    assert seen["unet3d_init"] is not None
    assert seen["input_shape"] == (6, 2, 5, 6, 4)

    assert y_out.shape == arr.shape
    assert y_out.dtype == np.complex64

    assert meta["spatial_dim"] == 3
    assert meta["in_channels"] == 2
    assert meta["image_axes"] == [1, 3, 0]
    assert meta["channel_axis"] is None

def test_infer_model_input_shape_with_channel_axis_for_3d_model(tmp_path, monkeypatch):
    # --------------------------------------------------
    # arrange
    # --------------------------------------------------
    arr = (
        np.random.randn(4, 5, 3, 6, 2)
        + 1j * np.random.randn(4, 5, 3, 6, 2)
    ).astype(np.complex64)

    input_path = tmp_path / "input.npy"
    np.save(input_path, arr)

    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"dummy")

    cfg = make_cfg(
        image_axes=(1, 3, 0),   # spatial sizes -> (5, 6, 4)
        channel_axis=2,         # size 3
        fourier_axes=(),
        normalization=False,
    )

    seen = {
        "input_shape": None,
        "init_args": None,
        "unet2d_init": False,
    }

    class DummyUNet3D(torch.nn.Module):
        def __init__(self, in_channels, out_channels, features):
            super().__init__()
            seen["init_args"] = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "features": features,
            }

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):
            return

        def forward(self, x):
            seen["input_shape"] = tuple(x.shape)
            return x

    class DummyUNet2D:
        def __init__(self, *args, **kwargs):
            seen["unet2d_init"] = True
            raise AssertionError("UNet2D should not be constructed in this test")

    monkeypatch.setattr(api, "UNet2D", DummyUNet2D)
    monkeypatch.setattr(api, "UNet3D", DummyUNet3D)
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: {})

    # --------------------------------------------------
    # act
    # --------------------------------------------------
    y_out, meta = api.infer(
        cfg=cfg,
        ckpt_path=ckpt_path,
        input_path=input_path,
        batch_size=64,
        device="cpu",
    )

    # --------------------------------------------------
    # assert
    # image_axes = (1,3,0) -> spatial = (5,6,4)
    # channel_axis = 2 -> C = 3 -> model channels = 2*C = 6
    # remaining axis = 4 -> N = 2
    # expected model input: (2, 6, 5, 6, 4)
    # --------------------------------------------------
    assert seen["unet2d_init"] is False
    assert seen["init_args"] is not None
    assert seen["init_args"]["in_channels"] == 6
    assert seen["init_args"]["out_channels"] == 6

    assert seen["input_shape"] == (2, 6, 5, 6, 4)

    assert y_out.shape == arr.shape
    assert y_out.dtype == np.complex64

    assert meta["spatial_dim"] == 3
    assert meta["in_channels"] == 6
    assert meta["image_axes"] == [1, 3, 0]
    assert meta["channel_axis"] == 2
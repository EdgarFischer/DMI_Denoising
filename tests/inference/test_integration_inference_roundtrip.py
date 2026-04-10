import numpy as np
import torch
import pytest

from types import SimpleNamespace

import denoising.inference.api as api


def make_cfg_roundtrip(image_axes, channel_axis=None, fourier_axes=(), normalization=False):
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


def make_cfg_infer(
    *,
    image_axes,
    channel_axis,
    fourier_axes=(3,),
    normalization=True,
    patching_enabled=False,
    patch_sizes=None,
    patch_strides=None,
    weight_mode="average",
):
    return SimpleNamespace(
        data=SimpleNamespace(
            image_axes=list(image_axes),
            channel_axis=channel_axis,
            fourier_axes=list(fourier_axes),
            normalization=normalization,
        ),
        patching=SimpleNamespace(
            enabled=patching_enabled,
            patch_sizes=list(patch_sizes) if patch_sizes is not None else None,
        ),
        inference=SimpleNamespace(
            patch_strides=list(patch_strides) if patch_strides is not None else None,
            weight_mode=weight_mode,
        ),
        model=SimpleNamespace(
            features=[8, 16],
        ),
    )


class IdentityNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return x


def make_complex_test_volume(shape=(5, 6, 7, 8, 9), seed=0):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


@pytest.fixture
def dummy_ckpt(tmp_path):
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"dummy")
    return ckpt


@pytest.fixture
def patch_identity_model(monkeypatch):
    """
    Replace UNet2D/UNet3D with an identity model and torch.load with empty state.
    """
    created_models = []

    def _factory(in_channels, out_channels, features):
        m = IdentityNet(in_channels, out_channels, features)
        created_models.append(m)
        return m

    monkeypatch.setattr(api, "UNet2D", _factory)
    monkeypatch.setattr(api, "UNet3D", _factory)
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: {"model_state": {}})

    return created_models


def test_infer_identity_roundtrip_preserves_complex_volume_with_channel_axis(tmp_path, monkeypatch):
    """
    Identity model + no FFT + no normalization
    => output must reconstruct the original complex volume exactly.
    """
    rng = np.random.default_rng(0)
    arr = (
        rng.standard_normal((4, 5, 3, 6, 2))
        + 1j * rng.standard_normal((4, 5, 3, 6, 2))
    ).astype(np.complex64)

    input_path = tmp_path / "input.npy"
    np.save(input_path, arr)

    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"dummy")

    cfg = make_cfg_roundtrip(
        image_axes=(1, 3, 0),
        channel_axis=2,
        fourier_axes=(),
        normalization=False,
    )

    class DummyUNet3D(torch.nn.Module):
        def __init__(self, in_channels, out_channels, features):
            super().__init__()

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):
            return

        def forward(self, x):
            return x

    class DummyUNet2D:
        def __init__(self, *args, **kwargs):
            raise AssertionError("UNet2D should not be constructed in this test")

    monkeypatch.setattr(api, "UNet2D", DummyUNet2D)
    monkeypatch.setattr(api, "UNet3D", DummyUNet3D)
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: {})

    y_out, meta = api.infer(
        cfg=cfg,
        ckpt_path=ckpt_path,
        input_path=input_path,
        batch_size=64,
        device="cpu",
    )

    assert y_out.shape == arr.shape
    assert y_out.dtype == np.complex64
    assert np.allclose(y_out, arr)

    assert meta["spatial_dim"] == 3
    assert meta["image_axes"] == [1, 3, 0]
    assert meta["channel_axis"] == 2


def test_infer_full_volume_identity_and_branch(tmp_path, dummy_ckpt, monkeypatch, patch_identity_model):
    """
    Full-volume inference:
    - should NOT call patch inference helper
    - should preserve the original complex input exactly
    """
    x = make_complex_test_volume(shape=(5, 6, 7, 8, 9), seed=1)
    input_path = tmp_path / "input.npy"
    np.save(input_path, x)

    cfg = make_cfg_infer(
        image_axes=[0, 1, 2],
        channel_axis=4,
        fourier_axes=[3],
        normalization=True,
        patching_enabled=False,
        patch_sizes=None,
        patch_strides=None,
        weight_mode="average",
    )

    patch_called = {"value": False}

    def _forbidden_patch_runner(*args, **kwargs):
        patch_called["value"] = True
        raise AssertionError("Patch inference helper should not be called in full-volume inference.")

    monkeypatch.setattr(api, "_run_model_on_patches_structured", _forbidden_patch_runner)

    y, meta = api.infer(
        cfg=cfg,
        ckpt_path=dummy_ckpt,
        input_path=input_path,
        batch_size=4,
        device="cpu",
    )

    assert patch_called["value"] is False
    assert len(patch_identity_model) == 1
    assert patch_identity_model[0].forward_calls > 0

    np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-5)
    assert meta["output_shape"] == list(x.shape)


@pytest.mark.parametrize("channel_axis", [3, 4])
def test_infer_patch_identity_and_branch_xyz_plus_extra_axis(
    tmp_path,
    dummy_ckpt,
    monkeypatch,
    patch_identity_model,
    channel_axis,
):
    """
    Patch inference identity tests for:
      - image_axes = [0,1,2]
      - channel_axis = 3 or 4
    """
    x = make_complex_test_volume(shape=(5, 6, 7, 8, 9), seed=2 + channel_axis)
    input_path = tmp_path / f"input_ca{channel_axis}.npy"
    np.save(input_path, x)

    if channel_axis == 3:
        patch_sizes = [4, 3, 4, 5]
        patch_strides = [2, 2, 2, 3]
    else:
        patch_sizes = [4, 3, 4, 5]
        patch_strides = [2, 2, 2, 3]

    cfg = make_cfg_infer(
        image_axes=[0, 1, 2],
        channel_axis=channel_axis,
        fourier_axes=[3],
        normalization=True,
        patching_enabled=True,
        patch_sizes=patch_sizes,
        patch_strides=patch_strides,
        weight_mode="average",
    )

    patch_called = {"value": False}
    orig_patch_runner = api._run_model_on_patches_structured

    def _wrapped_patch_runner(*args, **kwargs):
        patch_called["value"] = True
        return orig_patch_runner(*args, **kwargs)

    monkeypatch.setattr(api, "_run_model_on_patches_structured", _wrapped_patch_runner)

    y, meta = api.infer(
        cfg=cfg,
        ckpt_path=dummy_ckpt,
        input_path=input_path,
        batch_size=8,
        device="cpu",
    )

    assert patch_called["value"] is True
    assert len(patch_identity_model) == 1
    assert patch_identity_model[0].forward_calls > 0

    np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-5)
    assert meta["output_shape"] == list(x.shape)


@pytest.mark.parametrize(
    "image_axes, channel_axis, patch_sizes, patch_strides",
    [
        ([0, 1, 3], 2, [4, 3, 4, 5], [2, 2, 2, 3]),
        ([0, 2, 3], 1, [4, 3, 4, 5], [2, 2, 2, 3]),
        ([1, 2, 3], 0, [4, 3, 4, 5], [2, 2, 2, 3]),
    ],
)
def test_infer_patch_identity_for_multiple_axis_configurations(
    tmp_path,
    dummy_ckpt,
    monkeypatch,
    patch_identity_model,
    image_axes,
    channel_axis,
    patch_sizes,
    patch_strides,
):
    """
    Broader axis-handling test:
    several valid (image_axes, channel_axis) combinations should stay identity.
    """
    x = make_complex_test_volume(shape=(5, 6, 7, 8, 9), seed=11)
    input_path = tmp_path / f"input_axes_{channel_axis}.npy"
    np.save(input_path, x)

    cfg = make_cfg_infer(
        image_axes=image_axes,
        channel_axis=channel_axis,
        fourier_axes=[3],
        normalization=True,
        patching_enabled=True,
        patch_sizes=patch_sizes,
        patch_strides=patch_strides,
        weight_mode="average",
    )

    patch_called = {"value": False}
    orig_patch_runner = api._run_model_on_patches_structured

    def _wrapped_patch_runner(*args, **kwargs):
        patch_called["value"] = True
        return orig_patch_runner(*args, **kwargs)

    monkeypatch.setattr(api, "_run_model_on_patches_structured", _wrapped_patch_runner)

    y, meta = api.infer(
        cfg=cfg,
        ckpt_path=dummy_ckpt,
        input_path=input_path,
        batch_size=8,
        device="cpu",
    )

    assert patch_called["value"] is True
    np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-5)
    assert meta["output_shape"] == list(x.shape)
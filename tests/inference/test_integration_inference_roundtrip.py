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


def test_infer_identity_roundtrip_preserves_complex_volume_with_channel_axis(tmp_path, monkeypatch):
    # --------------------------------------------------
    # arrange
    # Use a nontrivial axis setup to test permutation logic.
    # arr shape = (4, 5, 3, 6, 2)
    # --------------------------------------------------
    rng = np.random.default_rng(0)
    arr = (
        rng.standard_normal((4, 5, 3, 6, 2))
        + 1j * rng.standard_normal((4, 5, 3, 6, 2))
    ).astype(np.complex64)

    input_path = tmp_path / "input.npy"
    np.save(input_path, arr)

    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"dummy")

    cfg = make_cfg(
        image_axes=(1, 3, 0),   # nontrivial order
        channel_axis=2,         # also nontrivial
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
    # Identity model + no FFT + no normalization
    # => output must exactly reconstruct the original complex volume
    # --------------------------------------------------
    assert y_out.shape == arr.shape
    assert y_out.dtype == np.complex64
    assert np.allclose(y_out, arr)

    assert meta["spatial_dim"] == 3
    assert meta["image_axes"] == [1, 3, 0]
    assert meta["channel_axis"] == 2
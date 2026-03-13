from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from denoising.data.mrsi_nd_dataset import MRSiNDataset
from denoising.data.transforms import StratifiedAxisMasking
from denoising.models.unet2d import UNet2D
from denoising.losses.n2v_loss import masked_mse_loss


def test_training_step_runs_end_to_end_2d():
    # --------------------------------------------------
    # arrange
    # standardized internal shape: (X, Y, Z, F, T, D)
    # --------------------------------------------------
    rng = np.random.default_rng(0)
    arr = (
        rng.standard_normal((6, 5, 2, 4, 1, 1))
        + 1j * rng.standard_normal((6, 5, 2, 4, 1, 1))
    ).astype(np.complex64)

    transform = StratifiedAxisMasking(
        num_masked_pixels=4,
        window_size=3,
        random_mask_noisy=False,
    )

    ds = MRSiNDataset(
        data=arr,
        image_axes=(0, 1),      # X,Y go to network
        channel_axis=None,
        masked_axes=(0, 1),     # global masked axes = X,Y
        transform=transform,
    )

    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = UNet2D(
        in_channels=2,
        out_channels=2,
        features=[4, 8],
    )

    x, target, mask = next(iter(loader))

    # --------------------------------------------------
    # sanity checks on batch shapes
    # expected: (B, C, X, Y) with B=1, C=2
    # --------------------------------------------------
    assert x.ndim == 4
    assert target.ndim == 4
    assert mask.ndim == 4

    assert x.shape[0] == 1
    assert x.shape[1] == 2
    assert target.shape == x.shape
    assert mask.shape == x.shape

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------------------------------
    # act
    # --------------------------------------------------
    optimizer.zero_grad()

    pred = model(x)
    loss = masked_mse_loss(pred, target, mask)

    loss.backward()
    optimizer.step()

    # --------------------------------------------------
    # assert
    # --------------------------------------------------
    assert torch.isfinite(loss).item()
    assert loss.ndim == 0

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert any(torch.isfinite(g).all().item() for g in grads if g is not None)
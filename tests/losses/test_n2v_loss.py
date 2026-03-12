import torch
from denoising.losses.n2v_loss import masked_mse_loss


def test_masked_mse_loss_only_uses_masked_pixels():

    pred = torch.tensor([[[[3., 4.]]]])   # shape (1,1,1,2)
    tgt  = torch.tensor([[[[0., 0.]]]])

    mask = torch.tensor([[[[1., 0.]]]])   # nur erster Pixel zählt

    loss = masked_mse_loss(pred, tgt, mask)

    # nur erster Pixel:
    # (2-0)^2 = 4
    expected = torch.tensor(9.0)

    assert torch.isclose(loss, expected)
import torch
from denoising.losses.n2v_loss import (
    masked_mse_loss,
    residual_variance_scaled_masked_mse_loss,
)


def test_masked_mse_loss_only_uses_masked_pixels():

    pred = torch.tensor([[[[3., 4.]]]])   # shape (1,1,1,2)
    tgt  = torch.tensor([[[[0., 0.]]]])

    mask = torch.tensor([[[[1., 0.]]]])   # nur erster Pixel zählt

    loss = masked_mse_loss(pred, tgt, mask)

    # nur erster Pixel:
    # (2-0)^2 = 4
    expected = torch.tensor(9.0)

    assert torch.isclose(loss, expected)


def test_residual_variance_scaling_matches_detached_sigma_squared():
    pred = torch.tensor(
        [[[[[1.0, 2.0, 4.0, 8.0]]], [[[0.5, 1.0, 2.0, 4.0]]]]],
        requires_grad=True,
    )
    tgt = torch.zeros_like(pred)
    mask = torch.ones((1, 1, 1, 1, 4))
    frequency_mask = torch.tensor([False, True, True, True])

    loss = residual_variance_scaled_masked_mse_loss(
        pred,
        tgt,
        mask,
        spectral_axis=2,
        frequency_mask=frequency_mask,
        epsilon=1e-8,
    )

    with torch.no_grad():
        residual_magnitude = torch.linalg.vector_norm(pred, dim=1)
        sigma = residual_magnitude[..., frequency_mask].std(
            dim=-1, correction=1, keepdim=True
        ) + 1e-8
        expected = masked_mse_loss(
            pred, tgt, mask, weight=sigma.unsqueeze(1).square().reciprocal()
        )
    assert torch.allclose(loss, expected)


def test_residual_sigma_is_detached_from_gradient():
    pred = torch.tensor(
        [[[[[1.0, 2.0, 4.0]]], [[[0.5, 1.0, 2.0]]]]],
        requires_grad=True,
    )
    tgt = torch.zeros_like(pred)
    mask = torch.ones((1, 1, 1, 1, 3))

    loss = residual_variance_scaled_masked_mse_loss(
        pred, tgt, mask, spectral_axis=2, epsilon=1e-8
    )
    loss.backward()

    with torch.no_grad():
        sigma = torch.linalg.vector_norm(pred, dim=1).std(
            dim=-1, correction=1, keepdim=True
        ) + 1e-8
        expected_gradient = 2.0 * pred / (pred.numel() * sigma.unsqueeze(1).square())
    assert torch.allclose(pred.grad, expected_gradient)

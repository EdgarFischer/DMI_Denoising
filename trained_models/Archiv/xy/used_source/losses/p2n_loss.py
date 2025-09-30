import torch

def p2n_loss(
    x_p: torch.Tensor,
    x_n: torch.Tensor
) -> torch.Tensor:
    """
    Positive2Negative L2 consistency loss.

    Computes the mean squared error between x_p and x_n.

    Args:
        x_p (torch.Tensor): Denoised output for positively scaled noise variant, shape (B, C, H, W)
        x_n (torch.Tensor): Denoised output for negatively scaled noise variant, shape (B, C, H, W)

    Returns:
        torch.Tensor: Scalar MSE loss value averaged over batch and spatial dimensions.
    """
    # Difference between the two outputs
    diff = x_p - x_n  # shape (B, C, H, W)

    # Mean squared error
    loss = (diff ** 2).mean()
    return loss


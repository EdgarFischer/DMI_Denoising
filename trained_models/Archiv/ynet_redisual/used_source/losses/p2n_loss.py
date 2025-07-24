# losses/p2n_loss.py
# -----------------------------------------------------------------------------
# Loss-Funktionen für Positive2Negative / Denoised Consistency Supervision.
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F

__all__ = ["p2n_loss", "dc_loss", "p2n_total_loss"]


def p2n_loss(x_pos: torch.Tensor,
             x_neg: torch.Tensor,
             p: float = 1.5,
             reduction: str = "mean") -> torch.Tensor:
    """Positive2Negative / DCS Loss: ||x_pos - x_neg||_p."""
    diff = (x_pos - x_neg).abs().pow(p)
    if reduction == "mean":
        return diff.mean()
    elif reduction == "sum":
        return diff.sum()
    elif reduction == "none":
        return diff
    raise ValueError(f"Unknown reduction '{reduction}'.")


def dc_loss(x_hat: torch.Tensor,
            inp: torch.Tensor,
            mode: str = "mse",
            weight: torch.Tensor | None = None,
            delta: float = 1.0) -> torch.Tensor:
    """
    Data-Consistency (Anker). Default: MSE.
    mode: 'mse' | 'l1' | 'huber'
    """
    if weight is not None:
        # broadcast weight
        while weight.dim() < x_hat.dim():
            weight = weight.unsqueeze(1)
        diff = x_hat - inp
        if mode == "l1":
            return (weight * diff.abs()).mean()
        elif mode == "huber":
            absd = diff.abs()
            quad = torch.clamp(absd, max=delta)
            lin  = absd - quad
            return (weight * (0.5*quad**2 + delta*lin)).mean()
        else:  # mse
            return (weight * diff.pow(2)).mean()

    if mode == "l1":
        return F.l1_loss(x_hat, inp)
    elif mode == "huber":
        return F.smooth_l1_loss(x_hat, inp, beta=delta)
    else:  # mse default
        return F.mse_loss(x_hat, inp)


def p2n_total_loss(x_pos: torch.Tensor,
                   x_neg: torch.Tensor,
                   x_hat: torch.Tensor | None = None,
                   inp: torch.Tensor | None = None,
                   lambda_dc: float = 1.0,
                   lambda_dcs: float = 0.1,
                   p: float = 1.5,
                   mode: str = "mse",
                   weight: torch.Tensor | None = None,
                   delta: float = 1.0,
                   reduction: str = "mean"):
    """
    Gesamt-Loss: L = λ_dc * DC(x_hat, inp) + λ_dcs * P2N(x_pos, x_neg).
    Gibt (total, loss_dc, loss_dcs) zurück.
    """
    loss_dcs = p2n_loss(x_pos, x_neg, p=p, reduction=reduction)

    if (lambda_dc > 0.0) and (x_hat is not None) and (inp is not None):
        loss_dc = dc_loss(x_hat, inp, mode=mode, weight=weight, delta=delta)
    else:
        loss_dc = loss_dcs.new_zeros(())

    total = lambda_dc * loss_dc + lambda_dcs * loss_dcs
    return total, loss_dc, loss_dcs




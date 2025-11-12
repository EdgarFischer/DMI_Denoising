import torch

def masked_mse_loss(pred: torch.Tensor,
                    tgt : torch.Tensor,
                    mask: torch.Tensor,
                    weight: torch.Tensor | None = None) -> torch.Tensor:
    """
    MSE nur über maskierte Positionen.
    Optional: zusätzlicher Gewicht-Tensor gleicher Shape wie pred
    """
    if mask.dtype != pred.dtype:
        mask = mask.to(dtype=pred.dtype, device=pred.device)
    else:
        mask = mask.to(device=pred.device)

    if weight is not None:
        weight = weight.to(dtype=pred.dtype, device=pred.device)

    if mask.dim() != pred.dim():
        raise ValueError("mask muss gleiche Dimensionalität wie pred haben.")
    if mask.size(1) == 1:
        mask = mask.expand_as(pred)
    elif mask.size(1) != pred.size(1):
        raise ValueError("mask Kanäle ungleich pred Kanäle.")

    diff = (pred - tgt) ** 2
    if weight is not None:
        diff = diff * weight           # <- Curriculum-Gewicht

    diff = diff * mask
    denom = mask.sum()
    return diff.sum() / denom if denom.item() else diff.sum() * 0.0




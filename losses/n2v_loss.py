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

def combined_loss_simple(y_hat: torch.Tensor,
                         x_raw: torch.Tensor,
                         x_tmppca: torch.Tensor,
                         B: torch.Tensor,
                         alpha: float = 100.0) -> torch.Tensor:
    """
    Proof-of-principle: Noise2Self (roh) + tmppca-Anker,
    beide über dieselbe Blind-Spot-Maske B (keine Peak-Maske).

    L = MSE_B(y, x_raw) + alpha * MSE_B(y, x_tmppca)
    """
    # Targets nicht durch den Graph propagieren
    x_raw_t    = x_raw.detach()
    x_tmppca_t = x_tmppca.detach()

    # Falls B nur 1 Kanal hat, übernimmt masked_mse_loss das Expand
    B = B.to(device=y_hat.device)

    L_raw = masked_mse_loss(y_hat, x_raw_t,    mask=B)      # Noise2Self-Teil
    L_tmp = masked_mse_loss(y_hat, x_tmppca_t, mask=B)      # tmppca-Anker

    return L_tmp #L_raw + alpha * L_tmp


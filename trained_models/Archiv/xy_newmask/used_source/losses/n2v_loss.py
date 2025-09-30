# losses/n2v_loss.py

import torch

def masked_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    MSE nur auf maskierten Pixel-Kanal-Paaren:
    output, target: (B, C, H, W)
    mask:           (B, 1, H, W)
    """

    # 1) Quadrierter Fehler
    diff2 = (output - target) ** 2      # → (B, C, H, W)

    # 2) Maske auf Kanal-Ebene erweitern
    mask_expanded = mask.expand_as(diff2)  # → (B, C, H, W)

    # 3) Nur maskierte Positionen zählen
    masked_diff2 = diff2 * mask_expanded

    # 4) Summe der maskierten quadrierten Fehler
    total_error = masked_diff2.sum()       # Skalar

    # 5) Anzahl aller maskierten Pixel-Kanal-Paare
    num_masked = mask_expanded.sum()       # Skalar

    # 6) Durchschnitt = total_error / num_masked
    return total_error / num_masked


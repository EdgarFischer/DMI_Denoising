# losses/n2v_loss.py
import torch


def masked_mse_loss(pred: torch.Tensor,
                    tgt: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    MSE nur über maskierte Positionen.
    Inputs:
      pred : (B,C,H,W)
      tgt  : (B,C,H,W)
      mask : (B,1,H,W) *oder* (B,C,H,W)
             (bei Y-Net: (B,2,H,W) für Noisy-Kanäle)

    Broadcast-Regeln:
      - Wenn mask.shape[1] == 1 → auf alle C Kanäle expandieren
      - Wenn mask.shape[1] == C_pred → direkt verwenden
    """
    if mask.dtype != pred.dtype:
        mask = mask.to(dtype=pred.dtype, device=pred.device)
    else:
        mask = mask.to(device=pred.device)

    if mask.dim() != pred.dim():
        raise ValueError("mask muss gleiche Dimensionalität wie pred haben (B,C,H,W) oder (B,1,H,W).")

    if mask.size(1) == 1:
        mask = mask.expand_as(pred)
    elif mask.size(1) != pred.size(1):
        raise ValueError(f"mask.shape[1]={mask.size(1)} passt nicht zu pred.shape[1]={pred.size(1)}.")

    diff = (pred - tgt) ** 2
    diff = diff * mask
    denom = mask.sum()
    if denom.item() == 0:
        # kein maskiertes Pixel → Loss=0
        return diff.sum() * 0.0
    return diff.sum() / denom



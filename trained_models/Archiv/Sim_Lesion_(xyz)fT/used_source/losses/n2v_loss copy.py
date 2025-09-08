# losses/n2v_loss.py
import torch


def masked_mse_loss(pred: torch.Tensor,
                    tgt: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    MSE nur über maskierte Positionen,
    Frequenz-Bins >95 (H-Achse) werden 10× stärker gewichtet.
    Inputs:
      pred : (B,C,H,W)
      tgt  : (B,C,H,W)
      mask : (B,1,H,W) oder (B,C,H,W)
    """
    # 1) Mask-Type/Device anpassen
    if mask.dtype != pred.dtype:
        mask = mask.to(pred.dtype, pred.device)
    else:
        mask = mask.to(pred.device)

    # 2) Dimension prüfen & ggf. auf (B,C,H,W) ausdehnen
    if mask.dim() != pred.dim():
        raise ValueError("mask muss (B,1,H,W) oder (B,C,H,W) sein.")
    if mask.size(1) == 1:
        mask = mask.expand_as(pred)
    elif mask.size(1) != pred.size(1):
        raise ValueError(f"mask.shape[1]={mask.size(1)} passt nicht zu pred.shape[1]={pred.size(1)}.")

    # 3) Weight-Map bauen: alle H-Bins <=95 → 1, >95 → 10
    #    mask hat Shape (B,C,H,W), wir nutzen dieselbe Shape
    weight = torch.ones_like(mask)
    weight[:, :, 96:, :] = 10.0  # falls 0-Index: H-Bins 96… end sind >95

    # 4) Loss-Berechnung
    diff = (pred - tgt)**2
    diff = diff * mask * weight

    # 5) Normierung
    denom = (mask * weight).sum()
    if denom.item() == 0:
        return diff.sum() * 0.0
    return diff.sum() / denom




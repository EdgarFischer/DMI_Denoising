"""losses/supervised_to_lowrank_loss.py

Einfache Loss-Funktionen für *supervised* Training gegen Low-Rank-Targets.

Orientiert sich stilistisch an `masked_mse_loss` aus `n2v_loss.py`,
aber generischer gehalten:

    supervised_to_lowrank_loss(pred, tgt, mask=None, weight=None)

- `pred`, `tgt`: Tensor (B,C,...) – beliebige räumliche Dimensionalität.
- `mask`      : optional; broadcast-kompatibel zu `pred`.
                Typische Formen: (B,1,H,W) oder (B,C,H,W).
- `weight`    : optional reelle Gewichte (z.B. Peak-Gewichtungen) – gleiche Broadcast-Regeln.
- Reduktion   : standardmäßig Mittelwert über maskierte (und ggf. gewichtete) Elemente.

Zusätzlich:
- `supervised_l1_loss` falls du lieber L1 möchtest.
- `supervised_huber_loss` (smooth L1) mit delta.

Alle Losses geben *Skalar* zurück.
"""

from __future__ import annotations
import torch
from typing import Optional


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def _expand_mask_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast-Helfer wie in n2v_loss.

    Erwartet x.shape == (B,1,...) oder (B,C,...). Gibt expandierten Tensor zurück,
    so dass shape == ref.shape.
    """
    if x.dim() != ref.dim():
        # Versuche len-Diff via unsqueeze auf Kanalachse zu korrigieren
        if x.dim() == ref.dim() - 1:  # (B,...) → (B,1,...)
            x = x.unsqueeze(1)
        else:
            raise ValueError("Mask/Weight muss gleiche Dimensionalität wie pred haben (oder (B,...) broadcastbar).")

    if x.size(0) != ref.size(0):
        raise ValueError("Batch-Dimension mismatch zwischen Mask/Weight und pred.")

    if x.size(1) == 1 and ref.size(1) > 1:
        x = x.expand(-1, ref.size(1), *([-1] * (ref.dim() - 2)))
    elif x.size(1) not in (1, ref.size(1)):
        raise ValueError(f"Mask/Weight Kanal-Dimension {x.size(1)} passt nicht zu pred {ref.size(1)}.")
    return x


# -----------------------------------------------------------------------------
# Kern: MSE
# -----------------------------------------------------------------------------
def supervised_to_lowrank_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE zwischen `pred` und `tgt`, optional maskiert + gewichtet.

    Args:
        pred: (B,C,*)
        tgt : (B,C,*) – gleiche Shape wie pred.
        mask: optional; (B,1,*) oder (B,C,*). Bool/float.
        weight: optional; (B,1,*) oder (B,C,*). float.
        reduction: 'mean' | 'sum' | 'none'.
    """
    if pred.shape != tgt.shape:
        raise ValueError("pred und tgt müssen gleiche Shape haben.")

    diff = (pred - tgt) ** 2  # (B,C,...)

    if mask is not None:
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        mask = _expand_mask_like(mask, pred)
        diff = diff * mask

    if weight is not None:
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        weight = _expand_mask_like(weight, pred)
        diff = diff * weight

    if reduction == "none":
        return diff

    if reduction == "sum":
        return diff.sum()

    # mean: gewichtete Mittelung – denom aus Maske*Weight falls vorhanden
    if mask is not None or weight is not None:
        denom = torch.ones_like(diff)
        if mask is not None:
            denom = denom * mask
        if weight is not None:
            denom = denom * weight
        denom_sum = denom.sum()
        if denom_sum.item() == 0:
            return diff.sum() * 0.0
        return diff.sum() / denom_sum

    # plain mean über alle Elemente
    return diff.mean()


# -----------------------------------------------------------------------------
# L1 (Abs) – optional mit Maske/Weight
# -----------------------------------------------------------------------------
def supervised_l1_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if pred.shape != tgt.shape:
        raise ValueError("pred und tgt müssen gleiche Shape haben.")

    diff = (pred - tgt).abs()

    if mask is not None:
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        mask = _expand_mask_like(mask, pred)
        diff = diff * mask

    if weight is not None:
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        weight = _expand_mask_like(weight, pred)
        diff = diff * weight

    if reduction == "none":
        return diff
    if reduction == "sum":
        return diff.sum()
    if mask is not None or weight is not None:
        denom = torch.ones_like(diff)
        if mask is not None:
            denom = denom * mask
        if weight is not None:
            denom = denom * weight
        denom_sum = denom.sum()
        if denom_sum.item() == 0:
            return diff.sum() * 0.0
        return diff.sum() / denom_sum
    return diff.mean()


# -----------------------------------------------------------------------------
# Huber / Smooth-L1 – nützlich wenn Peaks nicht über-bestraft werden sollen
# -----------------------------------------------------------------------------
def supervised_huber_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    if pred.shape != tgt.shape:
        raise ValueError("pred und tgt müssen gleiche Shape haben.")

    abs_err = (pred - tgt).abs()
    quad = torch.clamp(abs_err, max=delta)
    lin  = abs_err - quad
    diff = 0.5 * quad * quad / delta + lin  # Smooth-L1 Kern

    if mask is not None:
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        mask = _expand_mask_like(mask, pred)
        diff = diff * mask

    if weight is not None:
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        weight = _expand_mask_like(weight, pred)
        diff = diff * weight

    if reduction == "none":
        return diff
    if reduction == "sum":
        return diff.sum()
    if mask is not None or weight is not None:
        denom = torch.ones_like(diff)
        if mask is not None:
            denom = denom * mask
        if weight is not None:
            denom = denom * weight
        denom_sum = denom.sum()
        if denom_sum.item() == 0:
            return diff.sum() * 0.0
        return diff.sum() / denom_sum
    return diff.mean()


# -----------------------------------------------------------------------------
# Convenience: alias identisch zu Kernfunktion
# -----------------------------------------------------------------------------
def supervised_mse_loss(*args, **kwargs):
    return supervised_to_lowrank_loss(*args, **kwargs)


__all__ = [
    "supervised_to_lowrank_loss",
    "supervised_mse_loss",
    "supervised_l1_loss",
    "supervised_huber_loss",
]




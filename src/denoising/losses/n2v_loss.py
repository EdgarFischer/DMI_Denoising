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


def residual_variance_scaled_masked_mse_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    spectral_axis: int,
    frequency_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """PHIVE-style residual-variance-scaled masked MSE.

    ``pred`` and ``tgt`` contain real and imaginary parts in channel 1. For
    every voxel, sigma is the spectral standard deviation of the current
    complex residual magnitude. Sigma is deliberately estimated under
    ``no_grad`` exactly like PHIVE, then the otherwise unchanged masked mean
    squared error is divided by sigma squared.
    """
    if pred.shape != tgt.shape:
        raise ValueError("pred and tgt must have identical shapes.")
    if pred.ndim < 3 or pred.shape[1] != 2:
        raise ValueError(
            "Residual scaling expects real/imaginary channels with shape "
            "(B,2,...,F)."
        )
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    spectral_dim = int(spectral_axis) + 2
    if spectral_dim < 2 or spectral_dim >= pred.ndim:
        raise ValueError(f"Invalid spectral_axis {spectral_axis} for {pred.shape}.")

    sigma = residual_standard_deviation(
        pred,
        tgt,
        spectral_axis=spectral_axis,
        frequency_mask=frequency_mask,
        epsilon=epsilon,
    )
    inverse_variance = sigma.square().reciprocal()

    return masked_mse_loss(pred, tgt, mask, weight=inverse_variance)


def residual_standard_deviation(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    spectral_axis: int,
    frequency_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return detached per-voxel PHIVE residual sigma, channel-broadcastable."""
    spectral_dim = int(spectral_axis) + 2
    with torch.no_grad():
        residual = pred - tgt
        magnitude = torch.linalg.vector_norm(residual, dim=1)
        magnitude_spectral_dim = spectral_dim - 1
        if frequency_mask is not None:
            frequency_mask = frequency_mask.to(
                device=pred.device, dtype=torch.bool
            ).reshape(-1)
            if frequency_mask.numel() != pred.shape[spectral_dim]:
                raise ValueError("frequency_mask length does not match spectrum.")
            indices = torch.nonzero(frequency_mask, as_tuple=False).squeeze(-1)
            if indices.numel() < 2:
                raise ValueError("At least two frequency points are required.")
            magnitude = magnitude.index_select(magnitude_spectral_dim, indices)
        sigma = magnitude.std(
            dim=magnitude_spectral_dim, correction=1, keepdim=True
        ).add(float(epsilon))
        return sigma.unsqueeze(1)

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

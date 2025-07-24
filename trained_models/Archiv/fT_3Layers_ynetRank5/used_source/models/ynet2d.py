## models/ynet2d.py
# -----------------------------------------------------------------------------
# Zweipfad-U-Net (Y-Net) für Spektrum×Zeit-Denoising
#   • Pfad 1: verrauschte Eingabe (noisy)
#   • Pfad 2: Low‑Rank‑Approximation (lowrank)
#   • Beide Encoder besitzen eigene Skip‑Connections. Im Decoder werden pro Ebene
#     die Features von (noisy‑skip, lowrank‑skip, upconv‑Output) konkatenert.
#   • Decoder nutzt damit 3× so viele Eingangskanäle wie ein klassisches U‑Net.
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence

# ---------- Hilfsbaustein: 2 × Conv(3×3) -------------------------------------

def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Zwei 3×3‑Faltungen mit ReLU (ohne Normierung/Dropout)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )


# ---------- Y‑Net mit 2 Encodern --------------------------------------------

class YNet2D(nn.Module):
    """Minimaler zweipfadiger 2‑D‑U‑Net‑Decoder (‚Y‑Net‘).

    Parameters
    ----------
    in_ch_noisy : int, default=2
        Kanalanzahl der verrauschten Eingabe (z. B. Real + Imag).
    in_ch_lr : int, default=2
        Kanalanzahl der Low‑Rank‑Eingabe.
    out_channels : int, default=2
        Kanalanzahl der Ausgabe (meistens = in_ch_noisy).
    features : Sequence[int]
        Kanalbreiten der Ebenen, analog zu klassischem U‑Net.
    """

    def __init__(
        self,
        in_ch_noisy: int = 2,
        in_ch_lr: int = 2,
        out_channels: int = 2,
        features: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()

        # ---------- Encoder A (noisy) ----------
        self.enc_noisy = nn.ModuleList()
        self.pool_noisy = nn.ModuleList()
        ch = in_ch_noisy
        for feat in features:
            self.enc_noisy.append(double_conv(ch, feat))
            self.pool_noisy.append(nn.MaxPool2d(kernel_size=2))
            ch = feat

        # ---------- Encoder B (low‑rank) ----------
        self.enc_lr = nn.ModuleList()
        self.pool_lr = nn.ModuleList()
        ch = in_ch_lr
        for feat in features:
            self.enc_lr.append(double_conv(ch, feat))
            self.pool_lr.append(nn.MaxPool2d(kernel_size=2))
            ch = feat

        # ---------- Gemeinsamer Bottleneck ----------
        # Eingang = Concatenation der beiden Encoder‑Ausgänge: 2×features[-1]
        self.bottleneck = double_conv(features[-1] * 2, features[-1] * 2)

        # ---------- Decoder ----------
        self.upconvs: nn.ModuleList[nn.ConvTranspose2d] = nn.ModuleList()
        self.dec_blocks: nn.ModuleList[nn.Sequential] = nn.ModuleList()
        ch = features[-1] * 2  # Ausgangskanäle der Bottleneck

        for feat in reversed(features):
            # Up‑Convolution reduziert auf ‚feat‘ Kanäle
            self.upconvs.append(nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2))
            # Decoder‑Eingang = upconv‑Output (feat) + skip_noisy (feat) + skip_lr (feat)
            self.dec_blocks.append(double_conv(feat * 3, feat))
            ch = feat

        # ---------- Output‑Konvolution ----------
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ---------------- Forward -------------------------------------------------
    def forward(
        self, x_noisy: torch.Tensor, x_lr: torch.Tensor
    ) -> torch.Tensor:
        """Vorwärtsdurchlauf.

        Beide Eingaben müssen die gleiche Auflösung haben: (B, C, H, W).
        """
        assert (
            x_noisy.shape[2:] == x_lr.shape[2:]
        ), "Noisy und Low‑Rank müssen identische räumliche Maße besitzen."

        B, _, H, W = x_noisy.shape
        L = len(self.pool_noisy)  # Anzahl Ebenen → zum symmetrischen Padding
        factor = 1 << L  # 2**L

        # ---------- Symmetrisch aufrunden (für divisibility by 2^L) ----------
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

        x_noisy_p = F.pad(x_noisy, [pad_left, pad_right, pad_top, pad_bottom])
        x_lr_p = F.pad(x_lr, [pad_left, pad_right, pad_top, pad_bottom])

        # ---------- Encoderpfad A (noisy) ----------
        skips_noisy: list[torch.Tensor] = []
        x_a = x_noisy_p
        for enc, pool in zip(self.enc_noisy, self.pool_noisy):
            x_a = enc(x_a)
            skips_noisy.append(x_a)
            x_a = pool(x_a)

        # ---------- Encoderpfad B (low‑rank) ----------
        skips_lr: list[torch.Tensor] = []
        x_b = x_lr_p
        for enc, pool in zip(self.enc_lr, self.pool_lr):
            x_b = enc(x_b)
            skips_lr.append(x_b)
            x_b = pool(x_b)

        # ---------- Bottleneck ----------
        x = torch.cat([x_a, x_b], dim=1)
        x = self.bottleneck(x)

        # ---------- Decoder (mit 2×Skips) ----------
        for up, dec, skip_a, skip_b in zip(
            self.upconvs, self.dec_blocks, reversed(skips_noisy), reversed(skips_lr)
        ):
            x = up(x)
            # Höhe/Breite können durch ungerade Dimensionen minimal abweichen →
            # falls nötig, zuschneiden.
            if x.shape[-2:] != skip_a.shape[-2:]:
                x = F.pad(x, [0, skip_a.shape[-1] - x.shape[-1], 0, skip_a.shape[-2] - x.shape[-2]])
            x = torch.cat([skip_a, skip_b, x], dim=1)
            x = dec(x)

        # ---------- Output & Rückcroppen ----------
        x = self.out_conv(x)
        x = x[..., pad_top : pad_top + H, pad_left : pad_left + W]
        return x


# -----------------------------------------------------------------------------
# Quick‑&‑Dirty Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 2, 129, 257  # absichtlich ungerade zum Check
    model = YNet2D()
    x_noisy = torch.randn(B, C, H, W)
    x_lr = torch.randn(B, C, H, W)
    with torch.no_grad():
        y = model(x_noisy, x_lr)
    print("Output shape:", y.shape)



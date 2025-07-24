# models/ynet2d.py
# -----------------------------------------------------------------------------
# Zweipfad‑U‑Net (Y‑Net) mit **Gated** Skip‑Connections für Spektrum×Zeit‑Denoising
#   • Pfad 1: verrauschte Eingabe (noisy)
#   • Pfad 2: Low‑Rank‑Approximation (lowrank)
#   • Jeder Decoder mischt die beiden Skip‑Features adaptiv:
#       mixed = g * skip_lr + (1‑g) * skip_noisy
#     wobei g∈[0,1] durch eine 1×1‑Conv+Sigmoid pro Ebene gelernt wird.
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

# ---------- 2×3×3‑Conv‑Block --------------------------------------------------

def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )


# ---------- Y‑Net‑Klasse ------------------------------------------------------

class YNet2D(nn.Module):
    """Zweipfadiger 2‑D‑U‑Net‑Decoder mit Gating zwischen Low‑Rank und Noisy."""

    def __init__(
        self,
        in_ch_noisy: int = 2,
        in_ch_lr: int = 2,
        out_channels: int = 2,
        features: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()

        # ---------- Encoder Noisy ----------
        self.enc_noisy, self.pool_noisy = nn.ModuleList(), nn.ModuleList()
        ch = in_ch_noisy
        for feat in features:
            self.enc_noisy.append(double_conv(ch, feat))
            self.pool_noisy.append(nn.MaxPool2d(2))
            ch = feat

        # ---------- Encoder Low‑Rank ----------
        self.enc_lr, self.pool_lr = nn.ModuleList(), nn.ModuleList()
        ch = in_ch_lr
        for feat in features:
            self.enc_lr.append(double_conv(ch, feat))
            self.pool_lr.append(nn.MaxPool2d(2))
            ch = feat

        # ---------- Bottleneck ----------
        self.bottleneck = double_conv(features[-1] * 2, features[-1] * 2)

        # ---------- Decoder ----------
        self.upconvs, self.gates, self.dec_blocks = (nn.ModuleList() for _ in range(3))
        ch = features[-1] * 2
        for feat in reversed(features):
            # UpSampling
            self.upconvs.append(nn.ConvTranspose2d(ch, feat, 2, 2))
            # Gate: 1×1 Conv auf concat(skip_noisy, skip_lr) → feat Kanäle, danach Sigmoid
            self.gates.append(
                nn.Sequential(
                    nn.Conv2d(2 * feat, feat, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
            )
            # Decoder‑Conv erwartet mixed (feat) + up (feat) = 2*feat Kanäle
            self.dec_blocks.append(double_conv(feat * 2, feat))
            ch = feat

        # ---------- Output ----------
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ---------------- Forward -----------------------------------------------
    def forward(self, x_noisy: torch.Tensor, x_lr: torch.Tensor) -> torch.Tensor:
        """Vorwärtsdurchlauf mit originalem Low‑Rank-Pfad (keine Residual‑Subtraktion)."""
        assert x_noisy.shape[2:] == x_lr.shape[2:], "Input‑Shapes müssen übereinstimmen."
        B, _, H, W = x_noisy.shape
        L = len(self.pool_noisy)
        factor = 1 << L

        # ---------- symmetrisches Padding ----------
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        x_noisy_p = F.pad(x_noisy, [pad_left, pad_right, pad_top, pad_bottom])
        x_lr_p    = F.pad(x_lr,    [pad_left, pad_right, pad_top, pad_bottom])

        # ---------- Encoder ----------
        skips_noisy, skips_lr = [], []
        x_a, x_b = x_noisy_p, x_lr_p
        for encA, poolA, encB, poolB in zip(
            self.enc_noisy, self.pool_noisy, self.enc_lr, self.pool_lr
        ):
            y_a = encA(x_a); skips_noisy.append(y_a); x_a = poolA(y_a)
            y_b = encB(x_b); skips_lr  .append(y_b); x_b = poolB(y_b)

        # ---------- Bottleneck ----------
        x = self.bottleneck(torch.cat([x_a, x_b], dim=1))

        # ---------- Decoder mit Gating ----------
        for up, gate, dec, s_noisy, s_lr in zip(
            self.upconvs, self.gates, self.dec_blocks,
            reversed(skips_noisy), reversed(skips_lr)
        ):
            x = up(x)
            if x.shape[-2:] != s_noisy.shape[-2:]:
                x = F.pad(x, [0, s_noisy.shape[-1] - x.shape[-1],
                              0, s_noisy.shape[-2] - x.shape[-2]])
            g = gate(torch.cat([s_noisy, s_lr], dim=1))          # (B,feat,H,W)
            mixed = g * s_lr + (1 - g) * s_noisy                 # konvexe Mischung
            x = dec(torch.cat([mixed, x], dim=1))

        # ---------- Output & Identity-Skip ----------
        out = self.out_conv(x) #+ x_noisy_p
        return out[..., pad_top:pad_top + H, pad_left:pad_left + W]
    
    def forward_with_gates(self, x_noisy: torch.Tensor, x_lr: torch.Tensor):
        """
        Wie forward(), aber gibt zusätzlich alle Gate-Maps zurück.
        Returns:
          out   : Tensor (B, C, H, W)
          gates : List[Tensor] (Gate-Map pro Decoder-Ebene)
        """
        # identischer Aufbau wie forward, jedoch mit Gate-Sammlung
        assert x_noisy.shape[2:] == x_lr.shape[2:], "Input-Shapes müssen übereinstimmen."
        B, _, H, W = x_noisy.shape
        L = len(self.pool_noisy);
        factor = 1 << L
        # Padding
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pad_top, pad_bottom = pad_h//2, pad_h - pad_h//2
        pad_left, pad_right = pad_w//2, pad_w - pad_w//2
        x_noisy_p = F.pad(x_noisy, [pad_left, pad_right, pad_top, pad_bottom])
        x_lr_p    = F.pad(x_lr,    [pad_left, pad_right, pad_top, pad_bottom])
        # Encoder
        skips_noisy, skips_lr = [], []
        x_a, x_b = x_noisy_p, x_lr_p
        for encA, poolA, encB, poolB in zip(self.enc_noisy, self.pool_noisy, self.enc_lr, self.pool_lr):
            y_a = encA(x_a); skips_noisy.append(y_a); x_a = poolA(y_a)
            y_b = encB(x_b); skips_lr .append(y_b); x_b = poolB(y_b)
        # Bottleneck
        x = self.bottleneck(torch.cat([x_a, x_b], dim=1))
        # Decoder
        all_gates = []
        for up, gate, dec, s_noisy, s_lr in zip(self.upconvs, self.gates, self.dec_blocks, reversed(skips_noisy), reversed(skips_lr)):
            x = up(x)
            if x.shape[-2:] != s_noisy.shape[-2:]:
                x = F.pad(x, [0, s_noisy.shape[-1]-x.shape[-1], 0, s_noisy.shape[-2]-x.shape[-2]])
            g = gate(torch.cat([s_noisy, s_lr], dim=1))
            all_gates.append(g)
            mixed = g * s_lr + (1 - g) * s_noisy
            x = dec(torch.cat([mixed, x], dim=1))
        # Output
        out = self.out_conv(x)
        out = out[..., pad_top:pad_top+H, pad_left:pad_left+W]
        return out, all_gates




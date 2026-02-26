# models/ynet2d.py
# -----------------------------------------------------------------------------
# Zweipfad-U-Net (Y-Net) mit reiner Concat-Fusion der Skip-Features (ohne Gates)
#   • Pfad 1: verrauschte Eingabe (noisy)
#   • Pfad 2: Low-Rank-Approximation (lowrank)
#   • Decoder bekommt pro Ebene: concat(skip_noisy, skip_lr, upsampled) = 3*feat
# -----------------------------------------------------------------------------

# models/ynet2d.py
# -----------------------------------------------------------------------------
# Quick&Dirty: 4-Kanal-U-Net (ein Encoder) bei unverändertem Namen/Forward.
#   • Input: concat([noisy_real, noisy_imag, aux_real, aux_imag])  → 4 Kanäle
#   • Output: 2 Kanäle (real, imag)
#   • Beibehaltung der Klasse/Signaturen für Pipeline-Kompatibilität.
# -----------------------------------------------------------------------------

# from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Sequence

# # ---------- 2×3×3-Conv-Block --------------------------------------------------

# def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
#     return nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#         nn.ReLU(inplace=True),
#     )

# # ---------- "YNet2D" (eigentlich 4-Kanal-U-Net) ------------------------------

# class YNet2D(nn.Module):
#     """
#     Pipeline-kompatibler Ersatz: ein Encoder mit 4 Eingangskanälen.
#     Beibehält die Signatur (x_noisy, x_lr), cat't aber intern zu 4 Kanälen.
#     """

#     def __init__(
#         self,
#         in_ch_noisy: int = 2,
#         in_ch_lr: int = 2,
#         out_channels: int = 2,
#         features: Sequence[int] = (64, 128, 256, 512),
#     ) -> None:
#         super().__init__()

#         in_channels = in_ch_noisy + in_ch_lr  # typ. 4

#         # ---------- Encoder (ein Pfad) ----------
#         self.encoder = nn.ModuleList()
#         self.pools   = nn.ModuleList()
#         ch = in_channels
#         for feat in features:
#             self.encoder.append(double_conv(ch, feat))
#             self.pools  .append(nn.MaxPool2d(2))
#             ch = feat

#         # ---------- Bottleneck (klassisch UNet) ----------
#         self.bottleneck = double_conv(features[-1], features[-1] * 2)

#         # ---------- Decoder ----------
#         self.upconvs   = nn.ModuleList()
#         self.dec_blocks= nn.ModuleList()
#         ch = features[-1] * 2
#         for feat in reversed(features):
#             self.upconvs.append(nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2))
#             # klassisch: concat(skip, up) → 2*feat
#             self.dec_blocks.append(double_conv(feat * 2, feat))
#             ch = feat

#         # ---------- Output ----------
#         self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#     # ---------------- Forward -----------------------------------------------
#     def forward(self, x_noisy: torch.Tensor, x_lr: torch.Tensor) -> torch.Tensor:
#         """
#         Beibehaltener Aufruf: (x_noisy, x_lr) → cat → UNet.
#         x_noisy, x_lr : (B, 2, H, W), gleiche räumliche Größe.
#         """
#         assert x_noisy.shape[2:] == x_lr.shape[2:], "Input-Shapes müssen übereinstimmen."
#         x = torch.cat([x_noisy, x_lr], dim=1)  # (B, 4, H, W)

#         B, C, H, W = x.shape
#         L      = len(self.pools)
#         factor = 1 << L  # 2**L

#         # ---------- symmetrisches Padding ----------
#         pad_h = (factor - H % factor) % factor
#         pad_w = (factor - W % factor) % factor
#         pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
#         pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
#         x_p = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

#         # ---------- Encoder ----------
#         skips = []
#         z = x_p
#         for enc, pool in zip(self.encoder, self.pools):
#             z = enc(z)
#             skips.append(z)
#             z = pool(z)

#         # ---------- Bottleneck ----------
#         z = self.bottleneck(z)

#         # ---------- Decoder ----------
#         for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
#             z = up(z)
#             if z.shape[-2:] != skip.shape[-2:]:
#                 z = F.pad(
#                     z,
#                     [0, skip.shape[-1] - z.shape[-1],
#                      0, skip.shape[-2] - z.shape[-2]]
#                 )
#             z = dec(torch.cat([skip, z], dim=1))

#         # ---------- Output & Cropping ----------
#         out = self.out_conv(z)
#         return out[..., pad_top: pad_top + H, pad_left: pad_left + W]




# ALT funktioniert aber
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

# ---------- 2×3×3-Conv-Block --------------------------------------------------

def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )

# ---------- Y-Net-Klasse (Concat) --------------------------------------------

class YNet2D(nn.Module):
    """Zweipfadiger 2-D-U-Net-Decoder mit Concat-Fusion (ohne Gates)."""

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

        # ---------- Encoder Low-Rank ----------
        self.enc_lr, self.pool_lr = nn.ModuleList(), nn.ModuleList()
        ch = in_ch_lr
        for feat in features:
            self.enc_lr.append(double_conv(ch, feat))
            self.pool_lr.append(nn.MaxPool2d(2))
            ch = feat

        # ---------- Bottleneck ----------
        # concat der beiden tiefsten Encoder-Features → 2*feat
        self.bottleneck = double_conv(features[-1] * 2, features[-1] * 2)

        # ---------- Decoder ----------
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch = features[-1] * 2
        for feat in reversed(features):
            # Upsampling auf die aktuelle Ebene
            self.upconvs.append(nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2))
            # Decoder-Block erwartet: concat(skip_noisy, skip_lr, up) = 3*feat
            self.dec_blocks.append(double_conv(3 * feat, feat))
            ch = feat

        # ---------- Output ----------
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ---------------- Forward -----------------------------------------------
    def forward(self, x_noisy: torch.Tensor, x_lr: torch.Tensor) -> torch.Tensor:
        """Vorwärtsdurchlauf mit Concat-Fusion der Skip-Features."""
        assert x_noisy.shape[2:] == x_lr.shape[2:], "Input-Shapes müssen übereinstimmen."
        B, _, H, W = x_noisy.shape
        L = len(self.pool_noisy)
        factor = 1 << L  # 2**L

        # ---------- symmetrisches Padding ----------
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        x_noisy_p = F.pad(x_noisy, [pad_left, pad_right, pad_top, pad_bottom])
        x_lr_p    = F.pad(x_lr,    [pad_left, pad_right, pad_top, pad_bottom])

        # ---------- Encoder ----------
        skips_noisy, skips_lr = [], []
        xa, xb = x_noisy_p, x_lr_p
        for encA, poolA, encB, poolB in zip(
            self.enc_noisy, self.pool_noisy, self.enc_lr, self.pool_lr
        ):
            ya = encA(xa); skips_noisy.append(ya); xa = poolA(ya)
            yb = encB(xb); skips_lr  .append(yb); xb = poolB(yb)

        # ---------- Bottleneck ----------
        x = self.bottleneck(torch.cat([xa, xb], dim=1))

        # ---------- Decoder (Concat) ----------
        for up, dec, s_noisy, s_lr in zip(
            self.upconvs, self.dec_blocks, reversed(skips_noisy), reversed(skips_lr)
        ):
            x = up(x)
            # ggf. Größe anpassen (Numerik/Padding-Schutz)
            if x.shape[-2:] != s_noisy.shape[-2:]:
                x = F.pad(
                    x,
                    [0, s_noisy.shape[-1] - x.shape[-1],
                     0, s_noisy.shape[-2] - x.shape[-2]]
                )
            # Concat der Skip-Features beider Pfade + upsampled Feature
            fuse = torch.cat([s_noisy, s_lr], dim=1)     # (B, 2*feat, H, W)
            x = dec(torch.cat([fuse, x], dim=1))         # (B, 3*feat, H, W)

        # ---------- Output & Cropping ----------
        out = self.out_conv(x)
        return out[..., pad_top: pad_top + H, pad_left: pad_left + W]





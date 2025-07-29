# models/unet2d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Hilfsbaustein: 2 × Conv(3×3) ----------
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )

# ---------- Minimal-U-Net 2D ----------
class UNet2D(nn.Module):
    """
    Schlanker 2-D-U-Net-Encoder/Decoder.
    • Für komplexe Daten: Standard = 2 Ein- und 2 Ausgabekanäle (Real + Imag)
    • Keine BatchNorm, kein Residual-Add, keine Regularisierung
    """

    def __init__(
        self,
        in_channels: int = 2,         # ← Real + Imag
        out_channels: int = 2,        # ← Real + Imag
        features: tuple = (64, 128, 256, 512),
    ):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.ModuleList()
        self.pools   = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoder.append(double_conv(ch, feat))
            self.pools  .append(nn.MaxPool2d(kernel_size=2))
            ch = feat

        # -------- Bottleneck --------
        self.bottleneck = double_conv(features[-1], features[-1] * 2)

        # -------- Decoder --------
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ch = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2))
            self.decoder.append(double_conv(ch, feat))
            ch = feat

        # -------- Output --------
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # -------- Forward --------
    def forward(self, x):
        B, C, H, W = x.shape
        L      = len(self.pools)          # z.B. 3 Ebenen
        factor = 1 << L                   # 2**L

        # 1) symmetrisch aufrunden
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        x_p = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        # 2) U-Net-Durchlauf
        identity = x
        x = x_p
        skips = []
        for enc, pool in zip(self.encoder, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.final_conv(x)

        # 3) symmetrisch zurückcroppen
        x = x[..., pad_top : pad_top + H, pad_left : pad_left + W]

        return x + identity



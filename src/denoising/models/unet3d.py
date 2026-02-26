# models/unet3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Hilfsbaustein: 2 × Conv(3×3×3) ----------
def double_conv3d(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )

# ---------- Minimal-U-Net 3D ----------
class UNet3D(nn.Module):
    """
    Schlanker 3-D-U-Net-Encoder/Decoder.
    • Eingabe- / Ausgabe-Kanäle wie 2-D-Version (Real + Imag = 2)
    • Keine BatchNorm, kein Residual-Add, keine Regularisierung
    """

    def __init__(
        self,
        in_channels: int = 2,          # Real + Imag
        out_channels: int = 2,
        features: tuple = (16, 32, 64, 128, 256),
    ):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.ModuleList()
        self.pools   = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoder.append(double_conv3d(ch, feat))
            self.pools  .append(nn.MaxPool3d(kernel_size=2))
            ch = feat

        # -------- Bottleneck --------
        self.bottleneck = double_conv3d(features[-1], features[-1] * 2)

        # -------- Decoder --------
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ch = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(ch, feat, kernel_size=2, stride=2))
            self.decoder.append(double_conv3d(ch, feat))
            ch = feat

        # -------- Output --------
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    # -------- Forward --------
    def forward(self, x):
        """
        x : (B, C, Z, H, W)
        """
        B, C, D, H, W = x.shape
        L      = len(self.pools)       # Anzahl Downsamplings
        factor = 1 << L                # 2**L

        # 1) symmetrisch auf Vielfache von 2**L padden
        pad_d = (factor - D % factor) % factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor

        pad_front, pad_back   = pad_d // 2, pad_d - pad_d // 2
        pad_top,   pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left,  pad_right  = pad_w // 2, pad_w - pad_w // 2

        # Reihenfolge für F.pad: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x_p = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back])

        # 2) Encoder
        skips = []
        x = x_p
        for enc, pool in zip(self.encoder, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # 3) Bottleneck
        x = self.bottleneck(x)

        # 4) Decoder + Skip-Verbindungen
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = up(x)
            # ggf. durch Padding entstehende Offsets angleichen
            if x.shape[2:] != skip.shape[2:]:
                z_diff = skip.size(2) - x.size(2)
                y_diff = skip.size(3) - x.size(3)
                x_diff = skip.size(4) - x.size(4)
                x = F.pad(x, [
                    x_diff // 2, x_diff - x_diff // 2,
                    y_diff // 2, y_diff - y_diff // 2,
                    z_diff // 2, z_diff - z_diff // 2
                ])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        # 5) Output-Layer
        x = self.final_conv(x)

        # 6) symmetrisch zurückcroppen
        x = x[:,
              :,
              pad_front : pad_front + D,
              pad_top   : pad_top   + H,
              pad_left  : pad_left  + W]

        return x






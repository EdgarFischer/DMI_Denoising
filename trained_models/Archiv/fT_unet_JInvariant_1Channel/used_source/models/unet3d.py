# models/unet3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 2× Conv(3×3×3) ----------
def double_conv3d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )

# ---------- Minimal-U-Net 3-D ----------
class UNet3D(nn.Module):
    """
    Schlanker 3-D-U-Net-Encoder/Decoder
    • Zwei Ein-/Ausgabekanäle (Real + Imag)
    • Kein BatchNorm, kein Residual-Add, keine Regularisierung
    """

    def __init__(
        self,
        in_channels : int = 2,
        out_channels: int = 2,
        features    : tuple = (64, 128, 256, 512),
    ):
        super().__init__()

        # Encoder -----------------------------------------------------------
        self.encoder = nn.ModuleList()
        self.pools   = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoder.append(double_conv3d(ch, feat))
            self.pools  .append(nn.MaxPool3d(kernel_size=2))
            ch = feat

        # Bottleneck --------------------------------------------------------
        self.bottleneck = double_conv3d(features[-1], features[-1] * 2)

        # Decoder -----------------------------------------------------------
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ch = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(ch, feat, 2, 2))
            self.decoder.append(double_conv3d(ch, feat))
            ch = feat

        # Output-Schicht ----------------------------------------------------
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, D, H, W)   →  Rückgabe: (B, C_out, D, H, W)
        """
        B, _, D, H, W = x.shape
        L      = len(self.pools)
        factor = 1 << L                         # 2**L

        # Symmetrisches Padding -------------------------------------------
        pad_d = (factor - D % factor) % factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pf, pb = pad_d // 2, pad_d - pad_d // 2
        pt, pbm = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2

        x = F.pad(x, [pl, pr, pt, pbm, pf, pb])  # Reihenfolge: W,H,D

        # Encoder -----------------------------------------------------------
        skips = []
        for enc, pool in zip(self.encoder, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck + Decoder ---------------------------------------------
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        # Output-Conv & Crop ----------------------------------------------
        x = self.final_conv(x)
        x = x[..., pf:pf+D, pt:pt+H, pl:pl+W]

        return x





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


# ---------- interner Kern‑U‑Net‑Block (1 Pfad) ----------
class _UNetCore(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: tuple):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        self.pools   = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoder.append(double_conv(ch, feat))
            self.pools  .append(nn.MaxPool2d(kernel_size=2))
            ch = feat

        # Bottleneck
        self.bottleneck = double_conv(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ch = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, feat, kernel_size=2, stride=2))
            self.decoder.append(double_conv(ch, feat))
            ch = feat

        # Output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        L      = len(self.pools)
        factor = 1 << L                        # 2**L

        # symmetrisches Padding
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pad_top,    pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left,   pad_right  = pad_w // 2, pad_w - pad_w // 2
        x_p = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        # U‑Net‑Durchlauf
        skips = []
        for enc, pool in zip(self.encoder, self.pools):
            x_p = enc(x_p)
            skips.append(x_p)
            x_p = pool(x_p)

        x_p = self.bottleneck(x_p)

        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x_p = up(x_p)
            x_p = torch.cat([skip, x_p], dim=1)
            x_p = dec(x_p)

        x_p = self.final_conv(x_p)

        # symmetrisch zurückcroppen
        x_p = x_p[..., pad_top : pad_top + H, pad_left : pad_left + W]
        return x_p


# ---------- Dual‑Pfad‑U‑Net ----------
class UNet2D(nn.Module):
    """
    Gleiche API wie zuvor, aber:
    • Pfad A verarbeitet die maskierten Spektren unverändert.
    • Pfad B führt FFT entlang der 1. 2‑D‑Achse (H), läuft durch ein zweites U‑Net,
      anschließend iFFT, bevor beide Pfade fusioniert werden.
    """

    def __init__(
        self,
        in_channels: int = 2,         # Real + Imag
        out_channels: int = 2,        # Real + Imag
        features: tuple  = (64, 128, 256, 512),
    ):
        super().__init__()

        # Zwei identische U‑Net‑Ker­ne
        self.spec_unet = _UNetCore(in_channels, out_channels, features)
        self.fid_unet  = _UNetCore(in_channels, out_channels, features)

        # 1×1‑Fusions‑Conv: 2·out_channels → out_channels
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)


    # ---------- Forward ----------
    def forward(self, x):
        """
        x: Tensor [B, 2, H, W]  (Real/Imag)
        H = Spektral‑Achse, W = Zeit
        """
        B, C, H, W = x.shape

        # ---------------- Pfad A: direktes Spektrum ----------------
        spec_out = self.spec_unet(x)        # [B, 2, H, W]

        # ---------------- Pfad B: FFT ➀ → U‑Net → iFFT ➁ ----------------
        # ➀ FFT (Spektrum → FID):
        real, imag = x[:, 0], x[:, 1]       # [B, H, W]
        x_complex  = torch.complex(real, imag)
        # FFT entlang der "ersten" 2‑D‑Achse (H)
        x_fft_cplx = torch.fft.fft(x_complex, dim=1)
        # zurück in 2‑Kanäle
        x_fft = torch.stack([x_fft_cplx.real, x_fft_cplx.imag], dim=1)  # [B, 2, H, W]

        # U‑Net in FID‑Domäne
        fid_out = self.fid_unet(x_fft)      # [B, 2, H, W]

        # ➁ iFFT (FID → Spektrum) vor Fusion
        fid_real, fid_imag = fid_out[:, 0], fid_out[:, 1]
        fid_cplx  = torch.complex(fid_real, fid_imag)
        fid_ifft  = torch.fft.ifft(fid_cplx, dim=1)
        fid_ifft  = torch.stack([fid_ifft.real, fid_ifft.imag], dim=1)  # [B, 2, H, W]

        # ---------------- Fusion & Rückgabe ----------------
        merged = torch.cat([spec_out, fid_ifft], dim=1)  # [B, 4, H, W]
        out    = self.fusion_conv(merged)                # [B, 2, H, W]
        return out





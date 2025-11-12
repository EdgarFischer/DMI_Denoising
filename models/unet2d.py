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

# ---------- Einzelner U-Net-Durchlauf (Core) ----------
class _UNetCore(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: tuple):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.features     = features

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
        factor = 1 << L  # 2**L

        # symm. aufrunden
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        x_p = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        # U-Net
        skips = []
        z = x_p
        for enc, pool in zip(self.encoder, self.pools):
            z = enc(z)
            skips.append(z)
            z = pool(z)
        z = self.bottleneck(z)
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            z = up(z)
            z = torch.cat([skip, z], dim=1)
            z = dec(z)
        z = self.final_conv(z)

        # zurückcroppen
        z = z[..., pad_top : pad_top + H, pad_left : pad_left + W]
        return z

# ---------- Stackbarer U-Net 2D ----------
class UNet2D(nn.Module):
    """
    Schlanker 2-D-U-Net mit optionalem Stacking.
    - Intern: U-Net Concatenation (klassisch)
    - Zwischen Stufen: additive Residuals mit lernbarem alpha (Init 0.0)
    """

    def __init__(
        self,
        in_channels: int = 2,          # Real + Imag
        out_channels: int = 2,         # Real + Imag
        features: tuple = (64, 128, 256, 512),
        layers: int = 1,               # Anzahl gestapelter U-Nets
        learnable_scale: bool = True,  # alpha lernbar?
        init_scale: float = 0.0,       # Initialwert für alpha
    ):
        super().__init__()
        assert layers >= 1, "layers muss >= 1 sein"

        self.layers = layers

        # Erste Stufe: in_channels -> out_channels
        self.stage0 = _UNetCore(in_channels, out_channels, features)

        # Folge-Stufen: out_channels -> out_channels
        self.stages = nn.ModuleList([
            _UNetCore(out_channels, out_channels, features)
            for _ in range(layers - 1)
        ])

        # Residual-Skalierungen (eine pro Stufe)
        alphas = []
        # alpha für Stage 0
        if learnable_scale:
            self.alpha0 = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.register_buffer("alpha0", torch.tensor(init_scale, dtype=torch.float32))
        # alpha für Stage 1..N-1
        for _ in range(layers - 1):
            if learnable_scale:
                alphas.append(nn.Parameter(torch.tensor(init_scale, dtype=torch.float32)))
            else:
                # Buffer als nicht-trainierbar
                buf = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32), requires_grad=False)
                alphas.append(buf)
        self.alphas = nn.ParameterList(alphas) if learnable_scale else nn.ParameterList(alphas)

        # Optionaler 1×1-Projektor für den ersten Residual-Add (falls C_in != C_out)
        self.proj0 = None
        if in_channels != out_channels:
            self.proj0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Stage 0 (evtl. Proj + Residual)
        y0 = self.stage0(x)
        if self.layers == 1:
            return y0  # identisches Verhalten zu deinem Original (kein äußerer Residual)

        x_cur = (self.proj0(x) if self.proj0 is not None else x) + self.alpha0 * y0

        # Weitere Stufen mit Residual-Add
        for i, stage in enumerate(self.stages):
            yi = stage(x_cur)
            alpha = self.alphas[i] if len(self.alphas) > 0 else 0.0
            x_cur = x_cur + alpha * yi

        return x_cur









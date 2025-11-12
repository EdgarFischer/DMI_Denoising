# models/unet3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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

        # -------- Zweites U-Net (eigene Gewichte) --------
        # Deepcopies der ersten Module, in neue ModuleLists gepackt
        self.encoder2    = nn.ModuleList(copy.deepcopy([m for m in self.encoder]))
        self.pools2      = nn.ModuleList(copy.deepcopy([m for m in self.pools]))
        self.upconvs2    = nn.ModuleList(copy.deepcopy([m for m in self.upconvs]))
        self.decoder2    = nn.ModuleList(copy.deepcopy([m for m in self.decoder]))
        self.bottleneck2 = copy.deepcopy(self.bottleneck)
        self.final_conv2 = copy.deepcopy(self.final_conv)

    # -------- Forward --------
    def forward(self, x):
        """
        x : (B, C, Z, H, W)
        """
        # =========================
        # PASS 1
        # =========================
        x_in1 = x  # für Residual-Verbindung

        B, C, D, H, W = x.shape
        L      = len(self.pools)       # Anzahl Downsamplings
        factor = 1 << L                # 2**L

        # 1) symmetrisch auf Vielfache von 2**L padden
        pad_d = (factor - D % factor) % factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor

        pad_front1, pad_back1   = pad_d // 2, pad_d - pad_d // 2
        pad_top1,   pad_bottom1 = pad_h // 2, pad_h - pad_h // 2
        pad_left1,  pad_right1  = pad_w // 2, pad_w - pad_w // 2

        # Reihenfolge für F.pad: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x_p1 = F.pad(x, [pad_left1, pad_right1, pad_top1, pad_bottom1, pad_front1, pad_back1])

        # 2) Encoder
        skips1 = []
        x1 = x_p1
        for enc, pool in zip(self.encoder, self.pools):
            x1 = enc(x1)
            skips1.append(x1)
            x1 = pool(x1)

        # 3) Bottleneck
        x1 = self.bottleneck(x1)

        # 4) Decoder + Skip-Verbindungen
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips1)):
            x1 = up(x1)
            # ggf. durch Padding entstehende Offsets angleichen
            if x1.shape[2:] != skip.shape[2:]:
                z_diff = skip.size(2) - x1.size(2)
                y_diff = skip.size(3) - x1.size(3)
                x_diff = skip.size(4) - x1.size(4)
                x1 = F.pad(x1, [
                    x_diff // 2, x_diff - x_diff // 2,
                    y_diff // 2, y_diff - y_diff // 2,
                    z_diff // 2, z_diff - z_diff // 2
                ])
            x1 = torch.cat([skip, x1], dim=1)
            x1 = dec(x1)

        # 5) Output-Layer
        x1 = self.final_conv(x1)

        # 6) symmetrisch zurückcroppen
        x1 = x1[:,
                :,
                pad_front1 : pad_front1 + D,
                pad_top1   : pad_top1   + H,
                pad_left1  : pad_left1  + W]
        
        # Residual des 1. Passes
        x1 = x1 + x_in1

        # =========================
        # PASS 2 (eigene Gewichte)
        # =========================
        x_in2 = x1  # für Residual-Verbindung

        B, C, D, H, W = x1.shape
        L      = len(self.pools2)
        factor = 1 << L

        pad_d = (factor - D % factor) % factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor

        pad_front2, pad_back2   = pad_d // 2, pad_d - pad_d // 2
        pad_top2,   pad_bottom2 = pad_h // 2, pad_h - pad_h // 2
        pad_left2,  pad_right2  = pad_w // 2, pad_w - pad_w // 2

        x_p2 = F.pad(x1, [pad_left2, pad_right2, pad_top2, pad_bottom2, pad_front2, pad_back2])

        skips2 = []
        x2 = x_p2
        for enc, pool in zip(self.encoder2, self.pools2):
            x2 = enc(x2)
            skips2.append(x2)
            x2 = pool(x2)

        x2 = self.bottleneck2(x2)

        for up, dec, skip in zip(self.upconvs2, self.decoder2, reversed(skips2)):
            x2 = up(x2)
            if x2.shape[2:] != skip.shape[2:]:
                z_diff = skip.size(2) - x2.size(2)
                y_diff = skip.size(3) - x2.size(3)
                x_diff = skip.size(4) - x2.size(4)
                x2 = F.pad(x2, [
                    x_diff // 2, x_diff - x_diff // 2,
                    y_diff // 2, y_diff - y_diff // 2,
                    z_diff // 2, z_diff - z_diff // 2
                ])
            x2 = torch.cat([skip, x2], dim=1)
            x2 = dec(x2)

        x2 = self.final_conv2(x2)

        x2 = x2[:,
                :,
                pad_front2 : pad_front2 + D,
                pad_top2   : pad_top2   + H,
                pad_left2  : pad_left2  + W]

        # Residual des 2. Passes
        #x2 = x2 + x_in2

        return x2





# basis_unet2d_coeff.py  ▸  KOMPLETTER CODE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------------------------------------------------
# 1) Komplex-Hilfsfunktionen  (bleiben unverändert)
# -------------------------------------------------------------------------
def torch_project(x_ri: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    B, _, f, T = x_ri.shape
    x_c = (x_ri[:,0] + 1j*x_ri[:,1]).permute(0,2,1)   # (B,T,f)
    c   = torch.matmul(x_c, V.conj())                 # (B,T,r)
    c   = c.permute(0,2,1)                            # (B,r,T)
    return torch.stack([c.real, c.imag], dim=1)       # (B,2,r,T)

def torch_reconstruct(c_ri: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    c_c = (c_ri[:,0] + 1j*c_ri[:,1]).permute(0,2,1)   # (B,T,r)
    x   = torch.matmul(c_c, V.T).permute(0,2,1)       # (B,f,T)
    return torch.stack([x.real, x.imag], dim=1)       # (B,2,f,T)

# -------------------------------------------------------------------------
# 2) Encoder, der bis exakt 1×1 schrumpft
# -------------------------------------------------------------------------
class BottleEncoder(nn.Module):
    """
    Komprimiert (B,2,f,T) bis exakt (B,2r,1,1):
      • solange T>1  → MaxPool2d(2,2)
      • danach        MaxPool2d((2,1)) nur auf f
    """
    def __init__(self, r: int, base_ch: int = 64):
        super().__init__()
        layers, in_ch = [], 2
        f_layers = [base_ch, base_ch*2, base_ch*4, base_ch*8]  # beliebig erweiterbar

        for out_ch in f_layers:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True)
            ]
            # Dummy-Layer, wird später überschrieben
            layers.append(nn.Identity())
            in_ch = out_ch

        self.blocks = nn.ModuleList(layers)
        self.final  = nn.Sequential(
            nn.Conv2d(in_ch, 2*r, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):
        _, _, f, T = x.shape
        idx = 0
        for i in range(0, len(self.blocks), 4):
            conv, bn, act, pool = self.blocks[i:i+4]

            x = conv(x); x = bn(x); x = act(x)

            # ▼ Pooling-Strategie
            if T > 1:
                pool_layer = nn.MaxPool2d((2,2))
                T //= 2; f //= 2
            elif f > 1:
                pool_layer = nn.MaxPool2d((2,1))
                f //= 2
            else:
                pool_layer = nn.Identity()

            self.blocks[i+3] = pool_layer  # ersetze Identity durch tatsächliches Layer
            x = pool_layer(x)

            if f == 1 and T == 1:
                break  # kein weiteres Downsampling nötig

            idx += 1

        return self.final(x)  # (B,2r,1,1)

# -------------------------------------------------------------------------
# 3) Haupt-Modell  (API bleibt wie bisher)
# -------------------------------------------------------------------------
class BasisCoeffUNet2D(nn.Module):
    """
    * Input : (B,2,f,T)   – Real/Imag-Spektren
    * Output: (B,2,f,T)   – Rekonstruktion im Low-Rank-Raum
    """
    def __init__(self, basis: np.ndarray,
                 base_ch: int = 64,
                 num_enc_blocks: int = 3):
        super().__init__()
        f, r = basis.shape
        self.f, self.r = f, r

        # Basis als komplexer Buffer (wird auto-device-gemanagt)
        self.register_buffer('V', torch.from_numpy(basis).to(torch.complex64))

        # Encoder bis (B,2r,1,1)
        self.encoder = BottleEncoder(r, base_ch=base_ch)

    # ------------------------------------------------------------------
    def forward(self, x):                # x: (B,2,f,T)
        B, _, _, T = x.shape

        # 1) Encoder → Bottleneck  (B,2r,1,1)
        c_bottle = self.encoder(x)       #           ^
        c_bottle = c_bottle.view(B, 2*self.r)        # (B,2r)

        # 2) als (B,2,r,T) interpretieren  (broadcast über Zeit)
        c_ri = c_bottle.view(B, 2, self.r, 1).expand(-1, -1, -1, T)

        # 3) Rekonstruktion via fixer Basis
        x_rec = torch_reconstruct(c_ri, self.V)      # (B,2,f,T)
        return x_rec

















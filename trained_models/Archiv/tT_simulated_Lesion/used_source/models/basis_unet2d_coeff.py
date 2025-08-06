# basis_unet2d_coeff.py
import torch
import torch.nn as nn
import numpy as np
from models.unet2d import UNet2D      # deine schlanke 2-D-U-Net-Klasse

class BasisCoeffUNet2D(nn.Module):
    """
    • Input  : (B, 2,  f, T)   – Real/Imag, 96×8 z. B.
    • Output : (B, 2,  f, T)   – denoised Spektren
    Interner Weg:
        U-Net  → 2*r Koeff-Maps (B,2*r,1,T)
        fixierte 1×1-Conv (Basis) → (B,2*f,1,T) → reshape → (B,2,f,T)
    """

    def __init__(self, basis: np.ndarray, features=(32,64,128,256,512)):
        """
        basis : (f, r) komplex, orthonormal  – z. B. V_r aus noisy_data
        """
        super().__init__()
        f, r = basis.shape
        self.f, self.r = f, r

        # 1) U-Net erzeugt Re+Im Koeffizienten
        self.coeff_net = UNet2D(
            in_channels = 2,
            out_channels = 2*r,          # Re/Im für r Basis-Vektoren
            features = features,
        )

        # 2) fixierte 1×1-Conv implementiert Basis-Multiplikation
        self.basis_layer = nn.Conv2d(
            in_channels = 2*r,
            out_channels = 2*f,
            kernel_size = 1,
            bias = False,
        )
        self._init_basis_weights(basis)
        self.basis_layer.requires_grad_(False)    # **nicht trainierbar**

    # ---------- Basis-Gewichte als Real-Im Matrix ----------
    def _init_basis_weights(self, V: np.ndarray):
        """
        baut Gewichtsmatrix W (2*f, 2*r) gemäß
            out_R = Σ (a_R * v_R − a_I * v_I)
            out_I = Σ (a_R * v_I + a_I * v_R)
        """
        f, r = V.shape
        W = np.zeros((2*f, 2*r), dtype=np.float32)
        for k in range(f):
            for i in range(r):
                vR, vI = V[k, i].real, V[k, i].imag
                # Re-Ausgang
                W[2*k    , 2*i    ] =  vR
                W[2*k    , 2*i + 1] = -vI
                # Im-Ausgang
                W[2*k + 1, 2*i    ] =  vI
                W[2*k + 1, 2*i + 1] =  vR
        weight = torch.from_numpy(W[:, :, None, None])   # (C_out,C_in,1,1)
        self.basis_layer.weight.data.copy_(weight)

    # ---------- Forward ----------
    def forward(self, x):
        """
        x: (B, 2, f, T)
        """
        B, C, f, T = x.shape
        # U-Net over (f,T) liefert Koeff-Maps
        coeff = self.coeff_net(x)           # (B, 2*r, f, T)
        # Mittelung über f → r-Koeffizienten pro Zeit­punkt
        coeff = coeff.mean(dim=2, keepdim=True)         # (B, 2*r, 1, T)

        # Fixierte Basis-Layer → (B, 2*f, 1, T)
        spec  = self.basis_layer(coeff)                 # (B, 2*f, 1, T)

        # in (B,2,f,T) umformen
        spec  = spec.view(B, 2, f, T)

        return spec





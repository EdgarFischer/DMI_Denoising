import numpy as np

def image_grid_fov1(N):
    # x in (-0.5, 0.5) with spacing 1/N
    x = (np.arange(N) - N/2) / N
    X, Y = np.meshgrid(x, x, indexing="xy")
    return np.stack([X.ravel(), Y.ravel()], axis=1)  # (N^2,2)

def backward_operator_from_k(N, kx, ky, w=None, alpha="trace"):
    """
    General 2D backward operator on the fixed Cartesian image grid (FOV=1):

        x = B y
        B[n,m] = alpha * w[m] * exp(i 2π k_m · x_n)

    kx,ky: (M,) in cycles per FOV (here FOV=1) -> numerically same as cycles per unit length
    w:     quadrature / density-comp weights (M,)
    alpha: "trace" => scale so trace(BB^H) = N^2 (comparable average noise power)
    """
    r = image_grid_fov1(N)                        # (N^2,2)
    k = np.stack([kx, ky], axis=1)                # (M,2)

    phase = np.exp(2j*np.pi * (k @ r.T)).T        # (N^2, M)

    if w is None:
        w = np.ones_like(kx, dtype=float)

    B = phase * w[None, :]

    if alpha == "trace":
        diag = np.sum(np.abs(B)**2, axis=1)
        alpha_val = np.sqrt((N**2) / np.sum(diag))
        B = alpha_val * B

    return B

def cartesian_k_grid_fov1(N):
    # k in cycles per FOV=1 on the DFT grid: k = m, with x = n/N
    m = np.arange(N) - N//2
    KX, KY = np.meshgrid(m, m, indexing="xy")
    return KX.ravel(), KY.ravel()

import numpy as np

import numpy as np

def crt_2d_rings_const_theta(
    N: int,
    oversamp_factor: float = 1.0,
    N_rings: int | None = None,
    include_dc: bool = False,
    dk_target: float = 1.0,
    kmax: float | None = None,
    verbose: bool = True,
):
    """
    CRT rings with CONSTANT angular spacing for all rings.

    Nyquist condition (checked at outer ring):
        r_max * Δθ <= dk_target
    """

    if N_rings is None:
        N_rings = N

    if kmax is None:
        kmax = N / 2.0   # physical k-units for FOV=1

    dr = kmax / N_rings
    r_max = (N_rings - 0.5) * dr

    # worst-case (outer ring) Nyquist condition
    N_theta_min = int(np.ceil((2.0 * np.pi * r_max) / dk_target))
    N_theta = int(np.ceil(oversamp_factor * N_theta_min))
    N_theta = max(N_theta, 1)

    dtheta = 2.0 * np.pi / N_theta
    thetas = np.arange(N_theta) * dtheta

    if verbose:
        print("CRT configuration:")
        print(f"  N_rings           = {N_rings}")
        print(f"  N_theta (per ring)= {N_theta}")
        print(f"  kmax              = {kmax:.3f}")
        print(f"  dr                = {dr:.3f}")
        print(f"  r_max             = {r_max:.3f}")
        print(f"  Δθ                = {dtheta:.4f} rad")
        print(f"  r_max * Δθ        = {r_max * dtheta:.3f}  (<= dk_target={dk_target})")
        print(f"  Nyquist OK        = {r_max * dtheta <= dk_target}")

    kxs, kys, wts = [], [], []

    if include_dc:
        kxs.append(np.array([0.0]))
        kys.append(np.array([0.0]))
        wts.append(np.array([np.pi * (dr/2.0)**2], dtype=float))

    for j in range(N_rings):
        r = (j + 0.5) * dr
        kx = r * np.cos(thetas)
        ky = r * np.sin(thetas)
        w  = r * dr * dtheta

        kxs.append(kx)
        kys.append(ky)
        wts.append(np.full_like(kx, w, dtype=float))

    kx = np.concatenate(kxs)
    ky = np.concatenate(kys)
    w  = np.concatenate(wts)

    if verbose:
        print(f"  Total k-space points = {kx.size}")

    return kx, ky, w

def cov_to_corr(C):
    d = np.sqrt(np.real(np.diag(C)))
    return C / (d[:, None] * d[None, :])


import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("../src"))
from denoising.eval.NoiseCorrelations import *

def plot_corr_to_center(C, N, title="", vmax=None):
    center_idx = (N // 2) * N + (N // 2)  # row-major, center voxel
    corr = np.abs(C[center_idx, :]).reshape(N, N)

    if vmax is None:
        vmax = np.max(corr)

    im = plt.imshow(corr, cmap="plasma", vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=r"$|\rho|$")
    plt.title(title)
    plt.axis("off")

def plot_covariances(C_cart, C_crt):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axs[0].imshow(np.real(C_cart), cmap="viridis")
    axs[0].set_title("Cartesian Covariance")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(np.real(C_crt), cmap="viridis")
    axs[1].set_title("CRT Covariance")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    plt.tight_layout()
    plt.show()
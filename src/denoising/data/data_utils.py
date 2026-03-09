import os
import numpy as np
from scipy.io import loadmat

def load_and_preprocess_data(
    folder_names: list,
    base_path: str,
    fourier_axes: list = None,
    normalization: bool = True
) -> np.ndarray:
    """
    Loads 'data.npy' from each folder under base_path and concatenates along last axis (D).
    Standard behavior:
      1) Normalize in the original (FID) domain to max(|.|)=1 per folder.
      2) Optionally apply FFT (+ fftshift) on specified axes.
    Returns:
      data: np.ndarray with shape (X, Y, Z, t/f, T, D)
    """

    arrays = []
    for fold in folder_names:

        npy_fn = os.path.join(base_path, fold, "data.npy")
        mat_fn = os.path.join(base_path, fold, "CombinedCSI.mat")

        if os.path.exists(npy_fn):
            arr = np.load(npy_fn)  # expected shape (X,Y,Z,t,T)

        elif os.path.exists(mat_fn):
            mat = loadmat(mat_fn)
            arr = mat["csi"]["Data"][0, 0]

        else:
            raise FileNotFoundError(
                f"No data file found in {os.path.join(base_path, fold)} "
                f"(expected data.npy or CombinedCSI.mat)"
            )

        if arr.ndim == 5:
            arr = arr[..., np.newaxis]  # -> (X,Y,Z,t,T,1)

        # 1) STANDARD: normalize in FID domain (before any FFT)
        if normalization:
            maxv = np.max(np.abs(arr))
            if maxv > 0:
                arr = arr / maxv

        # 2) FFT (optional)
        if fourier_axes:
            for ax in fourier_axes:
                arr = np.fft.fft(arr, axis=ax)
                arr = np.fft.fftshift(arr, axes=ax)

        arrays.append(arr.astype(np.complex64, copy=False))

    return np.concatenate(arrays, axis=-1)

def low_rank_5d(data, rank):
    """
    Computes a low-rank decomposition of a tensor with shape (22, 22, 21, 96, 8)
    using truncated SVD.

    Args:
        data (np.ndarray): Numpy array of shape (x, y, z, t, T).
        rank (int): The number of singular values to keep (final rank).

    Returns:
        np.ndarray: The reconstructed tensor with rank 'rank'.
    """

    # Unpack dimensions
    x, y, z, t, T = data.shape
    
    # Reshape the 5D tensor into a 2D matrix of shape (x*y*z, t*T)
    # Use 'F' (Fortran) order to match MATLAB's column-major ordering
    reshaped_matrix = data.reshape((x * y * z * T, t), order='F')
    
    # Perform economy-size SVD (similar to MATLAB's "svd(..., 'econ')")
    U, singular_values, Vh = np.linalg.svd(reshaped_matrix, full_matrices=False)
    
    # Truncate the singular values to the desired rank
    k = min(rank, len(singular_values))  # safeguard: rank cannot exceed # of singular values
    singular_values_truncated = np.zeros_like(singular_values)
    singular_values_truncated[:k] = singular_values[:k]
    
    # Form the diagonal matrix of truncated singular values
    S_truncated = np.diag(singular_values_truncated)
    
    # Reconstruct the matrix using the truncated SVD components
    reconstructed_matrix = U @ S_truncated @ Vh
    
    # Reshape back to the original 5D shape, again using 'F' order
    reconstructed_tensor = reconstructed_matrix.reshape((x, y, z, t, T), order='F')
    
    return reconstructed_tensor

def low_rank(data: np.ndarray, rank: int) -> np.ndarray:
    """
    Computes a low-rank decomposition of a tensor with shape
      • (x, y, z, t, T)  → direkt per SVD
      • (x, y, z, t, T, D) → wendet SVD separat auf jede D-Scheibe an

    Args:
        data (np.ndarray): Eingabe mit 5 oder 6 Dimensionen.
        rank (int): Anzahl der Singulärwerte.

    Returns:
        np.ndarray: Rekonstruiertes Array in Original-Shape.
    """
    if data.ndim == 5:
        # Einzelfall: direkt 5D
        return low_rank_5d(data, rank)

    elif data.ndim == 6:
        # 6D: Apply low_rank_5d für jede D-Scheibe
        x, y, z, t, T, D = data.shape
        rec = np.zeros_like(data)
        for d in range(D):
            rec[..., d] = low_rank_5d(data[..., d], rank)
        return rec

    else:
        raise ValueError(f"low_rank expects 5 or 6 dims, got {data.ndim}")

def load_noisy_and_lowrank_data(
    folder_names: list,
    base_path: str,
    fourier_axes: list = None,
    normalize: bool = True,
    rank: int = 8,                    # z. B. 10
):
    """
    Liefert ein Tuple (noisy, lowrank) mit derselben Shape.
    """
    noisy = load_and_preprocess_data(
        folder_names, base_path, fourier_axes, normalize
    )
    lowrank = low_rank(noisy.copy(), rank=rank)   # deine SVD-Funktion
    return noisy, lowrank

def build_basis(noisy: np.ndarray, rank: int):
    # noisy: (x, y, z, t, T)
    # bring t ans Ende, damit reshape((…, t)) t-dimension isoliert
    noisy2 = noisy.transpose(0, 1, 2, 4, 3)  # jetzt (x, y, z, T, t)
    x, y, z, T, t = noisy2.shape

    # flatten spatial+T zu Zeilen, Spektrum als Spalten
    M = noisy2.reshape((x * y * z * T, t), order='F')

    # SVD auf jeder Zeile = Spektrum
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    # Vh[:rank] sind die Top-r rechten Singularvektoren (in R^t)
    V_r = Vh[:rank].conj().T   # (t, rank)

    return V_r, S[:rank]

def project(data: np.ndarray, V_r: np.ndarray):
    # data: (x, y, z, t, T), V_r: (t, rank)
    data2 = data.transpose(0, 1, 2, 4, 3)     # (x,y,z,T,t)
    x, y, z, T, t = data2.shape

    M = data2.reshape((x*y*z*T, t), order='F')   # (N, t)
    C = M @ V_r                                 # (N, rank)
    R = C @ V_r.conj().T                        # (N, t)
    recon = R.reshape((x, y, z, T, t), order='F')\
               .transpose(0,1,2,4,3)            # zurück zu (x,y,z,t,T)

    return recon, C

def mse(a, b):
    return np.mean(np.abs(a - b)**2)
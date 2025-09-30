import os
import numpy as np

def load_and_preprocess_data(
    folder_names: list,
    base_path: str,
    fourier_axes: list = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Lädt jeweils 'data.npy' aus jedem Unterordner in base_path und
    stapelt sie entlang der letzten Achse (D). Führt optional FFT und
    Normierung durch.

    Args:
      folder_names: Liste von Ordnern unter base_path, z.B. ['P03','P04',...]
      base_path:    Pfad zu Deinem 'datasets'-Ordner
      fourier_axes: Liste von Achsen, auf denen np.fft.fft+fftshift angewandt werden soll
      normalize:    True → jede Teildatenmenge wird auf max(abs)=1 skaliert

    Returns:
      data: np.ndarray mit Shape (X, Y, Z, t, T, D)
    """
    arrays = []
    for fold in folder_names:
        fn = os.path.join(base_path, fold, 'data.npy')
        arr = np.load(fn)               # erwartet Shape (X,Y,Z,t,T)
        if arr.ndim == 5:
            arr = arr[..., np.newaxis]  # → (X,Y,Z,t,T,1)

        # 1) Normalisieren
        if normalize:
            maxv = np.max(np.abs(arr))
            if maxv > 0:
                arr = arr / maxv

        # 2) Fourieranalyse
        if fourier_axes:
            for ax in fourier_axes:
                # unge-shiftete FFT
                arr = np.fft.fft(arr, axis=ax)
                # zentrieren
                arr = np.fft.fftshift(arr, axes=ax)

        arrays.append(arr)

    # 3) Stapeln aller Runs → Shape (X,Y,Z,t,T,D)
    return np.concatenate(arrays, axis=-1)

def low_rank(data, rank):
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

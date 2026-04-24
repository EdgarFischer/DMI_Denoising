import os
import numpy as np
from scipy.io import loadmat
import logging 
from pathlib import Path
import numpy as np
import SimpleITK as sitk

def load_and_preprocess_data(
    folder_names: list,
    base_path: str,
    fourier_axes: list = None,
    normalization: bool = True,
    npy_name: str = "data.npy",
    mat_name: str = "CombinedCSI.mat",
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

        npy_fn = os.path.join(base_path, fold, npy_name)
        mat_fn = os.path.join(base_path, fold, mat_name)

        if os.path.exists(npy_fn):
            arr = np.load(npy_fn)  # expected shape (X,Y,Z,t,T)
            logging.info(f"[LOAD] {fold} (npy) loaded with shape {arr.shape}")

        elif os.path.exists(mat_fn):
            mat = loadmat(mat_fn)
            arr = mat["csi"]["Data"][0, 0]
            logging.info(f"[LOAD] {fold} (mat) loaded with shape {arr.shape}")

        else:
            raise FileNotFoundError(
                f"No data file found in {os.path.join(base_path, fold)} "
                f"(expected {npy_name} or {mat_name})"
            )

        if arr.ndim == 4:          # (X,Y,Z,t)
            arr = arr[..., np.newaxis, np.newaxis]  # -> (X,Y,Z,t,1,1)

        elif arr.ndim == 5:        # (X,Y,Z,t,T)
            arr = arr[..., np.newaxis]              # -> (X,Y,Z,t,T,1)

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

def load_dataset_list_from_folders(
    folder_names,
    base_path,
    fourier_axes=None,
    normalization=True,
    npy_name="data.npy",
    mat_name="CombinedCSI.mat",
):
    """
    Wrapper um load_and_preprocess_data, der statt eines gestackten Arrays
    eine Liste einzelner Datensätze zurückgibt.

    Returns
    -------
    dataset_list : list of np.ndarray
        Liste von Arrays mit Shape (X, Y, Z, t/f, T)
    """

    stacked = load_and_preprocess_data(
        folder_names=folder_names,
        base_path=base_path,
        fourier_axes=fourier_axes,
        normalization=normalization,
        npy_name=npy_name,
        mat_name=mat_name,
    )

    dataset_list = [stacked[..., i] for i in range(stacked.shape[-1])]
    return dataset_list

def load_dataset_list(
    subject_ids,
    method=None,
    base_dir="../datasets",
    suffix=None,
    fourier_axes=None,
    normalization=False,
    npy_name="data.npy",
    mat_name="CombinedCSI.mat",
):
    """
    Lädt mehrere Datensätze aus Ordnern.

    Parameters
    ----------
    subject_ids : list of str
    method : str or None
    base_dir : str
    suffix : str or None
        Optionaler Suffix (z. B. "normalized").
        Wenn None oder "" → kein Suffix wird verwendet.
    """

    folder_names = []

    for sid in subject_ids:
        parts = [sid]

        if suffix not in (None, ""):
            parts.append(suffix)

        if method is not None:
            parts.append(method)

        folder_name = "_".join(parts)
        folder_names.append(folder_name)

    return load_dataset_list_from_folders(
        folder_names=folder_names,
        base_path=base_dir,
        fourier_axes=fourier_axes,
        normalization=normalization,
        npy_name=npy_name,
        mat_name=mat_name,
    )

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

def apply_low_rank_5d_to_dataset_list(dataset_list, rank):
    """
    Wendet low_rank_5d auf eine Liste von 5D-Datensätzen an.

    Parameters
    ----------
    dataset_list : list of np.ndarray
        Liste von Arrays mit Shape (x, y, z, t, T)
    rank : int
        Zielrang für die truncierte SVD

    Returns
    -------
    lowrank_list : list of np.ndarray
        Liste der rekonstruierten 5D-Datensätze
    """

    lowrank_list = []

    for i, data in enumerate(dataset_list):
        if data.ndim != 5:
            raise ValueError(
                f"Datensatz an Index {i} hat ndim={data.ndim}, erwartet wird 5D"
            )

        print(f"Applying low-rank to dataset {i+1}/{len(dataset_list)} with shape {data.shape}")
        reconstructed = low_rank_5d(data, rank)
        lowrank_list.append(reconstructed)

    return lowrank_list

def load_sd_maps(folder):
    """
    Lädt SD/CRLB maps aus einem Ordner.
    Bevorzugt *_sd_map.mnc, fällt aber auf *_sd_map.nii.gz zurück.

    Parameters
    ----------
    folder : str or Path
        Ordner mit den SD/CRLB maps.

    Returns
    -------
    sd_maps : dict
        Dict der Form:
        {
            "Asp": np.ndarray,
            "GABA": np.ndarray,
            ...
        }
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    mnc_files = sorted(folder.glob("*_sd_map.mnc"))
    nii_files = sorted(folder.glob("*_sd_map.nii.gz"))

    if mnc_files:
        files = mnc_files
        suffix = "_sd_map.mnc"
        print(f"Using .mnc files from {folder}")
    elif nii_files:
        files = nii_files
        suffix = "_sd_map.nii.gz"
        print(f"Using .nii.gz files from {folder}")
    else:
        raise FileNotFoundError(
            f"No *_sd_map.mnc or *_sd_map.nii.gz files found in {folder}"
        )

    sd_maps = {}

    for f in files:
        metabolite = f.name.replace(suffix, "")
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)
        sd_maps[metabolite] = arr

    return sd_maps

def load_and_align_t1_to_mask(mag_path, mask_path):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    import numpy as np

    mag = resample_from_to(
        nib.load(mag_path),
        nib.load(mask_path),
        order=1
    ).get_fdata()

    mag = np.swapaxes(mag, 0, 1)[::-1, ::-1, :]
    return mag

def hamming_filter(arr, apply_dims, filter_width=100):
    """
    Apply an N‑D outer‑product Hamming filter along the specified axes.

    Parameters
    ----------
    arr : np.ndarray
        Input k‑space array. Wird hier direkt im k‑space gefiltert!
    apply_dims : sequence of int
        Achsen, entlang derer der Filter angewendet wird.
    filter_width : float, optional
        Prozentualer Anteil des Filters (100 = voller Hamming‑Window,
        50 = nur auf die äußeren 50 % angewendet).

    Returns
    -------
    out : np.ndarray
        Gefiltertes Array, gleiche Form wie `arr`.
    """
    out = arr.copy()
    for dim in apply_dims:
        n = arr.shape[dim]
        # Länge des 1D‑Hamming‑Fensters
        fw = int(np.ceil(filter_width/100 * n))
        # Erzeuge das 1D‑Fenster
        win = np.hamming(fw)
        # Links/rechts mit Einsen auffüllen, damit die Mitte unberührt bleibt
        pad_total = n - fw
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        full = np.concatenate([
            np.ones(pad_left, dtype=win.dtype),
            win,
            np.ones(pad_right, dtype=win.dtype)
        ])
        # Für Broadcast auf die korrekte Dimension bringen
        shape = [1] * arr.ndim
        shape[dim] = n
        full = full.reshape(shape)
        # multiplikative Kombination (OuterProduct über alle dims)
        out = out * full
    return out

def add_kspace_noise(
    data,
    noise_sigma,
    save_path=None,
    seed=None,
    apply_hamming=False,
    filter_width=100
):
    """
    Add complex Gaussian noise in k-space and return noisy image-space data.

    Parameters
    ----------
    data : np.ndarray
        Input data (x, y, z, t, T) or similar.
    noise_sigma : float
        Standard deviation of complex Gaussian noise (in k-space).
    save_path : str or Path, optional
        If given, saves result as data.npy.
    seed : int, optional
        Random seed for reproducibility.
    apply_hamming : bool
        Whether to apply Hamming filter in k-space.
    filter_width : int
        Width for Hamming filter (only used if apply_hamming=True).

    Returns
    -------
    noise_data : np.ndarray
        Noisy data in image space (same shape as input).
    """

 #Explanation: For in vivo noise strenght Anna used a noise sigma of 11.72*10⁵, however, she rescaled her simulated data by multipliying it with
# # 1.2*10⁴, then added noise with sigma 11.72*10⁵. I never adopted this rescaling and used the original scale. To get the same noise sigma in my scale I hence 
# # have to use 11.72*10⁵/(1.2*10⁴)~ 97.666666

    rng = np.random.default_rng(seed=10_000 + seed)

    # --- normalization factor ---
    MAX = np.abs(np.max(data))

    # --- go to k-space ---
    k_space = np.fft.fftshift(
        np.fft.fftn(data, axes=(0, 1, 2)),
        axes=(0, 1, 2)
    )

    # --- add complex Gaussian noise ---
    noise = (
        rng.standard_normal(k_space.shape)
        + 1j * rng.standard_normal(k_space.shape)
    ) * noise_sigma

    k_space_noisy = k_space + noise

    # --- optional Hamming filter ---
    if apply_hamming:
        k_space_noisy = hamming_filter(
            k_space_noisy,
            apply_dims=[0, 1, 2],
            filter_width=filter_width
        )

    # --- back to image space ---
    noise_data = np.fft.ifftn(
        np.fft.ifftshift(k_space_noisy, axes=(0, 1, 2)),
        axes=(0, 1, 2)
    ) / MAX

    # --- optional save ---
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path/"data.npy", noise_data)

    return noise_data

def apply_linewidth_broadening(data, delta_lw_hz, sw=476.0, save_path=None):
    """
    Apply uniform Lorentzian line broadening to a 5D array by multiplying
    the FID dimension with exp(-pi * delta_lw_hz * t).

    Assumes data shape: (x, y, z, t, T), i.e. FID axis = -2.

    Parameters
    ----------
    data : np.ndarray
        Input 5D complex-valued array.
    delta_lw_hz : float
        Additional linewidth broadening in Hz.
    sw : float, default=476.0
        Spectral bandwidth in Hz.
    save_path : str or Path, optional
        If given, saves the broadened array as .npy.

    Returns
    -------
    data_broad : np.ndarray
        Broadened data array with same shape as input.
    """
    dt = 1.0 / sw
    t = np.arange(data.shape[-2]) * dt

    broadening = np.exp(-np.pi * delta_lw_hz * t)  # shape: (Nt,)
    data_broad = data * broadening[None, None, None, :, None]

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, data_broad)

    return data_broad

def load_snr_data(base_dir, subjects, methods):
    base_dir = Path(base_dir)
    snr_data = {}

    for subject in subjects:
        snr_data[subject] = {}

        for method in methods:
            folder = f"{subject}_{method}"
            file_name = f"SNR_{folder}.npy"
            file_path = base_dir / folder / file_name

            snr_data[subject][method] = np.load(file_path)

    return snr_data


# def low_rank(data: np.ndarray, rank: int) -> np.ndarray:
#     """
#     Computes a low-rank decomposition of a tensor with shape
#       • (x, y, z, t, T)  → direkt per SVD
#       • (x, y, z, t, T, D) → wendet SVD separat auf jede D-Scheibe an

#     Args:
#         data (np.ndarray): Eingabe mit 5 oder 6 Dimensionen.
#         rank (int): Anzahl der Singulärwerte.

#     Returns:
#         np.ndarray: Rekonstruiertes Array in Original-Shape.
#     """
#     if data.ndim == 5:
#         # Einzelfall: direkt 5D
#         return low_rank_5d(data, rank)

#     elif data.ndim == 6:
#         # 6D: Apply low_rank_5d für jede D-Scheibe
#         x, y, z, t, T, D = data.shape
#         rec = np.zeros_like(data)
#         for d in range(D):
#             rec[..., d] = low_rank_5d(data[..., d], rank)
#         return rec

#     else:
#         raise ValueError(f"low_rank expects 5 or 6 dims, got {data.ndim}")

# def load_noisy_and_lowrank_data(
#     folder_names: list,
#     base_path: str,
#     fourier_axes: list = None,
#     normalize: bool = True,
#     rank: int = 8,                    # z. B. 10
# ):
#     """
#     Liefert ein Tuple (noisy, lowrank) mit derselben Shape.
#     """
#     noisy = load_and_preprocess_data(
#         folder_names, base_path, fourier_axes, normalize
#     )
#     lowrank = low_rank(noisy.copy(), rank=rank)   # deine SVD-Funktion
#     return noisy, lowrank

# def build_basis(noisy: np.ndarray, rank: int):
#     # noisy: (x, y, z, t, T)
#     # bring t ans Ende, damit reshape((…, t)) t-dimension isoliert
#     noisy2 = noisy.transpose(0, 1, 2, 4, 3)  # jetzt (x, y, z, T, t)
#     x, y, z, T, t = noisy2.shape

#     # flatten spatial+T zu Zeilen, Spektrum als Spalten
#     M = noisy2.reshape((x * y * z * T, t), order='F')

#     # SVD auf jeder Zeile = Spektrum
#     U, S, Vh = np.linalg.svd(M, full_matrices=False)

#     # Vh[:rank] sind die Top-r rechten Singularvektoren (in R^t)
#     V_r = Vh[:rank].conj().T   # (t, rank)

#     return V_r, S[:rank]

# def project(data: np.ndarray, V_r: np.ndarray):
#     # data: (x, y, z, t, T), V_r: (t, rank)
#     data2 = data.transpose(0, 1, 2, 4, 3)     # (x,y,z,T,t)
#     x, y, z, T, t = data2.shape

#     M = data2.reshape((x*y*z*T, t), order='F')   # (N, t)
#     C = M @ V_r                                 # (N, rank)
#     R = C @ V_r.conj().T                        # (N, t)
#     recon = R.reshape((x, y, z, T, t), order='F')\
#                .transpose(0,1,2,4,3)            # zurück zu (x,y,z,t,T)

#     return recon, C

# def mse(a, b):
#     return np.mean(np.abs(a - b)**2)
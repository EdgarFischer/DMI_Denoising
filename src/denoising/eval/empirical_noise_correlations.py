import numpy as np
import sys
import os

sys.path.append(os.path.abspath("../../src"))
from denoising.data.data_utils import *

def estimate_noise_masks(dataset_list, percentile=5, axes=(3,4)):
    """
    Schätzt Noise-Masken für mehrere Datensätze.

    Idee:
    Es wird über FID- und Repetitions-Achsen gemittelt, sodass unkorreliertes
    Rauschen gegen seinen Erwartungswert (~0) konvergiert und sich weitgehend
    auslöscht. Konsistentes Signal (z. B. metabolische Peaks) bleibt hingegen
    erhalten. Dadurch entstehen Volumen mit stark unterdrücktem Noise, in denen
    verbleibende Intensität primär auf echte Signalanteile zurückzuführen ist.

    Anschließend wird ein Perzentil-Threshold (z. B. unterste 5%) auf diese
    gemittelten Werte angewendet, um Voxel mit minimaler verbleibender
    Signalenergie als Noise-Voxel zu identifizieren.

    Parameters
    ----------
    dataset_list : list of np.ndarray
        Liste von Datenarrays, jeweils Shape (x, y, z, ...)
    percentile : float
        Prozentualer Threshold für Noise-Voxel
    axes : tuple
        Achsen über die gemittelt wird (z. B. (3,4) für t und T)

    Returns
    -------
    masks : list of np.ndarray
        Liste von Bool-Masken (Shape: (x, y, z))
    """

    masks = []

    for data in dataset_list:
        averaged = np.abs(np.mean(data, axis=axes))
        thr = np.percentile(averaged, percentile)
        mask_noise = averaged <= thr
        masks.append(mask_noise)

    return masks

def estimate_1d_acf_from_masked_voxels(data, mask_noise, subtract_mean=True, normalize=True):
    """
    Schätzt 1D-Autokorrelationen entlang der nicht-räumlichen Achsen,
    nachdem eine 3D-räumliche Noise-Maske angewendet wurde.

    Der Mittelwert wird aus dem gesamten Ensemble geschätzt (nicht pro Sequenz).

    Parameters
    ----------
    data : np.ndarray
        Array mit Shape (x, y, z, ...), z.B. (x, y, z, t, T)
    mask_noise : np.ndarray
        Bool-Array mit Shape (x, y, z)
    subtract_mean : bool
        Ob Ensemble-Mittelwert pro Achse abgezogen wird
    normalize : bool
        Ob ACF auf Lag 0 = 1 normiert wird

    Returns
    -------
    acfs : list of np.ndarray
        Liste der gemittelten 1D-ACFs für jede Nicht-Raum-Achse.
        Bei data.shape = (x,y,z,t,T):
            acfs[0] -> ACF entlang t
            acfs[1] -> ACF entlang T
    """

    if data.ndim < 4:
        raise ValueError("data muss mindestens 4D sein: (x, y, z, ...)")
    if mask_noise.shape != data.shape[:3]:
        raise ValueError("mask_noise muss Shape data.shape[:3] haben")

    # Noise-Voxel extrahieren
    noise_data = data[mask_noise]   # shape (N_voxels, ...)

    remaining_shape = noise_data.shape[1:]
    n_remaining = len(remaining_shape)

    acfs = []

    for ax in range(n_remaining):

        L = remaining_shape[ax]

        # Zielachse ans Ende
        arr = np.moveaxis(noise_data, ax + 1, -1)

        # Ensemble-Mittelwert über alle anderen Dimensionen
        if subtract_mean:
            mean_axis = arr.mean(axis=tuple(range(arr.ndim - 1)), keepdims=True)
            arr = arr - mean_axis

        # alles flatten -> viele Realisationen
        arr = arr.reshape(-1, L)

        acf = np.zeros(L, dtype=np.float64)

        for lag in range(L):
            num = np.sum(arr[:, :L-lag] * np.conj(arr[:, lag:]))
            acf[lag] = np.real(num)

        if normalize and acf[0] > 0:
            acf = acf / acf[0]

        acfs.append(acf)

    return acfs

def estimate_1d_acfs_for_dataset_list(
    dataset_list,
    mask_list,
    subtract_mean=True,
    normalize=True
):
    """
    Wendet estimate_1d_acf_from_masked_voxels auf mehrere Datensätze an.

    Parameters
    ----------
    dataset_list : list of np.ndarray
        Liste von Datenarrays, jeweils Shape (x, y, z, ...)
    mask_list : list of np.ndarray
        Liste von Bool-Masken, jeweils Shape (x, y, z)
    subtract_mean : bool
        Ob Ensemble-Mittelwert pro Achse abgezogen wird
    normalize : bool
        Ob ACF auf Lag 0 = 1 normiert wird

    Returns
    -------
    acfs_list : list of list of np.ndarray
        Für jeden Datensatz eine Liste von ACFs entlang der nicht-räumlichen Achsen.

        Beispiel bei data.shape = (x, y, z, t, T):
            acfs_list[i][0] -> ACF entlang t für Datensatz i
            acfs_list[i][1] -> ACF entlang T für Datensatz i
    """

    if len(dataset_list) != len(mask_list):
        raise ValueError("dataset_list und mask_list müssen gleich lang sein")

    acfs_list = []

    for data, mask_noise in zip(dataset_list, mask_list):
        acfs = estimate_1d_acf_from_masked_voxels(
            data,
            mask_noise,
            subtract_mean=subtract_mean,
            normalize=normalize
        )
        acfs_list.append(acfs)

    return acfs_list

def estimate_spatial_correlations(
    data,
    mask_noise,
    max_lag=None,
    subtract_mean=True,
    normalize=True
):
    """
    Schätzt räumliche Korrelationen entlang x, y und z
    in noise-dominierten Voxeln.

    Es werden nur Voxelpaare berücksichtigt, bei denen beide Voxel
    in der Noise-Maske liegen. Alle nicht-räumlichen Dimensionen
    werden als Ensemble-Dimensionen verwendet.

    Parameters
    ----------
    data : np.ndarray
        Array mit Shape (x, y, z, ...), z. B. (x, y, z, t, T)
    mask_noise : np.ndarray
        Bool-Array mit Shape (x, y, z)
    max_lag : int or None
        Maximaler räumlicher Lag. Falls None, wird für jede Achse
        die vollständige ACF bis zum maximal möglichen Lag berechnet.
    subtract_mean : bool
        Ob der Ensemble-Mittelwert vor der Korrelationsschätzung
        abgezogen werden soll.
    normalize : bool
        Ob auf Lag 0 = 1 normiert werden soll.

    Returns
    -------
    spatial_corrs : dict
        Dictionary mit Einträgen
            {
                "x": np.ndarray,
                "y": np.ndarray,
                "z": np.ndarray
            }
        Die Länge der Arrays hängt von der jeweiligen Achse ab.
    """

    if data.ndim < 4:
        raise ValueError("data muss mindestens 4D sein: (x, y, z, ...)")
    if mask_noise.shape != data.shape[:3]:
        raise ValueError("mask_noise muss Shape data.shape[:3] haben")
    if max_lag is not None and max_lag < 0:
        raise ValueError("max_lag muss >= 0 sein oder None")

    spatial_corrs = {}
    axis_map = {"x": 0, "y": 1, "z": 2}

    for axis_name, ax in axis_map.items():
        axis_len = data.shape[ax]

        if max_lag is None:
            current_max_lag = axis_len - 1
        else:
            current_max_lag = min(max_lag, axis_len - 1)

        corr_values = np.full(current_max_lag + 1, np.nan, dtype=np.float64)

        for lag in range(current_max_lag + 1):
            slicer1 = [slice(None)] * data.ndim
            slicer2 = [slice(None)] * data.ndim

            if lag == 0:
                slicer1[ax] = slice(None)
                slicer2[ax] = slice(None)
            else:
                slicer1[ax] = slice(0, -lag)
                slicer2[ax] = slice(lag, None)

            data1 = data[tuple(slicer1)]
            data2 = data[tuple(slicer2)]

            mask1 = mask_noise[tuple(slicer1[:3])]
            mask2 = mask_noise[tuple(slicer2[:3])]
            valid_pairs = mask1 & mask2

            if not np.any(valid_pairs):
                continue

            arr1 = data1[valid_pairs].reshape(-1)
            arr2 = data2[valid_pairs].reshape(-1)

            if subtract_mean:
                arr1 = arr1 - np.mean(arr1)
                arr2 = arr2 - np.mean(arr2)

            num = np.sum(arr1 * np.conj(arr2))
            num = np.real(num)

            if normalize:
                den1 = np.sum(np.abs(arr1) ** 2)
                den2 = np.sum(np.abs(arr2) ** 2)

                if den1 > 0 and den2 > 0:
                    corr = num / np.sqrt(den1 * den2)
                else:
                    corr = np.nan
            else:
                corr = num

            corr_values[lag] = corr

        if normalize and np.isfinite(corr_values[0]) and corr_values[0] != 0:
            corr_values = corr_values / corr_values[0]

        spatial_corrs[axis_name] = corr_values

    return spatial_corrs

def estimate_spatial_correlations_for_dataset_list(
    dataset_list,
    mask_list,
    max_lag=None,
    subtract_mean=True,
    normalize=True
):
    """
    Wendet estimate_spatial_correlations auf mehrere Datensätze an.

    Parameters
    ----------
    dataset_list : list of np.ndarray
        Liste von Datenarrays, jeweils Shape (x, y, z, ...)
    mask_list : list of np.ndarray
        Liste von Bool-Masken, jeweils Shape (x, y, z)
    max_lag : int
        Maximaler räumlicher Lag
    subtract_mean : bool
        Ob der Ensemble-Mittelwert vor der Korrelationsschätzung
        abgezogen werden soll
    normalize : bool
        Ob auf Lag 0 = 1 normiert werden soll

    Returns
    -------
    spatial_corrs_list : list of dict
        Liste von Dictionaries der Form
            {
                "x": np.ndarray der Länge max_lag+1,
                "y": np.ndarray der Länge max_lag+1,
                "z": np.ndarray der Länge max_lag+1
            }
    """

    if len(dataset_list) != len(mask_list):
        raise ValueError("dataset_list und mask_list müssen gleich lang sein")

    spatial_corrs_list = []

    for i, (data, mask_noise) in enumerate(zip(dataset_list, mask_list)):
        spatial_corrs = estimate_spatial_correlations(
            data=data,
            mask_noise=mask_noise,
            max_lag=max_lag,
            subtract_mean=subtract_mean,
            normalize=normalize
        )
        spatial_corrs_list.append(spatial_corrs)

    return spatial_corrs_list

def compute_acf_mean_std(acfs_list):
    """
    Berechnet Mittelwert und Standardabweichung der ACFs über mehrere Datensätze.

    Parameters
    ----------
    acfs_list : list of list of np.ndarray
        Ausgabe von estimate_1d_acfs_for_dataset_list.
        Struktur:
            acfs_list[i][j] = ACF für Datensatz i entlang Achse j

    Returns
    -------
    mean_acfs : list of np.ndarray
        Mittelwert pro Achse
    std_acfs : list of np.ndarray
        Standardabweichung pro Achse
    """

    n_axes = len(acfs_list[0])

    mean_acfs = []
    std_acfs = []

    for ax in range(n_axes):
        # Stacke alle Subjects für diese Achse
        acfs_ax = np.stack([acfs[ax] for acfs in acfs_list], axis=0)

        mean_acfs.append(np.mean(acfs_ax, axis=0))
        std_acfs.append(np.std(acfs_ax, axis=0))

    return mean_acfs, std_acfs

def compute_spatial_acf_mean_std(spatial_corrs_list):
    """
    Berechnet Mittelwert und Standardabweichung der räumlichen ACFs
    über mehrere Datensätze.

    Parameters
    ----------
    spatial_corrs_list : list of dict
        Ausgabe von estimate_spatial_correlations_for_dataset_list.
        Struktur:
            spatial_corrs_list[i]["x"] = Array für Datensatz i entlang x

    Returns
    -------
    mean_corrs : dict
        Mittelwert pro Achse
    std_corrs : dict
        Standardabweichung pro Achse
    """

    axes = ["x", "y", "z"]

    mean_corrs = {}
    std_corrs = {}

    for ax in axes:
        # alle Subjects für diese Achse sammeln
        arrs = [corrs[ax] for corrs in spatial_corrs_list]

        # ggf. unterschiedliche Längen → auf minimale Länge kürzen
        min_len = min(len(a) for a in arrs)
        arrs = [a[:min_len] for a in arrs]

        stacked = np.stack(arrs, axis=0)

        mean_corrs[ax] = np.mean(stacked, axis=0)
        std_corrs[ax] = np.std(stacked, axis=0)

    return mean_corrs, std_corrs

def _load_or_build_method_dataset(subject_ids, method_cfg, suffix="normalized", base_dir=None):
    """
    Returns a dataset list for one method.
    """
    mtype = method_cfg["type"]

    if mtype == "raw":
        kwargs = {"suffix": suffix}
        if base_dir is not None:
            kwargs["base_dir"] = base_dir
        return load_dataset_list(subject_ids, **kwargs)

    if mtype == "precomputed":
        kwargs = {
            "suffix": suffix,
            "method": method_cfg["method"],
        }
        if base_dir is not None:
            kwargs["base_dir"] = base_dir
        return load_dataset_list(subject_ids, **kwargs)

    if mtype == "callable":
        kwargs = {"suffix": suffix}
        if base_dir is not None:
            kwargs["base_dir"] = base_dir
        raw_list = load_dataset_list(subject_ids, **kwargs)

        fn = method_cfg["fn"]
        fn_kwargs = method_cfg.get("kwargs", {})
        return fn(raw_list, **fn_kwargs)

    raise ValueError(f"Unknown method type: {mtype}")


def _axis_name_to_index(axis_name):
    mapping = {"t": 0, "T": 1}
    if axis_name not in mapping:
        raise ValueError(f"Unsupported extra axis '{axis_name}'. Supported: {list(mapping.keys())}")
    return mapping[axis_name]


def run_noise_analysis_pipeline(
    subject_ids,
    methods,
    extra_axes=("t", "T"),
    mask_source="method",
    percentile=5,
    suffix="normalized",
    base_dir=None,
    compute_spatial=True,
):
    """
    Thin orchestration wrapper around existing functions.

    Returns
    -------
    results : dict
        {
            "datasets_by_method": ...,
            "masks_by_method": ...,
            "acf_stats_by_method": ...,
            "spatial_stats_by_method": ...,
            "extra_axes": ...,
            "subject_ids": ...,
        }
    """
    datasets_by_method = {}
    masks_by_method = {}
    acf_stats_by_method = {}
    spatial_stats_by_method = {}

    raw_list = None
    if mask_source == "raw":
        kwargs = {"suffix": suffix}
        if base_dir is not None:
            kwargs["base_dir"] = base_dir
        raw_list = load_dataset_list(subject_ids, **kwargs)

    for method_name, method_cfg in methods.items():
        data_list = _load_or_build_method_dataset(
            subject_ids=subject_ids,
            method_cfg=method_cfg,
            suffix=suffix,
            base_dir=base_dir,
        )
        datasets_by_method[method_name] = data_list

        if mask_source == "raw":
            mask_list = estimate_noise_masks(raw_list, percentile=percentile)
        elif mask_source == "method":
            mask_list = estimate_noise_masks(data_list, percentile=percentile)
        else:
            raise ValueError("mask_source must be 'raw' or 'method'")

        masks_by_method[method_name] = mask_list

        acfs = estimate_1d_acfs_for_dataset_list(data_list, mask_list)
        mean_acf, std_acf = compute_acf_mean_std(acfs)
        acf_stats_by_method[method_name] = (mean_acf, std_acf)

        if compute_spatial:
            spatial_corrs = estimate_spatial_correlations_for_dataset_list(data_list, mask_list)
            mean_spatial, std_spatial = compute_spatial_acf_mean_std(spatial_corrs)
            spatial_stats_by_method[method_name] = (mean_spatial, std_spatial)

    axis_indices = [_axis_name_to_index(ax) for ax in extra_axes]

    return {
        "subject_ids": subject_ids,
        "extra_axes": list(extra_axes),
        "axis_indices": axis_indices,
        "datasets_by_method": datasets_by_method,
        "masks_by_method": masks_by_method,
        "acf_stats_by_method": acf_stats_by_method,
        "spatial_stats_by_method": spatial_stats_by_method,
    }
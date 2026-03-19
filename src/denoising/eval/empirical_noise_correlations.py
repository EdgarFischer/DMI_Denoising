import numpy as np
import sys
import os

sys.path.append(os.path.abspath("../../src"))
from denoising.data.data_utils import *

def estimate_noise_masks(dataset_list, percentile=5, axes=(3, 4)):
    """
    Schätzt Noise-Masken für mehrere Datensätze.

    Idee:
    Es wird über die angegebenen Achsen gemittelt, sodass unkorreliertes
    Rauschen gegen seinen Erwartungswert (~0) konvergiert und sich weitgehend
    auslöscht. Konsistentes Signal bleibt hingegen erhalten.

    Parameters
    ----------
    dataset_list : list of np.ndarray
        Liste von Datenarrays, jeweils Shape (x, y, z, ...)
    percentile : float
        Prozentualer Threshold für Noise-Voxel
    axes : tuple
        Achsen über die gemittelt wird, z. B. (3,4) für t und T

    Returns
    -------
    masks : list of np.ndarray
        Liste von Bool-Masken mit derselben Shape wie die jeweiligen Datenarrays
    """

    masks = []

    for data in dataset_list:
        valid_axes = tuple(ax for ax in axes if ax < data.ndim)

        if len(valid_axes) == 0:
            raise ValueError(
                f"Keine gültigen Achsen in axes={axes} für data.ndim={data.ndim}"
            )

        averaged = np.abs(np.mean(data, axis=valid_axes))
        thr = np.percentile(averaged, percentile)
        mask_noise = averaged <= thr   # z. B. shape (x, y, z)

        # auf volle Datenshape erweitern
        n_extra_dims = data.ndim - mask_noise.ndim
        mask_noise = mask_noise.reshape(mask_noise.shape + (1,) * n_extra_dims)
        mask_noise = np.broadcast_to(mask_noise, data.shape)

        masks.append(mask_noise)

    return masks

def estimate_fid_noise_masks(dataset_list, percentile=30, fid_axis=3):
    """
    Schätzt FID-Noise-Masken für mehrere Datensätze.

    Idee:
    Es wird über alle Achsen außer der FID-Achse gemittelt.
    Dadurch entsteht ein 1D-Profil entlang der FID-Achse.
    FID-Punkte mit kleiner mittlerer Amplitude werden als noise-dominiert
    betrachtet.

    Parameters
    ----------
    dataset_list : list of np.ndarray
        Liste von Datenarrays, z. B. Shape (x, y, z, t) oder (x, y, z, t, T)
    percentile : float
        Prozentualer Threshold für Noise-FID-Punkte.
        Beispiel: 30 bedeutet, dass die 30% FID-Punkte mit kleinster
        mittlerer Amplitude als Noise markiert werden.
    fid_axis : int
        Index der FID-Achse, standardmäßig 3

    Returns
    -------
    masks : list of np.ndarray
        Liste von Bool-Masken mit derselben Shape wie die jeweiligen Datenarrays
    """
    masks = []

    for data in dataset_list:
        if fid_axis >= data.ndim:
            raise ValueError(
                f"fid_axis={fid_axis} ist ungültig für data.ndim={data.ndim}"
            )

        mean_axes = tuple(ax for ax in range(data.ndim) if ax != fid_axis)

        averaged = np.abs(np.mean(data, axis=mean_axes))   # shape (t,)
        thr = np.percentile(averaged, percentile)
        mask_noise = averaged <= thr                       # shape (t,)

        # auf volle Datenshape erweitern
        shape = [1] * data.ndim
        shape[fid_axis] = data.shape[fid_axis]
        mask_noise = mask_noise.reshape(shape)
        mask_noise = np.broadcast_to(mask_noise, data.shape)

        masks.append(mask_noise)

    return masks

def estimate_fid_window_masks(
    dataset_list,
    fid_axis=3,
    start_fraction=0.3,
    end_fraction=0.9,
):
    """
    Erzeugt zusammenhängende FID-Fenster-Masken.

    Parameters
    ----------
    dataset_list : list of np.ndarray
    fid_axis : int
        Index der FID-Achse
    start_fraction : float
        Start des Fensters relativ zur FID-Länge, z. B. 0.3
    end_fraction : float
        Ende des Fensters relativ zur FID-Länge, z. B. 0.9

    Returns
    -------
    masks : list of np.ndarray
        Bool-Masken mit derselben Shape wie data
    """
    if not (0 <= start_fraction < end_fraction <= 1):
        raise ValueError("Require 0 <= start_fraction < end_fraction <= 1")

    masks = []

    for data in dataset_list:
        if fid_axis >= data.ndim:
            raise ValueError(
                f"fid_axis={fid_axis} ist ungültig für data.ndim={data.ndim}"
            )

        L = data.shape[fid_axis]
        start_idx = int(np.floor(start_fraction * L))
        end_idx = int(np.ceil(end_fraction * L))

        mask_1d = np.zeros(L, dtype=bool)
        mask_1d[start_idx:end_idx] = True

        shape = [1] * data.ndim
        shape[fid_axis] = L
        mask = mask_1d.reshape(shape)
        mask = np.broadcast_to(mask, data.shape)

        masks.append(mask)

    return masks

def estimate_axis_correlations(
    data,
    mask_noise,
    axes=None,
    max_lag=None,
    subtract_mean=True,
    normalize=True,
):
    """
    Schätzt Autokorrelationen entlang beliebiger Achsen unter Verwendung
    einer voll-dimensionalen Bool-Maske.

    Für jeden Lag werden nur Paare berücksichtigt, bei denen beide Samples
    laut Maske gültig sind.

    Parameters
    ----------
    data : np.ndarray
        Datenarray, z. B. Shape (x, y, z, t, T) oder (x, y, z, t)
    mask_noise : np.ndarray
        Bool-Array mit derselben Shape wie data
    axes : iterable of int or None
        Achsen, entlang derer die Korrelation berechnet werden soll.
        Falls None, werden alle Achsen verwendet.
    max_lag : int, dict or None
        - None: volle Länge jeder Achse
        - int: gleicher max_lag für alle Achsen
        - dict: eigener max_lag pro Achse, z. B. {0: 5, 3: 20}
    subtract_mean : bool
        Ob vor der Korrelationsschätzung der Mittelwert der gültigen Samples
        pro Lag abgezogen werden soll.
    normalize : bool
        Ob auf Lag 0 = 1 normiert werden soll.

    Returns
    -------
    corrs : dict
        Dictionary: corrs[axis] = np.ndarray der Korrelationswerte
    """

    if data.shape != mask_noise.shape:
        raise ValueError("mask_noise muss dieselbe Shape wie data haben")

    if data.ndim < 1:
        raise ValueError("data muss mindestens 1D sein")

    if axes is None:
        axes = tuple(range(data.ndim))

    corrs = {}

    for ax in axes:
        axis_len = data.shape[ax]

        if max_lag is None:
            current_max_lag = axis_len - 1
        elif isinstance(max_lag, int):
            if max_lag < 0:
                raise ValueError("max_lag muss >= 0 sein")
            current_max_lag = min(max_lag, axis_len - 1)
        elif isinstance(max_lag, dict):
            lag_val = max_lag.get(ax, axis_len - 1)
            if lag_val < 0:
                raise ValueError("max_lag pro Achse muss >= 0 sein")
            current_max_lag = min(lag_val, axis_len - 1)
        else:
            raise ValueError("max_lag muss None, int oder dict sein")

        corr_values = np.full(current_max_lag + 1, np.nan, dtype=np.float64)

        for lag in range(current_max_lag + 1):
            slicer1 = [slice(None)] * data.ndim
            slicer2 = [slice(None)] * data.ndim

            if lag > 0:
                slicer1[ax] = slice(0, -lag)
                slicer2[ax] = slice(lag, None)

            data1 = data[tuple(slicer1)]
            data2 = data[tuple(slicer2)]

            mask1 = mask_noise[tuple(slicer1)]
            mask2 = mask_noise[tuple(slicer2)]

            valid_pairs = mask1 & mask2

            if not np.any(valid_pairs):
                continue

            arr1 = data1[valid_pairs].reshape(-1)
            arr2 = data2[valid_pairs].reshape(-1)

            if subtract_mean:
                arr1 = arr1 - np.mean(arr1)
                arr2 = arr2 - np.mean(arr2)

            n_pairs = arr1.size
            if n_pairs == 0:
                continue

            cov = np.sum(arr1 * np.conj(arr2)) / n_pairs
            corr_values[lag] = np.real(cov)

        if normalize and np.isfinite(corr_values[0]) and corr_values[0] > 0:
            corr_values = corr_values / corr_values[0]

        corrs[ax] = corr_values

    return corrs

def estimate_axis_correlations_for_dataset_list(
    dataset_list,
    mask_list,
    axes=None,
    max_lag=None,
    subtract_mean=True,
    normalize=True
):
    """
    Wendet estimate_axis_correlations auf mehrere Datensätze an.
    """

    if len(dataset_list) != len(mask_list):
        raise ValueError("dataset_list und mask_list müssen gleich lang sein")

    corrs_list = []

    for data, mask_noise in zip(dataset_list, mask_list):
        corrs = estimate_axis_correlations(
            data=data,
            mask_noise=mask_noise,
            axes=axes,
            max_lag=max_lag,
            subtract_mean=subtract_mean,
            normalize=normalize
        )
        corrs_list.append(corrs)

    return corrs_list

def compute_correlation_mean_std(corrs_list):
    """
    Berechnet Mittelwert und Standardabweichung der Korrelationen
    über mehrere Datensätze.

    Parameters
    ----------
    corrs_list : list of dict
        Liste von Dictionaries.
        Beispiel:
            corrs_list[i][ax] = Korrelationsarray für Datensatz i entlang Achse ax

        z. B.
            corrs_list[i][0]   -> x-Achse
            corrs_list[i][3]   -> t-Achse
        oder
            corrs_list[i]["x"] -> x-Achse
            corrs_list[i]["t"] -> t-Achse

    Returns
    -------
    mean_corrs : dict
        Mittelwert pro Achse
    std_corrs : dict
        Standardabweichung pro Achse
    """

    if len(corrs_list) == 0:
        raise ValueError("corrs_list darf nicht leer sein")

    axes = corrs_list[0].keys()

    mean_corrs = {}
    std_corrs = {}

    for ax in axes:
        arrs = [corrs[ax] for corrs in corrs_list]

        # falls unterschiedliche Längen vorkommen:
        min_len = min(len(a) for a in arrs)
        arrs = [a[:min_len] for a in arrs]

        stacked = np.stack(arrs, axis=0)

        mean_corrs[ax] = np.mean(stacked, axis=0)
        std_corrs[ax] = np.std(stacked, axis=0)

    return mean_corrs, std_corrs

def _build_load_kwargs(suffix="normalized", base_dir=None, method=None):
    kwargs = {}

    if suffix not in (None, ""):
        kwargs["suffix"] = suffix

    if base_dir is not None:
        kwargs["base_dir"] = base_dir

    if method is not None:
        kwargs["method"] = method

    return kwargs


def _load_or_build_method_dataset(subject_ids, method_cfg, suffix="normalized", base_dir=None):
    """
    Returns a dataset list for one method.
    """
    mtype = method_cfg["type"]

    if mtype == "raw":
        kwargs = _build_load_kwargs(
            suffix=suffix,
            base_dir=base_dir,
        )
        return load_dataset_list(subject_ids, **kwargs)

    if mtype == "precomputed":
        kwargs = _build_load_kwargs(
            suffix=suffix,
            base_dir=base_dir,
            method=method_cfg["method"],
        )
        return load_dataset_list(subject_ids, **kwargs)

    if mtype == "callable":
        kwargs = _build_load_kwargs(
            suffix=suffix,
            base_dir=base_dir,
        )
        raw_list = load_dataset_list(subject_ids, **kwargs)

        fn = method_cfg["fn"]
        fn_kwargs = method_cfg.get("kwargs", {})
        return fn(raw_list, **fn_kwargs)

    raise ValueError(f"Unknown method type: {mtype}")


def _axis_name_to_index(axis_name):
    mapping = {"x": 0, "y": 1, "z": 2, "t": 3, "T": 4}
    if axis_name not in mapping:
        raise ValueError(
            f"Unsupported axis '{axis_name}'. Supported: {list(mapping.keys())}"
        )
    return mapping[axis_name]


def run_noise_analysis_pipeline(
    subject_ids,
    methods,
    extra_axes=("t", "T"),
    mask_source="method",
    mask_type="spatial",
    percentile=5,
    fid_window=(0.3, 0.9),
    suffix="normalized",
    base_dir=None,
    compute_spatial=True,
    spatial_max_lag=None,
    extra_max_lag=None,
    subtract_mean=True,
    normalize=True,
):
    """
    Thin orchestration wrapper around the unified correlation functions.

    Parameters
    ----------
    subject_ids : list of str
    methods : dict
    extra_axes : tuple of str
        z. B. ("t", "T")
    mask_source : str
        "raw" oder "method"
    mask_type : str
        "spatial", "fid", oder "fid_window"
    percentile : float
        Wird für "spatial" und "fid" verwendet
    fid_window : tuple of float
        Nur für mask_type="fid_window", z. B. (0.3, 0.9)
    suffix : str
    base_dir : str or None
    compute_spatial : bool
    spatial_max_lag : int or None
    extra_max_lag : int, dict or None
    subtract_mean : bool
    normalize : bool

    Returns
    -------
    results : dict
        {
            "datasets_by_method": ...,
            "masks_by_method": ...,
            "acf_stats_by_method": ...,
            "spatial_stats_by_method": ...,
            "extra_axes": ...,
            "axis_indices": ...,
            "subject_ids": ...,
            "mask_type": ...,
            "mask_source": ...,
            "percentile": ...,
            "fid_window": ...,
        }
    """
    datasets_by_method = {}
    masks_by_method = {}
    acf_stats_by_method = {}
    spatial_stats_by_method = {}

    raw_list = None
    if mask_source == "raw":
        kwargs = _build_load_kwargs(
            suffix=suffix,
            base_dir=base_dir,
        )
        raw_list = load_dataset_list(subject_ids, **kwargs)

    axis_indices = [_axis_name_to_index(ax) for ax in extra_axes]

    for method_name, method_cfg in methods.items():
        data_list = _load_or_build_method_dataset(
            subject_ids=subject_ids,
            method_cfg=method_cfg,
            suffix=suffix,
            base_dir=base_dir,
        )
        datasets_by_method[method_name] = data_list

        if mask_source == "raw":
            mask_data_list = raw_list
        elif mask_source == "method":
            mask_data_list = data_list
        else:
            raise ValueError("mask_source must be 'raw' or 'method'")

        if mask_type == "spatial":
            mask_list = estimate_noise_masks(
                mask_data_list,
                percentile=percentile,
            )

        elif mask_type == "fid":
            mask_list = estimate_fid_noise_masks(
                mask_data_list,
                percentile=percentile,
            )

        elif mask_type == "fid_window":
            start_fraction, end_fraction = fid_window
            mask_list = estimate_fid_window_masks(
                mask_data_list,
                fid_axis=3,
                start_fraction=start_fraction,
                end_fraction=end_fraction,
            )

        else:
            raise ValueError("mask_type must be 'spatial', 'fid', or 'fid_window'")

        masks_by_method[method_name] = mask_list

        acfs = estimate_axis_correlations_for_dataset_list(
            dataset_list=data_list,
            mask_list=mask_list,
            axes=axis_indices,
            max_lag=extra_max_lag,
            subtract_mean=subtract_mean,
            normalize=normalize,
        )
        mean_acf, std_acf = compute_correlation_mean_std(acfs)
        acf_stats_by_method[method_name] = (mean_acf, std_acf)

        if compute_spatial:
            spatial_corrs = estimate_axis_correlations_for_dataset_list(
                dataset_list=data_list,
                mask_list=mask_list,
                axes=(0, 1, 2),
                max_lag=spatial_max_lag,
                subtract_mean=subtract_mean,
                normalize=normalize,
            )
            mean_spatial, std_spatial = compute_correlation_mean_std(spatial_corrs)
            spatial_stats_by_method[method_name] = (mean_spatial, std_spatial)

    return {
        "subject_ids": subject_ids,
        "extra_axes": list(extra_axes),
        "axis_indices": axis_indices,
        "datasets_by_method": datasets_by_method,
        "masks_by_method": masks_by_method,
        "acf_stats_by_method": acf_stats_by_method,
        "spatial_stats_by_method": spatial_stats_by_method,
        "mask_type": mask_type,
        "mask_source": mask_source,
        "percentile": percentile,
        "fid_window": fid_window,
    }
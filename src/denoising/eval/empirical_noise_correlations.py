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


def squeeze_trailing_singleton_dims(dataset_list, min_ndim=4):
    """
    Remove trailing singleton dimensions from each dataset, but keep at least
    `min_ndim` dimensions.
    """
    squeezed = []

    for data in dataset_list:
        arr = data
        while arr.ndim > min_ndim and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        squeezed.append(arr)

    return squeezed


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
        data_list = load_dataset_list(subject_ids, **kwargs)
        return squeeze_trailing_singleton_dims(data_list, min_ndim=4)

    if mtype == "precomputed":
        kwargs = _build_load_kwargs(
            suffix=suffix,
            base_dir=base_dir,
            method=method_cfg["method"],
        )
        data_list = load_dataset_list(subject_ids, **kwargs)
        return squeeze_trailing_singleton_dims(data_list, min_ndim=4)

    if mtype == "callable":
        kwargs = _build_load_kwargs(
            suffix=suffix,
            base_dir=base_dir,
        )
        raw_list = load_dataset_list(subject_ids, **kwargs)
        raw_list = squeeze_trailing_singleton_dims(raw_list, min_ndim=4)

        fn = method_cfg["fn"]
        fn_kwargs = method_cfg.get("kwargs", {})
        data_list = fn(raw_list, **fn_kwargs)
        return squeeze_trailing_singleton_dims(data_list, min_ndim=4)

    raise ValueError(f"Unknown method type: {mtype}")


def _infer_axis_names(ndim):
    """
    Infer axis names from dimensionality.

    4D -> x, y, z, t
    5D -> x, y, z, t, T
    """
    if ndim == 4:
        return ["x", "y", "z", "t"]
    if ndim == 5:
        return ["x", "y", "z", "t", "T"]

    raise ValueError(
        f"Unsupported ndim={ndim}. Currently only 4D and 5D are supported."
    )

def _resolve_max_lag(axis_len, axis, max_lag):
    """
    Resolve max_lag for one axis.

    Parameters
    ----------
    axis_len : int
        Length of the current axis
    axis : int
        Axis index
    max_lag : None, int, or dict

    Returns
    -------
    resolved_max_lag : int
    """
    if max_lag is None:
        return axis_len - 1

    if isinstance(max_lag, int):
        if max_lag < 0:
            raise ValueError("max_lag must be >= 0")
        return min(max_lag, axis_len - 1)

    if isinstance(max_lag, dict):
        lag_val = max_lag.get(axis, axis_len - 1)
        if lag_val < 0:
            raise ValueError("max_lag per axis must be >= 0")
        return min(lag_val, axis_len - 1)

    raise ValueError("max_lag must be None, int, or dict")


def estimate_axis_pair_correlations(
    data,
    mask_noise,
    axis_pairs=None,
    max_lag=None,
    subtract_mean=True,
    normalize=True,
    return_counts=False,
):
    """
    Estimate 2D (pairwise) autocorrelations for selected axis pairs.

    For each axis pair (ax1, ax2), estimates a 2D correlation map over
    lag pairs (lag1, lag2), using only pairs for which both samples are
    valid according to mask_noise.

    Parameters
    ----------
    data : np.ndarray
        Data array, e.g. shape (x, y, z, t, T)
    mask_noise : np.ndarray
        Bool array with same shape as data
    axis_pairs : iterable of tuple(int, int) or None
        Axis pairs to evaluate, e.g. ((0,1), (0,3), (3,4)).
        If None, all unique pairs (ax1 < ax2) are used.
    max_lag : None, int, or dict
        - None: full lag range for each axis
        - int: same max lag for all axes
        - dict: per-axis max lag, e.g. {0: 5, 3: 20}
    subtract_mean : bool
        Whether to subtract the mean of valid samples per lag pair
    normalize : bool
        Whether to normalize such that corr[0,0] = 1
    return_counts : bool
        Whether to also return the number of valid pairs per lag pair

    Returns
    -------
    pair_corrs : dict
        pair_corrs[(ax1, ax2)] = 2D np.ndarray of shape
        (max_lag_ax1 + 1, max_lag_ax2 + 1)
    pair_counts : dict, optional
        pair_counts[(ax1, ax2)] = 2D np.ndarray with valid pair counts
        Returned only if return_counts=True.
    """
    if data.shape != mask_noise.shape:
        raise ValueError("mask_noise must have the same shape as data")

    if data.ndim < 2:
        raise ValueError("data must be at least 2D for pair correlations")

    if axis_pairs is None:
        axis_pairs = []
        for ax1 in range(data.ndim):
            for ax2 in range(ax1 + 1, data.ndim):
                axis_pairs.append((ax1, ax2))

    pair_corrs = {}
    pair_counts = {}

    for ax1, ax2 in axis_pairs:
        if ax1 == ax2:
            raise ValueError(f"Axis pair ({ax1}, {ax2}) is invalid: axes must differ")

        if not (0 <= ax1 < data.ndim and 0 <= ax2 < data.ndim):
            raise ValueError(f"Invalid axis pair ({ax1}, {ax2}) for data.ndim={data.ndim}")

        max_lag1 = _resolve_max_lag(data.shape[ax1], ax1, max_lag)
        max_lag2 = _resolve_max_lag(data.shape[ax2], ax2, max_lag)

        corr_map = np.full((max_lag1 + 1, max_lag2 + 1), np.nan, dtype=np.float64)
        count_map = np.zeros((max_lag1 + 1, max_lag2 + 1), dtype=np.int64)

        for lag1 in range(max_lag1 + 1):
            for lag2 in range(max_lag2 + 1):
                slicer1 = [slice(None)] * data.ndim
                slicer2 = [slice(None)] * data.ndim

                if lag1 > 0:
                    slicer1[ax1] = slice(0, -lag1)
                    slicer2[ax1] = slice(lag1, None)

                if lag2 > 0:
                    slicer1[ax2] = slice(0, -lag2)
                    slicer2[ax2] = slice(lag2, None)

                data1 = data[tuple(slicer1)]
                data2 = data[tuple(slicer2)]

                mask1 = mask_noise[tuple(slicer1)]
                mask2 = mask_noise[tuple(slicer2)]

                valid_pairs = mask1 & mask2
                n_pairs = int(np.count_nonzero(valid_pairs))
                count_map[lag1, lag2] = n_pairs

                if n_pairs == 0:
                    continue

                arr1 = data1[valid_pairs].reshape(-1)
                arr2 = data2[valid_pairs].reshape(-1)

                if subtract_mean:
                    arr1 = arr1 - np.mean(arr1)
                    arr2 = arr2 - np.mean(arr2)

                cov = np.sum(arr1 * np.conj(arr2)) / n_pairs
                corr_map[lag1, lag2] = np.real(cov)

        if normalize and np.isfinite(corr_map[0, 0]) and corr_map[0, 0] > 0:
            corr_map = corr_map / corr_map[0, 0]

        pair_corrs[(ax1, ax2)] = corr_map
        pair_counts[(ax1, ax2)] = count_map

    if return_counts:
        return pair_corrs, pair_counts

    return pair_corrs


def estimate_axis_pair_correlations_for_dataset_list(
    dataset_list,
    mask_list,
    axis_pairs=None,
    max_lag=None,
    subtract_mean=True,
    normalize=True,
    return_counts=False,
):
    """
    Apply estimate_axis_pair_correlations to multiple datasets.
    """
    if len(dataset_list) != len(mask_list):
        raise ValueError("dataset_list and mask_list must have the same length")

    corrs_list = []
    counts_list = []

    for data, mask_noise in zip(dataset_list, mask_list):
        result = estimate_axis_pair_correlations(
            data=data,
            mask_noise=mask_noise,
            axis_pairs=axis_pairs,
            max_lag=max_lag,
            subtract_mean=subtract_mean,
            normalize=normalize,
            return_counts=return_counts,
        )

        if return_counts:
            pair_corrs, pair_counts = result
            corrs_list.append(pair_corrs)
            counts_list.append(pair_counts)
        else:
            corrs_list.append(result)

    if return_counts:
        return corrs_list, counts_list

    return corrs_list


def compute_pair_correlation_mean_std(corrs_list):
    """
    Compute mean and std of 2D pairwise correlations across datasets.

    Parameters
    ----------
    corrs_list : list of dict
        corrs_list[i][(ax1, ax2)] = 2D correlation map for dataset i

    Returns
    -------
    mean_corrs : dict
        mean_corrs[(ax1, ax2)] = mean 2D correlation map
    std_corrs : dict
        std_corrs[(ax1, ax2)] = std 2D correlation map
    """
    if len(corrs_list) == 0:
        raise ValueError("corrs_list must not be empty")

    axis_pairs = corrs_list[0].keys()

    mean_corrs = {}
    std_corrs = {}

    for pair in axis_pairs:
        arrs = [corrs[pair] for corrs in corrs_list]

        min_shape0 = min(a.shape[0] for a in arrs)
        min_shape1 = min(a.shape[1] for a in arrs)
        arrs = [a[:min_shape0, :min_shape1] for a in arrs]

        stacked = np.stack(arrs, axis=0)

        mean_corrs[pair] = np.mean(stacked, axis=0)
        std_corrs[pair] = np.std(stacked, axis=0)

    return mean_corrs, std_corrs


def _axis_pair_names_to_indices(pair_axes):
    """
    Convert axis-name pairs to index pairs.

    Example
    -------
    [("x", "t"), ("t", "T")] -> [(0, 3), (3, 4)]
    """
    return [(_axis_name_to_index(ax1), _axis_name_to_index(ax2)) for ax1, ax2 in pair_axes]




def run_noise_analysis_pipeline(
    subject_ids,
    methods,
    mask_source="method",
    mask_type="spatial",
    percentile=5,
    fid_window=(0.3, 0.9),
    suffix="normalized",
    base_dir=None,
    max_lag=None,
    compute_spatial=True,
    compute_pairwise=False,
    return_pair_counts=False,
    subtract_mean=True,
    normalize=True,
):
    """
    Orchestration wrapper around the correlation functions.

    Behavior
    --------
    - 1D correlations are always computed along all available axes.
    - 2D pairwise correlations are computed for all unique axis pairs if requested.
    - Dimensionality is inferred automatically from the data:
        4D -> (x, y, z, t)
        5D -> (x, y, z, t, T)

    Parameters
    ----------
    subject_ids : list of str
    methods : dict
    mask_source : str
        "raw" or "method"
    mask_type : str
        "spatial", "fid", or "fid_window"
    percentile : float
        Used for "spatial" and "fid"
    fid_window : tuple of float
        Only for mask_type="fid_window"
    suffix : str
    base_dir : str or None
    max_lag : None, int, or dict
        Used only for pairwise 2D autocorrelations.
        1D correlations are always computed over the full available range.
    compute_spatial : bool
        Whether to separately store the 1D correlations of spatial axes x, y, z.
    compute_pairwise : bool
        Whether to compute all pairwise 2D autocorrelations automatically.
    return_pair_counts : bool
        Whether to also keep valid-pair count maps for pairwise correlations.
    subtract_mean : bool
    normalize : bool

    Returns
    -------
    results : dict
    """
    datasets_by_method = {}
    masks_by_method = {}
    acf_stats_by_method = {}
    spatial_stats_by_method = {}
    pair_corr_stats_by_method = {}
    pair_count_stats_by_method = {}

    raw_list = None
    if mask_source == "raw":
        kwargs = _build_load_kwargs(
            suffix=suffix,
            base_dir=base_dir,
        )
        raw_list = load_dataset_list(subject_ids, **kwargs)
        raw_list = squeeze_trailing_singleton_dims(raw_list, min_ndim=4)

    inferred_axis_names = None
    inferred_axis_indices = None
    inferred_spatial_axes = None

    for method_name, method_cfg in methods.items():
        data_list = _load_or_build_method_dataset(
            subject_ids=subject_ids,
            method_cfg=method_cfg,
            suffix=suffix,
            base_dir=base_dir,
        )
        datasets_by_method[method_name] = data_list

        if len(data_list) == 0:
            raise ValueError(f"No datasets loaded for method '{method_name}'")

        ndim = data_list[0].ndim
        axis_names = _infer_axis_names(ndim)
        axis_indices = list(range(ndim))
        spatial_axes = tuple(ax for ax in (0, 1, 2) if ax < ndim)

        # store once globally, but check consistency across methods
        if inferred_axis_names is None:
            inferred_axis_names = axis_names
            inferred_axis_indices = axis_indices
            inferred_spatial_axes = spatial_axes
        else:
            if axis_names != inferred_axis_names:
                raise ValueError(
                    f"Inconsistent dimensionality across methods. "
                    f"Previous axis names: {inferred_axis_names}, current: {axis_names}"
                )

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

        # 1D correlations along all available axes
        acfs = estimate_axis_correlations_for_dataset_list(
            dataset_list=data_list,
            mask_list=mask_list,
            axes=axis_indices,
            max_lag=None,
            subtract_mean=subtract_mean,
            normalize=normalize,
        )
        mean_acf, std_acf = compute_correlation_mean_std(acfs)
        acf_stats_by_method[method_name] = (mean_acf, std_acf)

        # separate 1D spatial correlations
        if compute_spatial:
            spatial_corrs = estimate_axis_correlations_for_dataset_list(
                dataset_list=data_list,
                mask_list=mask_list,
                axes=spatial_axes,
                max_lag=None,
                subtract_mean=subtract_mean,
                normalize=normalize,
            )
            mean_spatial, std_spatial = compute_correlation_mean_std(spatial_corrs)
            spatial_stats_by_method[method_name] = (mean_spatial, std_spatial)

        # 2D pairwise correlations: all unique pairs automatically
        if compute_pairwise:
            if return_pair_counts:
                pair_corrs_list, pair_counts_list = estimate_axis_pair_correlations_for_dataset_list(
                    dataset_list=data_list,
                    mask_list=mask_list,
                    axis_pairs=None,
                    max_lag=max_lag,
                    subtract_mean=subtract_mean,
                    normalize=normalize,
                    return_counts=True,
                )

                mean_pair_corrs, std_pair_corrs = compute_pair_correlation_mean_std(pair_corrs_list)
                mean_pair_counts, std_pair_counts = compute_pair_correlation_mean_std(pair_counts_list)

                pair_corr_stats_by_method[method_name] = (mean_pair_corrs, std_pair_corrs)
                pair_count_stats_by_method[method_name] = (mean_pair_counts, std_pair_counts)

            else:
                pair_corrs_list = estimate_axis_pair_correlations_for_dataset_list(
                    dataset_list=data_list,
                    mask_list=mask_list,
                    axis_pairs=None,
                    max_lag=max_lag,
                    subtract_mean=subtract_mean,
                    normalize=normalize,
                    return_counts=False,
                )

                mean_pair_corrs, std_pair_corrs = compute_pair_correlation_mean_std(pair_corrs_list)
                pair_corr_stats_by_method[method_name] = (mean_pair_corrs, std_pair_corrs)

    return {
        "subject_ids": subject_ids,
        "axis_names": inferred_axis_names,
        "axis_indices": inferred_axis_indices,
        "spatial_axes": inferred_spatial_axes,
        "datasets_by_method": datasets_by_method,
        "masks_by_method": masks_by_method,
        "acf_stats_by_method": acf_stats_by_method,
        "spatial_stats_by_method": spatial_stats_by_method,
        "pair_corr_stats_by_method": pair_corr_stats_by_method,
        "pair_count_stats_by_method": pair_count_stats_by_method,
        "mask_type": mask_type,
        "mask_source": mask_source,
        "percentile": percentile,
        "fid_window": fid_window,
        "max_lag": max_lag,
        "compute_pairwise": compute_pairwise,
        "return_pair_counts": return_pair_counts,
    }


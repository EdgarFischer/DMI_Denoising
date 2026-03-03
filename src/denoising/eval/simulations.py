import os
import numpy as np
import sys

sys.path.append(os.path.abspath("../../src"))

def load_metab_map(root, method, metabolite, suffix="Orig"):
    """
    Loads: {root}/{method}/{metabolite}_amp_{method}_{suffix}.npy
    """
    fname = f"{metabolite}_amp_{method}_{suffix}.npy"
    path = os.path.join(root, method, fname)
    return np.flip(np.swapaxes(np.load(path), 0, -2), axis=1)/1000000

def load_multiple_metab_maps(root, methods, metabolite, suffix="Orig"):
    """
    methods: List of methods 

    returns: List of Metab Maps
    """
    return [load_metab_map(root, m, metabolite, suffix) for m in methods]

def error_mean_var_maps(maps, gt, *, mean_over_time=False, ddof=0):
    """
    Compute voxel-wise error mean/variance against GT for 4D metab maps (X,Y,Z,T).

    Parameters
    ----------
    maps : list[np.ndarray] or np.ndarray
        List of recon maps, each shape (X,Y,Z,T), or array shape (N,X,Y,Z,T).
    gt : np.ndarray
        Ground-truth map, shape (X,Y,Z,T).
    mean_over_time : bool
        If True: also average over time => outputs shape (X,Y,Z).
        If False: keep time dimension => outputs shape (X,Y,Z,T).
    ddof : int
        Degrees of freedom for variance (0 = population var, 1 = sample var).

    Returns
    -------
    err_mean : np.ndarray
    err_var  : np.ndarray
    """
    arr = np.stack(maps, axis=0) if isinstance(maps, (list, tuple)) else np.asarray(maps)
    if arr.ndim != 5:
        raise ValueError(f"`maps` must be list of 4D or 5D array, got shape {arr.shape}")
    gt = np.asarray(gt)
    if gt.shape != arr.shape[1:]:
        raise ValueError(f"GT shape {gt.shape} must match maps shape {arr.shape[1:]}")

    err = arr - gt[None, ...]                 # (N,X,Y,Z,T)
    if mean_over_time:
        err = err.mean(axis=-1)               # (N,X,Y,Z)

    err_mean = err.mean(axis=0)               # (X,Y,Z, T) or (X,Y,Z)
    err_var  = err.var(axis=0, ddof=ddof)     # (X,Y,Z, T) or (X,Y,Z)
    return err_mean, err_var

def error_mean_var_for_methods(root, methods, metabolite, gt_method, suffix="Orig", *, mean_over_time=False, ddof=0):
    """
    Convenience wrapper using your loaders.

    Returns
    -------
    err_mean_map, err_var_map : np.ndarray
    """
    gt = load_metab_map(root, gt_method, metabolite, suffix=suffix)
    maps = load_multiple_metab_maps(root, methods, metabolite, suffix=suffix)
    return error_mean_var_maps(maps, gt, mean_over_time=mean_over_time, ddof=ddof)

def relative_rmse(gt, pred, mask, eps=1e-12):
    """
    gt, pred: arrays with shape (x,y,z)
    mask: weight map (can be binary or float in [0,1])
    returns: weighted relative RMSE
    """

    gt = np.asarray(gt)
    pred = np.asarray(pred)
    w = np.asarray(mask)

    # gültige Voxels: w>0 und finite Werte
    m = (w > 0) & np.isfinite(w) & np.isfinite(gt) & np.isfinite(pred)

    if not np.any(m):
        return np.nan

    ww = w[m].astype(np.float64)
    diff2 = (gt[m] - pred[m])**2

    # gewichtetes MSE
    weighted_mse = np.sum(ww * diff2) / (np.sum(ww) + eps)

    # Normalisierung über Range in Ωm
    denom = (np.max(gt[m]) - np.min(gt[m])) + eps

    return np.sqrt(weighted_mse) / denom

def relative_rmse_time(gt, pred, mask, eps=1e-12):
    """
    Returns relative RMSE as a function of time (last index)
    """
    T = gt.shape[-1]
    RMSE = []

    for i in range(0,T):
        RMSE.append(relative_rmse(gt[..., i], pred[..., i], mask, eps))

    return RMSE

def relative_rmse_time_stats(gt, preds, mask, eps=1e-12):
    """
    preds: list of predictions with same shape as gt

    returns:
        mean_rmse_time, std_rmse_time
    """
    rmses = np.stack([relative_rmse_time(gt, p, mask, eps) for p in preds], axis=0)
    return rmses.mean(axis=0), rmses.std(axis=0)

# ---------- helper ----------
def compute_mean_std_rrmse_time(root, metab, mask, gt_method, rep_methods, suffix="Orig", eps=1e-12):
    """
    rep_methods: dict label -> list of method-folders (reps)
    returns: dict label -> (mean(T,), std(T,))
    """
    gt = load_metab_map(root, gt_method, metab, suffix)

    out = {}
    for label, reps in rep_methods.items():
        preds = load_multiple_metab_maps(root, reps, metab, suffix)
        mean_t, std_t = relative_rmse_time_stats(gt, preds, mask, eps)
        out[label] = (np.asarray(mean_t), np.asarray(std_t))
    return out
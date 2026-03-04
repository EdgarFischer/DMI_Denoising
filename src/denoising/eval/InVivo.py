import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.optimize import nnls

def suffix_from_method(method: str) -> str:
    # "P08_deep_tMPPCA_5D" -> "deep_tMPPCA_5D"
    return method.split("_", 1)[1] if "_" in method else method


# -------------------- YOUR FIT-RATE PIPELINE (needed for stats_*) --------------------
def _suffix(quality_clip: bool, outlier_clip: bool) -> str:
    return "OutlierClip" if outlier_clip else ("QualityClip" if quality_clip else "Orig")

def _load_map(
    metabolite: str,
    method: str,
    suffix: str,
    data_dir: str,
    *,
    kind: str = "amp",   # "amp" or "sd"
) -> np.ndarray:
    path = os.path.join(data_dir, method, f"{metabolite}_{kind}_{method}_{suffix}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File missing: {path}")
    return np.load(path)

def load_sd_map(
    metabolite: str,
    method: str,
    *,
    data_dir: str = "MetabMaps",
    quality_clip: bool = False,
    outlier_clip: bool = False,
) -> np.ndarray:
    suffix = _suffix(quality_clip, outlier_clip)
    f_sd = os.path.join(data_dir, method, f"{metabolite}_sd_{method}_{suffix}.npy")
    if not os.path.isfile(f_sd):
        raise FileNotFoundError(f"SD map missing: {f_sd}")
    sd = np.load(f_sd)
    if sd.ndim != 4:
        raise ValueError(f"Unexpected SD shape {sd.shape}, expected (X,Y,Z,T).")
    return sd

def load_brain_mask(
    subject: str,
    *,
    tissue_dir: str = "../datasets",
) -> np.ndarray:
    mask_path = os.path.join(tissue_dir, subject, "mask.npy")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Brain mask missing: {mask_path}")
    brain_mask = np.swapaxes(np.load(mask_path), 0, -1) > 0
    return brain_mask

def fit_rate_from_sd(
    sd_4d: np.ndarray,
    brain_mask: np.ndarray,
    *,
    crlb_thresh: float = 30.0,
) -> np.ndarray:
    if sd_4d.shape[:3] != brain_mask.shape:
        raise ValueError(f"Brain mask shape {brain_mask.shape} does not match SD map {sd_4d.shape[:3]}")
    T = sd_4d.shape[-1]
    rates = np.zeros(T, dtype=float)
    denom = brain_mask.sum()
    if denom == 0:
        return np.full(T, np.nan)
    for t in range(T):
        sd_t = sd_4d[..., t]
        ok = (sd_t <= crlb_thresh) & np.isfinite(sd_t) & brain_mask
        rates[t] = 100.0 * ok.sum() / denom
    return rates

def fit_rate_for_methods(
    metabolite: str,
    methods: list[str],
    subject: str,
    *,
    data_dir: str = "MetabMaps",
    tissue_dir: str = "../datasets",
    quality_clip: bool = False,
    outlier_clip: bool = False,
    crlb_thresh: float = 30.0,
) -> dict[str, np.ndarray]:
    brain_mask = load_brain_mask(subject, tissue_dir=tissue_dir)
    out = {}
    for m in methods:
        sd = load_sd_map(
            metabolite, m,
            data_dir=data_dir,
            quality_clip=quality_clip,
            outlier_clip=outlier_clip,
        )
        out[m] = fit_rate_from_sd(sd, brain_mask, crlb_thresh=crlb_thresh)
    return out

def fit_rate_group_stats(
    metabolite: str,
    subjects: list[str],
    *,
    method_suffixes: list[str] = ["noisy", "deep_tMPPCA_5D", "tMPPCA_5D"],
    data_dir: str = "MetabMaps",
    tissue_dir: str = "../datasets",
    crlb_thresh: float = 30.0,
    quality_clip: bool = False,
    outlier_clip: bool = False,
    strict: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    per_method = {suf: [] for suf in method_suffixes}
    for subj in subjects:
        methods = [f"{subj}_{suf}" for suf in method_suffixes]
        try:
            rates_dict = fit_rate_for_methods(
                metabolite, methods, subj,
                data_dir=data_dir,
                tissue_dir=tissue_dir,
                quality_clip=quality_clip,
                outlier_clip=outlier_clip,
                crlb_thresh=crlb_thresh,
            )
        except Exception as e:
            if strict:
                raise
            print(f"[WARN] Skipping {subj} ({metabolite}): {e}")
            continue

        for suf, m in zip(method_suffixes, methods):
            per_method[suf].append(rates_dict[m])

    stats = {}
    for suf, series_list in per_method.items():
        if len(series_list) == 0:
            stats[suf] = {"mean": np.array([np.nan]), "std": np.array([np.nan]), "n": np.array([0])}
            continue
        arr = np.stack(series_list, axis=0)  # (Nsub, T)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        n    = np.sum(np.isfinite(arr), axis=0)
        stats[suf] = {"mean": mean, "std": std, "n": n}
    return stats

def fit_pve_single_metabolite(
    metabo_map: np.ndarray,                # (X,Y,Z,T)
    tissue_maps: np.ndarray,               # (X,Y,Z,3) GM, WM, CSF
    brain_mask: np.ndarray | None = None,  # (X,Y,Z) or (X,Y,Z,T)
    *,
    method: str = "nnls"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate pure-tissue time courses for one metabolite.
    Returns:
      conc     (3,T)  order [WM, GM, CSF]
      conc_std (3,T)
    """
    if metabo_map.ndim != 4:
        raise ValueError(f"metabo_map must be (X,Y,Z,T), got {metabo_map.shape}")
    T = metabo_map.shape[-1]

    gm, wm, csf = (tissue_maps[..., i].ravel() for i in range(3))

    if brain_mask is None:
        brain_mask = np.ones(metabo_map.shape[:-1], dtype=bool)
        brain_mask = brain_mask[..., None] * np.ones((1, 1, 1, T), dtype=bool)
    else:
        brain_mask = np.asarray(brain_mask, dtype=bool)
        if brain_mask.ndim == 3:
            brain_mask = brain_mask[..., None] * np.ones((1, 1, 1, T), dtype=bool)
        elif brain_mask.ndim != 4:
            raise ValueError(f"brain_mask must be 3D or 4D, got {brain_mask.shape}")

    A_all = np.vstack([wm, gm, csf]).T     # (nVox,3) -> WM,GM,CSF
    Y_all = metabo_map.reshape(-1, T)

    conc = np.empty((3, T), dtype=np.float64)
    conc_std = np.empty((3, T), dtype=np.float64)

    for t in range(T):
        bm = brain_mask[..., t].ravel()
        A = A_all[bm]
        y = Y_all[bm, t]

        valid = np.isfinite(y) & (y != 0)
        A = A[valid]
        y = y[valid]

        if y.size < 3:
            conc[:, t] = np.nan
            conc_std[:, t] = np.nan
            continue

        if method == "nnls":
            x, _ = nnls(A, y)
            res = y - A @ x
            cov = np.linalg.pinv(A.T @ A)
        else:
            x, *_ = lstsq(A, y, rcond=None)
            res = y - A @ x
            cov = np.linalg.pinv(A.T @ A)

        dof = max(len(y) - 3, 1)
        sigma2 = (res @ res) / dof

        conc[:, t] = x
        conc_std[:, t] = np.sqrt(sigma2 * np.diag(cov))

    return conc.astype(np.float32), conc_std.astype(np.float32)

# =============================================================================
# =============   Per-subject PVE (ABSOLUTE + GLOBAL WATER SCALE)   ===========
# =============================================================================

def compute_subject_conc_abs_and_global_scale(
    *,
    metabolite: str,
    water_name: str,
    method_templates: list[str],    # e.g. ["{subject}_noisy", "{subject}_tMPPCA_5D", ...]
    subject: str,
    quality_clip: bool,
    outlier_clip: bool,
    data_dir: str,
    tissue_dir: str,
    crlb_threshold: float | None,
    scale_mode: str = "median_over_time",  # "median_over_time" or "per_timepoint"
    eps: float = 1e-8,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Returns:
      method_names: resolved names for this subject (M)
      conc_m:       list of (3,T) for metabolite (absolute)
      conc_m_sd:    list of (3,T)
      scale:        list of scale arrays:
                     - if scale_mode="median_over_time": each scale is scalar (float32)
                     - if scale_mode="per_timepoint":   each scale is (T,) (float32)
    """
    suffix = _suffix(quality_clip, outlier_clip)
    methods_subj = [tpl.format(subject=subject) for tpl in method_templates]

    # Segmentation & brain mask
    seg = np.load(os.path.join(tissue_dir, subject, "gm_wm_csf_segmentation.npy"))
    segmentations = np.swapaxes(seg, 0, -2).astype(np.float32)

    brain_mask = np.swapaxes(np.load(os.path.join(tissue_dir, subject, "mask.npy")), 0, -1) > 0

    # Optional CRLB mask from REFERENCE only (methods_subj[0])
    crlb_mask_4d = None
    if crlb_threshold is not None:
        src = methods_subj[0]
        crlb = _load_map(metabolite, src, suffix, data_dir, kind="sd").astype(np.float32)
        valid = np.isfinite(crlb) & (crlb < 1000)
        ok = valid & (crlb <= crlb_threshold)
        bm4 = brain_mask[..., None] * np.ones((1, 1, 1, crlb.shape[-1]), dtype=bool)
        crlb_mask_4d = bm4 & ok

    fit_mask = crlb_mask_4d if crlb_mask_4d is not None else brain_mask

    conc_m, conc_m_sd = [], []
    scales = []

    for m in methods_subj:
        # ---- load ABSOLUTE metabolite + water maps (no division!) ----
        amp_m = _load_map(metabolite, m, suffix, data_dir, kind="amp").astype(np.float32)
        amp_w = _load_map(water_name,  m, suffix, data_dir, kind="amp").astype(np.float32)

        # ---- PVE fits (absolute) ----
        c_m, c_m_sd = fit_pve_single_metabolite(amp_m, segmentations, brain_mask=fit_mask)
        c_w, _      = fit_pve_single_metabolite(amp_w, segmentations, brain_mask=fit_mask)

        # Water tissue timecourses (order [WM, GM, CSF])
        w_wm = c_w[0]
        w_gm = c_w[1]

        # Global water scale: average GM/WM (as you requested)
        w_gmwm = 0.5 * (w_gm + w_wm)

        if scale_mode == "per_timepoint":
            scale = w_gmwm.astype(np.float32)              # (T,)
        elif scale_mode == "median_over_time":
            scale = np.nanmedian(w_gmwm).astype(np.float32)  # scalar
        else:
            raise ValueError("scale_mode must be 'median_over_time' or 'per_timepoint'")

        # Guard against divide-by-zero
        if np.isscalar(scale):
            if not np.isfinite(scale) or abs(scale) < eps:
                scale = np.float32(np.nan)
        else:
            scale = np.where(np.isfinite(scale) & (np.abs(scale) > eps), scale, np.nan).astype(np.float32)

        conc_m.append(c_m)
        conc_m_sd.append(c_m_sd)
        scales.append(scale)

    return methods_subj, conc_m, conc_m_sd, scales
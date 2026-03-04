from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from numpy.linalg import lstsq
import sys
from scipy.optimize import nnls

sys.path.append(os.path.abspath("../src"))
from denoising.eval.InVivo import *


def plot_timecourse_metabolite_group_global_water(
    metabolite: str,
    method_templates: list[str],
    *,
    subjects: list[str],
    time_minutes: np.ndarray,
    y_label: str = "Concentration / global HDO (a.u.)",
    method_labels: dict[str, str] | None = None,
    quality_clip: bool = False,
    outlier_clip: bool = False,
    data_dir: str = "MetabMaps",
    tissue_dir: str = "../datasets",
    colors: dict[str, str] | None = None,
    water_name: str = "water",
    crlb_threshold: float | None = None,
    scale_mode: str = "median_over_time",   # "median_over_time" (recommended) or "per_timepoint"
    x_label: str = "Time (min)",
    xticks: np.ndarray | None = None,
    eps: float = 1e-8,
) -> None:
    """
    Group plot (Mean ± SD over subjects) for GM/WM timecourses.
    Colors encode methods, line style encodes tissue:
        solid = GM, dashed = WM

    GLOBAL water scaling:
      - Fit absolute concentrations for metabolite and water via PVE.
      - Normalize tissue time courses by a global subject-level water scale:
          scale = median_T(0.5*(HDO_GM(T)+HDO_WM(T)))  [default]
    """

    if len(method_templates) < 2:
        raise ValueError("Need at least 2 methods: [reference, comparison...]")
    if len(subjects) < 1:
        raise ValueError("subjects must be non-empty")
    if time_minutes is None or len(time_minutes) == 0:
        raise ValueError("time_minutes must be provided and non-empty")

    # Colors (paper-like defaults)
    default_colors = {
        "noisy": "#333333",
        "tMPPCA_5D": "#E69F00",
        "deep_tMPPCA_5D": "#009E73",
    }
    if colors is None:
        colors = default_colors

    # ---------- compute per subject ----------
    # We'll stack normalized tissue curves: (Nsubj, M, 3, T)
    conc_stack = []
    methods_resolved_0 = None

    for subj in subjects:
        methods_subj, conc_m, conc_m_sd, scales = compute_subject_conc_abs_and_global_scale(
            metabolite=metabolite,
            water_name=water_name,
            method_templates=method_templates,
            subject=subj,
            quality_clip=quality_clip,
            outlier_clip=outlier_clip,
            data_dir=data_dir,
            tissue_dir=tissue_dir,
            crlb_threshold=crlb_threshold,
            scale_mode=scale_mode,
            eps=eps,
        )

        if methods_resolved_0 is None:
            methods_resolved_0 = methods_subj

        # Normalize each method's tissue timecourses by its own global water scale
        conc_norm = []
        for c, s in zip(conc_m, scales):
            if np.isscalar(s):
                cn = c / (s + eps)
            else:
                # s is (T,) -> broadcast to (3,T)
                cn = c / (s[None, :] + eps)
            conc_norm.append(cn.astype(np.float32))

        conc_stack.append(np.stack(conc_norm, axis=0))

    conc_stack = np.stack(conc_stack, axis=0).astype(np.float32)  # (Nsubj,M,3,T)
    mean_conc = np.nanmean(conc_stack, axis=0)                    # (M,3,T)
    sd_conc = (np.nanstd(conc_stack, axis=0, ddof=1) if len(subjects) > 1 else np.zeros_like(mean_conc))

    # ---------- names ----------
    def _display_name(tpl: str, resolved: str) -> str:
        return tpl[len("{subject}_"):] if tpl.startswith("{subject}_") else resolved

    if methods_resolved_0 is None:
        raise RuntimeError("No methods resolved (unexpected).")

    names = [_display_name(t, r) for t, r in zip(method_templates, methods_resolved_0)]
    ref_name = names[0]
    comp_names = names[1:]

    # Pretty legend labels
    if method_labels is None:
        pretty = {n: n for n in names}
    else:
        pretty = {n: method_labels.get(n, n) for n in names}

    # ---------- sanity check: time axis length ----------
    T = mean_conc.shape[-1]
    if len(time_minutes) != T:
        raise ValueError(f"time_minutes length ({len(time_minutes)}) must match repeats T ({T})")

    # ---------- y-limits (common across panels) ----------
    wm = mean_conc[:, 0, :]
    gm = mean_conc[:, 1, :]
    wm_sd = sd_conc[:, 0, :]
    gm_sd = sd_conc[:, 1, :]

    ymin = np.nanmin(np.stack([wm - wm_sd, gm - gm_sd], axis=0))
    ymax = np.nanmax(np.stack([wm + wm_sd, gm + gm_sd], axis=0))
    pad = 0.06 * (ymax - ymin + 1e-12)

    # ---------- plot ----------
    fig, axes = plt.subplots(
        1, len(comp_names),
        figsize=(9.0, 3.7),
        sharey=True,
        squeeze=False
    )

    lw, ms, cap = 1.8, 3.5, 2
    elw = 1.0

    for j, name in enumerate(comp_names):
        ax = axes[0, j]
        ref_col = colors.get(ref_name, "#333333")
        cmp_col = colors.get(name, "#0072B2")

        # reference (GM solid, WM dashed)
        ax.errorbar(time_minutes, mean_conc[0, 1], yerr=sd_conc[0, 1],
                    color=ref_col, linestyle="-", marker="o",
                    lw=lw, ms=ms, capsize=cap, elinewidth=elw)
        ax.errorbar(time_minutes, mean_conc[0, 0], yerr=sd_conc[0, 0],
                    color=ref_col, linestyle="--", marker="o",
                    lw=lw, ms=ms, capsize=cap, elinewidth=elw)

        # comparison
        idx = j + 1
        ax.errorbar(time_minutes, mean_conc[idx, 1], yerr=sd_conc[idx, 1],
                    color=cmp_col, linestyle="-", marker="o",
                    lw=lw, ms=ms, capsize=cap, elinewidth=elw)
        ax.errorbar(time_minutes, mean_conc[idx, 0], yerr=sd_conc[idx, 0],
                    color=cmp_col, linestyle="--", marker="o",
                    lw=lw, ms=ms, capsize=cap, elinewidth=elw)

        ax.set_xlabel(x_label)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, alpha=0.2)

        ax.set_xticks(time_minutes if xticks is None else xticks)

        if j == 0:
            ax.set_ylabel(y_label)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # ---------- legends (methods + tissue styles) ----------
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color=colors.get(n, "k"), lw=lw, label=pretty[n])
        for n in names
    ]
    method_leg = axes[0, 0].legend(handles=method_handles, loc="upper left", frameon=False, fontsize=9)
    axes[0, 0].add_artist(method_leg)

    tissue_handles = [
        Line2D([0], [0], color="k", lw=lw, linestyle="-",  label="GM (solid)"),
        Line2D([0], [0], color="k", lw=lw, linestyle="--", label="WM (dashed)"),
    ]
    axes[0, 0].legend(handles=tissue_handles, loc="lower right", frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"SavedGraphics/timecourse_group_{metabolite}_globalHDO.pdf", bbox_inches="tight")
    plt.show()
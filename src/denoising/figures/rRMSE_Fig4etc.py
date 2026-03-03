import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os

sys.path.append(os.path.abspath("../src"))
from denoising.eval.simulations import *

def plot_rrmse_2x2(
    *,
    root: str,
    gt_method: str,
    suffix: str,
    metabs,               # list of (metab_name, mask)
    display_name: dict,   # e.g. {"water":"HDO", ...}
    time_axis: np.ndarray,
    rep_methods: dict,    # e.g. {"Noisy": [...], "tMPPCA":[...], ...}
    method_style: dict,   # e.g. {"Noisy": {...}, "xy": {...}, ...}
    ylabel_left="rRMSE",  # or "relative RMSE"
    title=None,
    figsize=(9.0, 7.2),
    lw=1.8,
    ms=3.5,
    cap=2,
    elw=1.0,
    ylims=None,           # e.g. {"water": (0.03,0.10), "Glc": (0.03,0.35)}
    legend_ax_index=1,    # axes[1]
    legend_loc="upper right",
    save_path=None,
    show=True,
):
    """
    Unified 2x2 rRMSE timecourse plotter.

    rep_methods:
        dict: label -> list of method-folder-names passed to compute_mean_std_rrmse_time

    method_style[label] can contain:
        color: str
        marker: str
        linestyle: str (default "-")
        use_errorbar: bool (default True)
        label: str (legend display label; default = key)
    """

    # ---- global style (keep consistent) ----
    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.ravel()

    # ---- legend handles (consistent with plotted styles) ----
    legend_handles = []
    for lab in rep_methods.keys():
        st = method_style.get(lab, {})
        legend_handles.append(
            Line2D(
                [0], [0],
                color=st.get("color", "k"),
                marker=st.get("marker", "o"),
                linestyle=st.get("linestyle", "-"),
                lw=lw,
                ms=ms,
                label=st.get("label", lab),
            )
        )

    # ---- plot panels ----
    for ax, (metab, mask) in zip(axes, metabs):

        curves = compute_mean_std_rrmse_time(
            root, metab, mask,
            gt_method=gt_method,
            rep_methods=rep_methods,
            suffix=suffix
        )

        x = np.asarray(time_axis)

        for lab in rep_methods.keys():
            st = method_style.get(lab, {})
            mean_t, std_t = curves[lab]
            mean_t = np.asarray(mean_t)

            use_errorbar = st.get("use_errorbar", True)

            if use_errorbar:
                std_t = np.asarray(std_t)
                ax.errorbar(
                    x, mean_t,
                    yerr=std_t,
                    color=st.get("color", "k"),
                    marker=st.get("marker", "o"),
                    linestyle=st.get("linestyle", "-"),
                    lw=lw,
                    ms=ms,
                    capsize=cap,
                    elinewidth=elw,
                    zorder=3,
                )
            else:
                ax.plot(
                    x, mean_t,
                    color=st.get("color", "k"),
                    marker=st.get("marker", "o"),
                    linestyle=st.get("linestyle", "-"),
                    lw=lw,
                    ms=ms,
                    zorder=3,
                )

        ax.set_title(display_name.get(metab, metab))

        ax.grid(True, alpha=0.2)
        ax.set_xticks(x)

        if ylims is not None and metab in ylims:
            ax.set_ylim(*ylims[metab])

        if ax in (axes[0], axes[2]):
            ax.set_ylabel(ylabel_left)
        else:
            ax.set_ylabel("")

    axes[2].set_xlabel("Time (min)")
    axes[3].set_xlabel("Time (min)")

    if title is not None:
        fig.text(0.02, 0.98, title, ha="left", va="top", fontsize=13)

    axes[legend_ax_index].legend(handles=legend_handles, frameon=False, loc=legend_loc)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

import numpy as np
import matplotlib.pyplot as plt

def plot_spectral_failure_cases_with_residuals(
    *,
    GT, xy, xT,
    x: int, y: int, z: int,
    t_left: int = 2,          # xy failure case
    t_right: int = 7,         # xT failure case
    ppm=None,
    ppm_lo: float = 6.0,
    ppm_hi: float = 1.0,
    scale: float = 1e-3,
    left_label: str = "xy",
    right_label: str = "xT",
    left_color: str = "#0072B2",
    right_color: str = "#8B1A1A",
    gt_color: str = "black",
    figsize=(9.0, 5.4),
    wspace: float = 0.25,
    hspace: float = 0.10,
    save_path: str | None = "SpectralConsistency_FailureCases_withResiduals.pdf",
    show: bool = True,
):
    """
    Plots two failure cases side-by-side (top: spectra, bottom: residuals),
    matching the layout of your snippet.

    Expected array shapes:
      GT[x,y,z,f,t], xy[x,y,z,f,t], xT[x,y,z,f,t]
    where f is spectral dimension and t is time index.

    ppm:
      1D array of length f. If None, will be inferred as linspace(8,1,f).
    """

    def robust_ylim_upper(arr, p_lo=1.0, p_hi=98.5, pad_up=0.25, pad_lo=0.20):
        lo, hi = np.percentile(arr, [p_lo, p_hi])
        rng = hi - lo
        return lo - pad_lo * rng, hi + pad_up * rng

    def symmetric_ylim(arr, p=98.5, pad=0.15):
        m = np.percentile(np.abs(arr), p)
        return -(1 + pad) * m, (1 + pad) * m

    # --- ppm + window ---
    f = GT.shape[3]
    if ppm is None:
        ppm = np.linspace(8, 1, f)
    ppm = np.asarray(ppm)

    mask = (ppm >= ppm_hi) & (ppm <= ppm_lo)
    ppm_w = ppm[mask]

    # --- extract spectra (real part) ---
    gt_L = GT[x, y, z, :, t_left].real[mask] * scale
    left = xy[x, y, z, :, t_left].real[mask] * scale
    res_L = left - gt_L

    gt_R = GT[x, y, z, :, t_right].real[mask] * scale
    right = xT[x, y, z, :, t_right].real[mask] * scale
    res_R = right - gt_R

    yl_spec_L = robust_ylim_upper(np.concatenate([gt_L, left]))
    yl_spec_R = robust_ylim_upper(np.concatenate([gt_R, right]))
    yl_res_L  = symmetric_ylim(res_L)
    yl_res_R  = symmetric_ylim(res_R)

    # --- plot layout ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], wspace=wspace, hspace=hspace)

    axL  = fig.add_subplot(gs[0, 0])
    axLr = fig.add_subplot(gs[1, 0], sharex=axL)

    axR  = fig.add_subplot(gs[0, 1])
    axRr = fig.add_subplot(gs[1, 1], sharex=axR)

    # LEFT
    axL.plot(ppm_w, gt_L, '--', color=gt_color, lw=1.4, label='GT')
    axL.plot(ppm_w, left, color=left_color, lw=1.6, label=left_label)
    axL.set_title(f"Failure case: {left_label}", fontsize=11)
    axL.set_xlim(ppm_lo, ppm_hi)
    axL.set_ylim(*yl_spec_L)
    axL.grid(True, alpha=0.15)
    axL.legend(frameon=False, loc="upper right", fontsize=9)
    axL.set_ylabel("Signal [a.u.]")

    axLr.axhline(0, color=gt_color, lw=0.8, alpha=0.6)
    axLr.plot(ppm_w, res_L, color=left_color, lw=1.2)
    axLr.set_ylim(*yl_res_L)
    axLr.grid(True, alpha=0.15)
    axLr.set_xlabel("Chemical shift [ppm]")
    axLr.set_ylabel("Residual")

    # RIGHT
    axR.plot(ppm_w, gt_R, '--', color=gt_color, lw=1.4, label='GT')
    axR.plot(ppm_w, right, color=right_color, lw=1.6, label=right_label)
    axR.set_title(f"Failure case: {right_label}", fontsize=11)
    axR.set_xlim(ppm_lo, ppm_hi)
    axR.set_ylim(*yl_spec_R)
    axR.grid(True, alpha=0.15)
    axR.legend(frameon=False, loc="upper right", fontsize=9)

    axRr.axhline(0, color=gt_color, lw=0.8, alpha=0.6)
    axRr.plot(ppm_w, res_R, color=right_color, lw=1.2)
    axRr.set_ylim(*yl_res_R)
    axRr.grid(True, alpha=0.15)
    axRr.set_xlabel("Chemical shift [ppm]")

    # remove top x tick labels
    plt.setp(axL.get_xticklabels(), visible=False)
    plt.setp(axR.get_xticklabels(), visible=False)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
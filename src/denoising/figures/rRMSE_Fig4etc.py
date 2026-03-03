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
    metabs,
    display_name: dict,
    time_axis: np.ndarray,
    rep_methods: dict,
    method_style: dict,
    ylabel_left="rRMSE",
    title=None,
    figsize=(9.0, 7.2),
    lw=1.8,
    ms=3.5,
    cap=2,
    elw=1.0,
    ylims=None,
    legend_ax_index=1,
    legend_loc="upper right",
    save_path=None,
    show=True,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    # ---- match original behavior exactly ----
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
        # IMPORTANT: avoid any global constrained-layout surprises
        "figure.constrained_layout.use": False,
    })

    # IMPORTANT: explicitly disable constrained_layout
    fig, axes = plt.subplots(
        2, 2,
        figsize=figsize,
        sharex=True,
        constrained_layout=False
    )
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

    # EXACTLY like your original script:
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)  # no bbox_inches="tight"

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

import numpy as np
import matplotlib.pyplot as plt

def plot_combined_rrmse_and_failure_cases(
    *,
    # --- rRMSE settings ---
    root: str,
    gt_method: str,
    suffix: str,
    rep_methods: dict,          # label -> list of folders
    display_method: dict,       # label -> legend name
    metabs,                     # list of (metab_name, mask)
    display_name: dict,         # metab -> title
    time_axis,
    default_colors: dict,       # label -> color
    default_markers: dict,      # label -> marker
    ylims: dict | None = None,  # metab -> (ymin, ymax)   <-- NEW

    # --- spectral failure settings ---
    GT, xy, xT,
    x: int, y: int, z: int,
    t_left: int,
    t_right: int,

    ppm=None,
    ppm_lo: float = 6.0,
    ppm_hi: float = 1.0,
    scale: float = 1e-3,

    figsize=(9.0, 10.8),
    save_path=None,
    show=True,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ---------------- helpers ----------------
    def robust_ylim_upper(arr, p_lo=1.0, p_hi=98.5, pad_up=0.25, pad_lo=0.20):
        lo, hi = np.percentile(arr, [p_lo, p_hi])
        rng = hi - lo
        return lo - pad_lo * rng, hi + pad_up * rng

    def symmetric_ylim(arr, p=98.5, pad=0.15):
        m = np.percentile(np.abs(arr), p)
        return -(1 + pad) * m, (1 + pad) * m

    # ---------------- figure ----------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 1.0, 0.9, 0.45],
        hspace=0.28,
        wspace=0.25
    )

    # ===================== TOP: rRMSE 2x2 =====================
    axes = np.array([
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ])

    lw, ms = 1.8, 3.5

    legend_handles = [
        Line2D(
            [0], [0],
            color=default_colors[k],
            marker=default_markers[k],
            linestyle="-",
            lw=lw,
            ms=ms,
            label=display_method.get(k, k),
        )
        for k in rep_methods.keys()
    ]

    for ax, (metab, mask) in zip(axes, metabs):

        curves = compute_mean_std_rrmse_time(
            root, metab, mask,
            gt_method=gt_method,
            rep_methods=rep_methods,
            suffix=suffix
        )

        xt = np.asarray(time_axis)

        for lab in rep_methods.keys():
            mean_t, _ = curves[lab]
            mean_t = np.asarray(mean_t)

            ax.plot(
                xt, mean_t,
                color=default_colors[lab],
                marker=default_markers[lab],
                linestyle="-",
                lw=lw,
                ms=ms,
                zorder=3,
            )

        ax.set_title(display_name.get(metab, metab))
        ax.grid(True, alpha=0.2)
        ax.set_xticks(xt)

        # --- NEW: ylims override (if provided) ---
        if ylims is not None and metab in ylims:
            ax.set_ylim(*ylims[metab])
        else:
            # fallback: original defaults from your script
            if metab == "water":
                ax.set_ylim(0.03, 0.10)
            elif metab == "Glc":
                ax.set_ylim(0.03, 0.35)

        if ax in (axes[0], axes[2]):
            ax.set_ylabel("rRMSE")

    axes[2].set_xlabel("Time (min)")
    axes[3].set_xlabel("Time (min)")
    axes[1].legend(handles=legend_handles, frameon=False, loc="upper right")

    # ===================== BOTTOM: Failure cases (spectra + residuals) =====================
    if ppm is None:
        ppm = np.linspace(8, 1, GT.shape[3])

    ppm = np.asarray(ppm)
    ppm_mask = (ppm >= ppm_hi) & (ppm <= ppm_lo)
    ppm_w = ppm[ppm_mask]

    # extract spectra
    gt_L  = GT[x, y, z, :, t_left].real[ppm_mask] * scale
    xy_L  = xy[x, y, z, :, t_left].real[ppm_mask] * scale
    res_L = xy_L - gt_L

    gt_R  = GT[x, y, z, :, t_right].real[ppm_mask] * scale
    xT_R  = xT[x, y, z, :, t_right].real[ppm_mask] * scale
    res_R = xT_R - gt_R

    yl_spec_L = robust_ylim_upper(np.concatenate([gt_L, xy_L]))
    yl_spec_R = robust_ylim_upper(np.concatenate([gt_R, xT_R]))
    yl_res_L  = symmetric_ylim(res_L)
    yl_res_R  = symmetric_ylim(res_R)

    axL  = fig.add_subplot(gs[2, 0])
    axR  = fig.add_subplot(gs[2, 1], sharex=axL)
    axLr = fig.add_subplot(gs[3, 0], sharex=axL)
    axRr = fig.add_subplot(gs[3, 1], sharex=axR)

    # LEFT: xy
    axL.plot(ppm_w, gt_L, '--', color='black', lw=1.4, label='GT')
    axL.plot(ppm_w, xy_L, color=default_colors["xy"], lw=1.6, label='xy')
    axL.set_title("Failure case: xy", fontsize=11)
    axL.set_xlim(ppm_lo, ppm_hi)
    axL.set_ylim(*yl_spec_L)
    axL.grid(True, alpha=0.15)
    axL.legend(frameon=False, loc="upper right", fontsize=9)
    axL.set_ylabel("Signal [a.u.]")

    axLr.axhline(0, color='black', lw=0.8, alpha=0.6)
    axLr.plot(ppm_w, res_L, color=default_colors["xy"], lw=1.2)
    axLr.set_ylim(*yl_res_L)
    axLr.grid(True, alpha=0.15)
    axLr.set_xlabel("Chemical shift [ppm]")
    axLr.set_ylabel("Residual")

    # RIGHT: xT
    axR.plot(ppm_w, gt_R, '--', color='black', lw=1.4, label='GT')
    axR.plot(ppm_w, xT_R, color=default_colors["xT"], lw=1.6, label='xT')
    axR.set_title("Failure case: xT", fontsize=11)
    axR.set_xlim(ppm_lo, ppm_hi)
    axR.set_ylim(*yl_spec_R)
    axR.grid(True, alpha=0.15)
    axR.legend(frameon=False, loc="upper right", fontsize=9)

    axRr.axhline(0, color='black', lw=0.8, alpha=0.6)
    axRr.plot(ppm_w, res_R, color=default_colors["xT"], lw=1.2)
    axRr.set_ylim(*yl_res_R)
    axRr.grid(True, alpha=0.15)
    axRr.set_xlabel("Chemical shift [ppm]")

    # remove top x tick labels for the spectra row
    plt.setp(axL.get_xticklabels(), visible=False)
    plt.setp(axR.get_xticklabels(), visible=False)

    fig.align_ylabels()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
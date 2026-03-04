import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import sys

sys.path.append(os.path.abspath("../src"))
from denoising.eval.InVivo import *

FONTSIZE = 16

# -------------------- STYLE / COLORS (paper-like, MRI) --------------------
DEFAULT_COLORS = {
    "noisy": "#333333",          # dark gray
    "tMPPCA_5D": "#E69F00",       # orange
    "deep_tMPPCA_5D": "#009E73",  # green
}
DEFAULT_MARKERS = {
    "noisy": "o",
    "tMPPCA_5D": "s",
    "deep_tMPPCA_5D": "^",
}

def apply_example_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
    })

# --- Paper-like font sizes (replace your old big rcParams) ---
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

def plot_group_fitrate_two_metabs(
    *,
    stats_glc: dict[str, dict[str, np.ndarray]],
    stats_glx: dict[str, dict[str, np.ndarray]],
    methods: list[str],
    labels: list[str],
    minutes_per_rep: float = 7.0,
    ymax: float = 100.0,
    title: str = "",
    savepath: str | None = None,
    colors_by_suffix: dict[str, str] | None = None,
    markers_by_suffix: dict[str, str] | None = None,
    title_left: str = "Glc",
    title_right: str = "Glx",
    legend_loc: str = "lower right",
):
    assert len(methods) == len(labels)

    apply_example_style()

    if colors_by_suffix is None:
        colors_by_suffix = DEFAULT_COLORS
    if markers_by_suffix is None:
        markers_by_suffix = DEFAULT_MARKERS

    # suffix order comes from methods => consistent colors across panels
    method_suffixes = [suffix_from_method(m) for m in methods]
    colors  = [colors_by_suffix.get(suf, "C0") for suf in method_suffixes]
    markers = [markers_by_suffix.get(suf, "o") for suf in method_suffixes]

    # infer T
    T = stats_glc[method_suffixes[0]]["mean"].shape[0]
    time_min = np.arange(2 * minutes_per_rep, minutes_per_rep * (T + 2), minutes_per_rep)

    # style parameters
    lw, ms, cap = 1.8, 3.5, 2
    elw = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    ax_glc, ax_glx = axes

    # --- Left metabolite ---
    for suf, col, mk, lab in zip(method_suffixes, colors, markers, labels):
        y = stats_glc[suf]["mean"]
        e = stats_glc[suf]["std"]

        ax_glc.errorbar(
            time_min, y, yerr=e,
            color=col,
            linestyle="-",
            marker=mk,
            lw=lw, ms=ms,
            capsize=cap,
            elinewidth=elw,
            label=lab
        )

    ax_glc.set_title(title_left)
    ax_glc.set_xlabel("Time (min)")
    ax_glc.set_ylabel("Fit rate [%]")
    ax_glc.grid(True, alpha=0.2)
    ax_glc.set_ylim(0, ymax)
    ax_glc.set_yticks(np.arange(0, ymax + 1, 20))
    ax_glc.set_xticks(time_min)

    # --- Right metabolite ---
    for suf, col, mk, lab in zip(method_suffixes, colors, markers, labels):
        y = stats_glx[suf]["mean"]
        e = stats_glx[suf]["std"]

        ax_glx.errorbar(
            time_min, y, yerr=e,
            color=col,
            linestyle="-",
            marker=mk,
            lw=lw, ms=ms,
            capsize=cap,
            elinewidth=elw,
            label=lab
        )

    ax_glx.set_title(title_right)
    ax_glx.set_xlabel("Time (min)")
    ax_glx.grid(True, alpha=0.2)
    ax_glx.set_ylim(0, ymax)
    ax_glx.set_yticks(np.arange(0, ymax + 1, 20))
    ax_glx.set_xticks(time_min)

    # legend
    handles = [
        Line2D([0], [0], color=colors_by_suffix.get(suf, "k"), lw=lw, label=lab)
        for suf, lab in zip(method_suffixes, labels)
    ]

    ax_glc.legend(handles=handles, frameon=False, loc=legend_loc, fontsize=9)

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if savepath:
        if not savepath.lower().endswith(".pdf"):
            savepath += ".pdf"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"💾 Saved as: {savepath}")

    plt.show()
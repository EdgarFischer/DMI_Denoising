import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_sim_maps_with_per_column_cbars(
    maps,
    errs,
    *,
    t_idx: int,
    z_idx: int,
    methods=("GT", "Proposed", "tMPPCA", "No denoising"),
    metabolites=("Glx", "Lac"),
    cmap_map="magma",
    cmap_err="RdBu_r",
    title=None,
    amp_cbar_labels=("Glx [a.u.]", "Lac [a.u.]"),
    err_cbar_labels=("Glx error [a.u.]", "Lac error [a.u.]"),
    err_abs_percentile=99.9,   # <-- percentile of |error|
    # layout controls
    figsize=None,
    gap=0.06,
    wspace=0.06,
    hspace=0.02,
    cbar_height=0.035,
    cbar_pad=0.008,
    save_path="simulationMaps.pdf",
    show=True,
):
    """
    Error scaling is based on a percentile of |error| pooled across methods.
    v = percentile(|error|, err_abs_percentile)
    error limits are set to (-v, +v)
    """

    def slc4(A):
        return A[:, :, z_idx, t_idx]

    # -------------------------------------------------
    # Amplitude scaling (unchanged: GT-based)
    # -------------------------------------------------
    amp_norm = {}
    for m in metabolites:
        gt_t = maps["GT"][m][:, :, :, t_idx]
        gt_vals = gt_t[np.isfinite(gt_t)]

        if gt_vals.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = np.percentile(gt_vals, 2)
            vmax = np.percentile(gt_vals, 99.5)
            if vmax <= vmin:
                vmax = vmin + 1e-6

        amp_norm[m] = colors.Normalize(vmin=vmin, vmax=vmax)

    # -------------------------------------------------
    # NEW: Error scaling via percentile of |error|
    # -------------------------------------------------
    err_lim = {}
    for m in metabolites:
        abs_vals = []

        for meth in methods:
            if meth == "GT":
                continue

            Em = errs.get(meth, {}).get(m, None)
            if Em is None:
                continue

            Em_t = Em[:, :, :, t_idx]
            Em_t = Em_t[np.isfinite(Em_t)]

            if Em_t.size:
                abs_vals.append(np.abs(Em_t))

        if len(abs_vals) == 0:
            err_lim[m] = (-1.0, 1.0)
            continue

        all_abs = np.concatenate(abs_vals, axis=0)
        v = np.percentile(all_abs, err_abs_percentile)

        if v == 0:
            v = 1e-6

        err_lim[m] = (-v, +v)

    # -------------------------------------------------
    # Figure layout
    # -------------------------------------------------
    nrows = len(methods)
    if figsize is None:
        figsize = (8.8, 1.6 * nrows + 0.9)

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=5,
        width_ratios=[1, 1, gap, 1, 1],
        wspace=wspace,
        hspace=hspace
    )

    col_titles = [
        metabolites[0], metabolites[1], "",
        f"{metabolites[0]} error", f"{metabolites[1]} error"
    ]

    bottom_axes = {}
    mappable_amp = {metabolites[0]: None, metabolites[1]: None}
    mappable_err = {metabolites[0]: None, metabolites[1]: None}

    for r, meth in enumerate(methods):

        # ----- amplitude columns -----
        for c, m in enumerate(metabolites):
            ax = fig.add_subplot(gs[r, c])
            A = slc4(maps[meth][m])

            im = ax.imshow(A.T, origin="lower",
                           cmap=cmap_map, norm=amp_norm[m])

            if mappable_amp[m] is None:
                mappable_amp[m] = im

            if r == 0:
                ax.set_title(col_titles[c], pad=6)

            if c == 0:
                ax.set_ylabel(meth, rotation=90,
                              fontsize=11, labelpad=12)

            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if r == nrows - 1:
                bottom_axes[("amp", m)] = ax

        # gap column
        fig.add_subplot(gs[r, 2]).axis("off")

        # ----- error columns -----
        for c, m in enumerate(metabolites):
            ax = fig.add_subplot(gs[r, c + 3])

            if r == 0:
                ax.set_title(col_titles[c + 3], pad=6)

            Em = errs.get(meth, {}).get(m, None)

            if meth == "GT" or Em is None:
                ax.text(0.5, 0.5, "",
                        ha="center", va="center",
                        transform=ax.transAxes)
            else:
                E = slc4(Em)
                imE = ax.imshow(
                    E.T,
                    origin="lower",
                    cmap=cmap_err,
                    vmin=err_lim[m][0],
                    vmax=err_lim[m][1],
                )

                if mappable_err[m] is None:
                    mappable_err[m] = imE

            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if r == nrows - 1:
                bottom_axes[("err", m)] = ax

    if title is not None:
        fig.text(0.02, 0.98, title,
                 ha="left", va="top", fontsize=13)

    # -------------------------------------------------
    # Colorbars
    # -------------------------------------------------
    def add_cbar_under(ax_ref, mappable, label):
        bbox = ax_ref.get_position()
        cax = fig.add_axes([
            bbox.x0,
            bbox.y0 - cbar_pad - cbar_height,
            bbox.width,
            cbar_height
        ])
        cb = fig.colorbar(mappable, cax=cax,
                          orientation="horizontal")
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    add_cbar_under(bottom_axes[("amp", metabolites[0])],
                   mappable_amp[metabolites[0]],
                   amp_cbar_labels[0])

    add_cbar_under(bottom_axes[("amp", metabolites[1])],
                   mappable_amp[metabolites[1]],
                   amp_cbar_labels[1])

    add_cbar_under(bottom_axes[("err", metabolites[0])],
                   mappable_err[metabolites[0]],
                   err_cbar_labels[0])

    add_cbar_under(bottom_axes[("err", metabolites[1])],
                   mappable_err[metabolites[1]],
                   err_cbar_labels[1])

    fig.subplots_adjust(bottom=0.12)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig

def plot_segmentation_row(
    segs,
    *,
    z_idx: int,
    order=("GM", "WM", "CSF", "Lesion"),
    row_label="Segmentations",
    title=None,
    cmap="gray",
    vmin=0.0,
    vmax=1.0,
    figsize=(8.8, 1.6),
    wspace=0.02,
    hspace=0.02,
    save_path=None,     # <-- neu
    show=True,          # <-- neu
):
    import numpy as np
    import matplotlib.pyplot as plt

    def slc3(A):
        A = np.squeeze(A)
        if A.ndim != 3:
            raise ValueError(f"Expected segmentation with 3 dims after squeeze, got shape {A.shape}")
        return A[:, :, z_idx]

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[1, 1, 1, 1],
        wspace=wspace,
        hspace=hspace
    )

    axes = []
    for c, name in enumerate(order):
        ax = fig.add_subplot(gs[0, c])
        axes.append(ax)

        A = slc3(segs[name])
        ax.set_facecolor("black")
        ax.imshow(A.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(name, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # ---- Zeilenlabel links ----
    if row_label is not None:
        bbox = axes[0].get_position()
        fig.text(
            bbox.x0 - 0.02,
            bbox.y0 + bbox.height / 2,
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11
        )

    if title is not None:
        fig.text(0.02, 0.98, title, ha="left", va="top", fontsize=13)

    # ---- Speichern optional ----
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
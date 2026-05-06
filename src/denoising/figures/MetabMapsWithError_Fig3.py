import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_sim_maps_with_per_column_cbars(
    maps,
    errs=None,
    *,
    t_idx: int,
    z_idx: int,
    methods=("GT", "Proposed", "tMPPCA", "No denoising"),
    metabolites=("Glc", "Glx", "Lac"),
    cmap_map="magma",
    cmap_err="RdBu_r",
    title=None,
    amp_cbar_labels=None,
    err_cbar_labels=None,
    err_abs_percentile=95,
    normalize_to_hdo=False,
    hdo_name="HDO",
    eps=1e-8,
    figsize=None,
    gap=0.06,
    wspace=0.055,
    hspace=0.02,
    cbar_height=0.030,
    cbar_pad=0.007,
    save_path="simulationMaps.pdf",
    show=True,
):

    def slc4(A):
        return A[:, :, z_idx, t_idx]

    metabolites = tuple(metabolites)
    n_metab = len(metabolites)

    if amp_cbar_labels is None:
        if normalize_to_hdo:
            amp_cbar_labels = tuple([f"{m} / HDO" for m in metabolites])
        else:
            amp_cbar_labels = tuple([f"{m} [a.u.]" for m in metabolites])

    if err_cbar_labels is None:
        if normalize_to_hdo:
            err_cbar_labels = tuple([f"{m} / HDO error" for m in metabolites])
        else:
            err_cbar_labels = tuple([f"{m} error [a.u.]" for m in metabolites])

    # -------------------------------------------------
    # Optional HDO normalization
    # -------------------------------------------------
    maps_plot = {}

    for meth in methods:

        maps_plot[meth] = {}

        if normalize_to_hdo:
            hdo = maps[meth][hdo_name]

        for m in metabolites:

            if normalize_to_hdo:

                valid = hdo > 0

                tmp = np.zeros_like(maps[meth][m], dtype=float)

                tmp[valid] = (
                    maps[meth][m][valid]
                    / hdo[valid]
                )

                maps_plot[meth][m] = tmp

            else:
                maps_plot[meth][m] = maps[meth][m]

    # -------------------------------------------------
    # Error maps
    # -------------------------------------------------
    err_maps_plot = {}

    for meth in methods:

        err_maps_plot[meth] = {}

        for m in metabolites:

            if meth == "GT":
                err_maps_plot[meth][m] = None
                continue

            if normalize_to_hdo:

                err_maps_plot[meth][m] = (
                    maps_plot[meth][m]
                    - maps_plot["GT"][m]
                )

            else:

                if errs is None:
                    raise ValueError(
                        "errs must be provided when normalize_to_hdo=False"
                    )

                err_maps_plot[meth][m] = errs.get(meth, {}).get(m, None)

    # -------------------------------------------------
    # Amplitude scaling
    # -------------------------------------------------
    amp_norm = {}

    for m in metabolites:

        gt_t = maps_plot["GT"][m][:, :, :, t_idx]
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
    # Error scaling
    # -------------------------------------------------
    err_lim = {}

    for m in metabolites:

        abs_vals = []

        for meth in methods:

            if meth == "GT":
                continue

            Em = err_maps_plot[meth][m]

            if Em is None:
                continue

            Em_t = Em[:, :, :, t_idx]
            Em_t = Em_t[np.isfinite(Em_t)]

            if Em_t.size:
                abs_vals.append(np.abs(Em_t))

        if len(abs_vals) == 0:

            err_lim[m] = (-1.0, 1.0)

        else:

            v = np.percentile(
                np.concatenate(abs_vals),
                err_abs_percentile
            )

            if v == 0:
                v = 1e-6

            err_lim[m] = (-v, +v)

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    nrows = len(methods)

    if figsize is None:
        figsize = (10.8, 1.55 * nrows + 0.85)

    fig = plt.figure(figsize=figsize)

    width_ratios = [1] * n_metab + [gap] + [1] * n_metab
    gap_col = n_metab

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=2 * n_metab + 1,
        width_ratios=width_ratios,
        wspace=wspace,
        hspace=hspace,
    )

    bottom_axes = {}

    mappable_amp = {m: None for m in metabolites}
    mappable_err = {m: None for m in metabolites}

    for r, meth in enumerate(methods):

        # -----------------------------------------
        # Amplitude columns
        # -----------------------------------------
        for c, m in enumerate(metabolites):

            ax = fig.add_subplot(gs[r, c])

            A = slc4(maps_plot[meth][m])

            im = ax.imshow(
                A.T,
                origin="lower",
                cmap=cmap_map,
                norm=amp_norm[m],
            )

            if mappable_amp[m] is None:
                mappable_amp[m] = im

            if r == 0:

                if normalize_to_hdo:
                    ax.set_title(f"{m} / HDO", pad=6, fontsize=11)
                else:
                    ax.set_title(m, pad=6, fontsize=11)

            if c == 0:
                ax.set_ylabel(
                    meth,
                    rotation=90,
                    fontsize=11,
                    labelpad=12
                )

            ax.set_xticks([])
            ax.set_yticks([])

            for s in ax.spines.values():
                s.set_visible(False)

            if r == nrows - 1:
                bottom_axes[("amp", m)] = ax

        # gap
        fig.add_subplot(gs[r, gap_col]).axis("off")

        # -----------------------------------------
        # Error columns
        # -----------------------------------------
        for c, m in enumerate(metabolites):

            ax = fig.add_subplot(gs[r, gap_col + 1 + c])

            if r == 0:

                if normalize_to_hdo:
                    ax.set_title(
                        f"{m} / HDO error",
                        pad=6,
                        fontsize=11
                    )
                else:
                    ax.set_title(
                        f"{m} error",
                        pad=6,
                        fontsize=11
                    )

            Em = err_maps_plot[meth][m]

            if Em is not None:

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

    # -------------------------------------------------
    # Colorbars
    # -------------------------------------------------
    def add_cbar_under(ax_ref, mappable, label):

        if mappable is None:
            return

        bbox = ax_ref.get_position()

        cax = fig.add_axes([
            bbox.x0,
            bbox.y0 - cbar_pad - cbar_height,
            bbox.width,
            cbar_height,
        ])

        cb = fig.colorbar(
            mappable,
            cax=cax,
            orientation="horizontal"
        )

        cb.set_label(label, fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)

    for i, m in enumerate(metabolites):

        add_cbar_under(
            bottom_axes[("amp", m)],
            mappable_amp[m],
            amp_cbar_labels[i],
        )

    for i, m in enumerate(metabolites):

        add_cbar_under(
            bottom_axes[("err", m)],
            mappable_err[m],
            err_cbar_labels[i],
        )

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
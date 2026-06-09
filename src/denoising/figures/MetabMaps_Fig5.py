# pip install nibabel matplotlib numpy scikit-image
import os, numpy as np, nibabel as nib
from nibabel.processing import resample_from_to
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

def plot_dmi_timecourse_multi(
    BASE_LIST, T_INDEX_LIST, save_path,
    T1_Z=106, DMI_Z=11,
    T1_WINDOW=(1268, 3619),
    LINEWIDTH=1.8, figsize_scale=1.0,
    method_labels=None, rep_labels=None,
    fontsize=24,
    DMI_BOX_I=None, DMI_BOX_J=None, box_linewidth=2.0, box_color='white',
    align_t1_to_dmi=True,
    LAC_CLIM=None, GLX_CLIM=None, clim_percentiles=(1, 99),
    metabolites=("Glc", "Glx", "Lac"),
    overlay_alpha=0.90,
    overlay_mask_mode="below",
    overlay_mask_value=0.0,
    overlay_mask_percentile=5.0,
    tumor_mask_filename="Tumor_1_mask.nii",
    tumor_mask_path=None,
    tumor_color="red",
    tumor_linewidth=2.0,
    show_tumor_legend=True,
    normalize_to_hdo=False,
    hdo_name="HDO",
    eps=1e-8,
):
    """
    Generalized version for >=1 metabolites.
    Example:
        metabolites=("Glc", "Glx", "Lac")

    If normalize_to_hdo=True:
        each metabolite map is shown as metabolite / HDO
        computed voxel-wise after resampling to T1 space.
        Outside valid HDO voxels (HDO <= eps), values are set to 0.
    """

    assert len(BASE_LIST) >= 1
    if method_labels is not None:
        assert len(method_labels) == len(BASE_LIST)
    if rep_labels is not None:
        assert len(rep_labels) == len(T_INDEX_LIST)
    assert isinstance(metabolites, (list, tuple)) and len(metabolites) >= 1

    metabolites = tuple(metabolites)
    n_metab = len(metabolites)
    n_times = len(T_INDEX_LIST)
    nrows = len(BASE_LIST)

    # --- T1 aus erster Methode ---
    base0 = BASE_LIST[0]
    T1_PATH = os.path.join(base0, "maps", "magnitude.nii")

    t1_img = nib.load(T1_PATH)
    t1 = t1_img.get_fdata()
    t1_aff, t1_shape = t1_img.affine, t1_img.shape

    # --- Tumor-Maske laden & auf T1 resamplen ---
    mask_t1_nn = None

    if tumor_mask_path is not None:
        MASK_PATH = tumor_mask_path
    else:
        MASK_PATH = os.path.join(base0, "maps", tumor_mask_filename) if tumor_mask_filename else None

    if MASK_PATH is not None and os.path.exists(MASK_PATH):
        mask_img = nib.load(MASK_PATH)
        mask_t1_nn = resample_from_to(mask_img, (t1_shape, t1_aff), order=0).get_fdata() > 0

    # --- DMI-Z -> T1-Z ---
    def dmi_z_to_t1_z(dmi_img, t1_affine, t1_shape, dmi_z):
        sx, sy, _ = dmi_img.shape
        ijk_dmi = np.array([sx / 2.0, sy / 2.0, dmi_z, 1.0])
        xyz = dmi_img.affine @ ijk_dmi
        ijk_t1 = np.linalg.inv(t1_affine) @ xyz
        return int(np.clip(np.rint(ijk_t1[2]), 0, t1_shape[2] - 1))

    # --- Markierungsbox ---
    def draw_dmi_voxel_box_via_resample(
        ax, dmi_img, i, j, dmi_z, t1_shape, t1_affine,
        t1_slice_index, edgecolor='white', lw=2.0
    ):
        if i is None or j is None:
            return

        shp = dmi_img.shape

        if not (0 <= i < shp[0] and 0 <= j < shp[1] and 0 <= dmi_z < shp[2]):
            return

        mask_dmi = np.zeros(shp, dtype=np.uint8)
        mask_dmi[i, j, dmi_z] = 1

        mask_t1 = resample_from_to(
            nib.Nifti1Image(mask_dmi, dmi_img.affine),
            (t1_shape, t1_aff),
            order=0
        ).get_fdata() > 0

        for c in measure.find_contours(mask_t1[..., t1_slice_index].T.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=edgecolor, linewidth=lw)

    # --- Tumor-Kontur ---
    def draw_tumor_contour(ax, z):
        if mask_t1_nn is None:
            return

        sl = mask_t1_nn[..., z]

        if not np.any(sl):
            return

        for c in measure.find_contours(sl.T.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=tumor_color, linewidth=tumor_linewidth)

    # --- Loader ---
    def load_metab_on_t1(BASE, T_index, metab):
        p = os.path.join(BASE, "maps", f"{T_index}", "Orig", f"{metab}_amp_map.nii")
        img = nib.load(p)
        metab_map = resample_from_to(img, (t1_shape, t1_aff), order=0).get_fdata()

        if not normalize_to_hdo:
            return metab_map

        p_hdo = os.path.join(BASE, "maps", f"{T_index}", "Orig", f"water_amp_map.nii")
        img_hdo = nib.load(p_hdo)
        hdo_map = resample_from_to(img_hdo, (t1_shape, t1_aff), order=0).get_fdata()

        out = np.zeros_like(metab_map, dtype=float)

        valid = (
            np.isfinite(metab_map)
            & np.isfinite(hdo_map)
            & (hdo_map > eps)
        )

        out[valid] = metab_map[valid] / hdo_map[valid]

        return out

    # --- Overlay-Maske ---
    def masked_overlay(vol3d, z, mode="below", value=0.0, percentile=5.0):
        sl = vol3d[..., z]

        if mode == "below":
            m = ~np.isfinite(sl) | (sl <= value)

        elif mode == "percentile":
            finite = sl[np.isfinite(sl)]

            if finite.size == 0:
                m = ~np.isfinite(sl)
            else:
                thr = np.percentile(finite, percentile)
                m = ~np.isfinite(sl) | (sl <= thr)

        else:
            m = ~np.isfinite(sl)

        return np.ma.array(sl, mask=m)

    # --- Referenzbild für z mapping ---
    ref_img_for_z = nib.load(
        os.path.join(
            BASE_LIST[0],
            "maps",
            f"{T_INDEX_LIST[0]}",
            "Orig",
            f"{metabolites[0]}_amp_map.nii"
        )
    )

    T1_Z_MATCH_GLOBAL = dmi_z_to_t1_z(ref_img_for_z, t1_aff, t1_shape, DMI_Z)

    # --- Auto CLIM pro Metabolit ---
    def gather_vals(metab):
        vals = []

        for B in BASE_LIST:
            for T in T_INDEX_LIST:
                a = load_metab_on_t1(B, T, metab)
                vals.append(a[np.isfinite(a)])

        vals = [v for v in vals if v.size > 0]

        if len(vals) == 0:
            return (0.0, 1.0)

        vals = np.concatenate(vals)

        p1, p99 = np.percentile(vals, clim_percentiles)

        if p1 == p99:
            p99 = p1 + 1.0

        return float(p1), float(p99)

    clim_by_metab = {}

    for metab in metabolites:
        # alte Kompatibilität
        if metab == "Lac" and LAC_CLIM is not None:
            clim_by_metab[metab] = LAC_CLIM
        elif metab == "Glx" and GLX_CLIM is not None:
            clim_by_metab[metab] = GLX_CLIM
        else:
            clim_by_metab[metab] = gather_vals(metab)

    # === Layout ===
    # T1 | label | all metabolite/time maps | one colorbar per metabolite
    ncols = 1 + 1 + n_metab * n_times + n_metab

    # gleiche Gesamtbreite wie vorher ungefähr beibehalten
    fig_w = figsize_scale * 24.0
    fig_h = figsize_scale * (nrows * 4.0)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200, constrained_layout=True)
    fig.patch.set_facecolor('white')

    width_ratios = [1, 0.2] + [1] * (n_metab * n_times) + [0.06] * n_metab

    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1] * nrows
    )

    cmap_t1 = mpl.cm.get_cmap("gray").copy()
    cmap_met = {m: mpl.cm.get_cmap("magma") for m in metabolites}
    norm_met = {
        m: mpl.colors.Normalize(
            vmin=clim_by_metab[m][0],
            vmax=clim_by_metab[m][1],
            clip=True
        )
        for m in metabolites
    }

    t1_vmin, t1_vmax = T1_WINDOW

    last_im = {m: None for m in metabolites}

    for r, BASE in enumerate(BASE_LIST):

        ref_img = nib.load(
            os.path.join(
                BASE,
                "maps",
                f"{T_INDEX_LIST[0]}",
                "Orig",
                f"{metabolites[0]}_amp_map.nii"
            )
        )

        T1_Z_MATCH = dmi_z_to_t1_z(ref_img, t1_aff, t1_shape, DMI_Z)

        # T1 only first row
        if r == 0:
            ax_t1 = fig.add_subplot(gs[r, 0])
            ax_t1.axis('off')
            ax_t1.set_facecolor('white')

            zshow = T1_Z_MATCH_GLOBAL if align_t1_to_dmi else T1_Z

            ax_t1.imshow(
                t1[..., zshow].T,
                cmap=cmap_t1,
                vmin=t1_vmin,
                vmax=t1_vmax,
                origin='lower',
                interpolation='nearest'
            )

            draw_tumor_contour(ax_t1, zshow)

            if DMI_BOX_I is not None and DMI_BOX_J is not None:
                draw_dmi_voxel_box_via_resample(
                    ax_t1,
                    ref_img_for_z,
                    DMI_BOX_I,
                    DMI_BOX_J,
                    DMI_Z,
                    t1_shape,
                    t1_aff,
                    zshow,
                    edgecolor=box_color,
                    lw=box_linewidth
                )

        else:
            ax_blank = fig.add_subplot(gs[r, 0])
            ax_blank.axis('off')
            ax_blank.set_facecolor('white')

        # method label
        ax_label = fig.add_subplot(gs[r, 1])
        ax_label.axis('off')
        ax_label.set_facecolor('white')

        label = method_labels[r] if method_labels is not None else os.path.basename(os.path.normpath(BASE))

        ax_label.text(
            0.5,
            0.5,
            label,
            color='black',
            fontsize=fontsize,
            rotation=90,
            ha='center',
            va='center',
            transform=ax_label.transAxes
        )

        # metabolite blocks
        for mi, metab in enumerate(metabolites):

            block_offset = 2 + mi * n_times

            for j, T in enumerate(T_INDEX_LIST):

                ax = fig.add_subplot(gs[r, block_offset + j])
                ax.axis('off')
                ax.set_facecolor('white')

                ax.imshow(
                    t1[..., T1_Z_MATCH].T,
                    cmap=cmap_t1,
                    vmin=t1_vmin,
                    vmax=t1_vmax,
                    origin='lower',
                    interpolation='nearest'
                )

                mvol = load_metab_on_t1(BASE, T, metab)

                msl = masked_overlay(
                    mvol,
                    T1_Z_MATCH,
                    mode=overlay_mask_mode,
                    value=overlay_mask_value,
                    percentile=overlay_mask_percentile
                )

                last_im[metab] = ax.imshow(
                    msl.T,
                    cmap=cmap_met[metab],
                    norm=norm_met[metab],
                    origin='lower',
                    interpolation='nearest',
                    alpha=overlay_alpha
                )

                draw_tumor_contour(ax, T1_Z_MATCH)

                if r == 0:
                    tlabel = rep_labels[j] if rep_labels is not None else f"T={T}"
                    ax.set_title(
                        tlabel,
                        color='black',
                        fontsize=fontsize,
                        pad=4
                    )

    # --- Colorbars ---
    for mi, metab in enumerate(metabolites):

        cax = fig.add_subplot(gs[:, ncols - n_metab + mi])
        cax.set_facecolor('white')

        if last_im[metab] is not None:
            cbar = fig.colorbar(last_im[metab], cax=cax)
            cbar.ax.set_facecolor('white')
            cbar.outline.set_edgecolor('black')
            plt.setp(cbar.ax.get_yticklabels(), color='black', fontsize=fontsize)

            if normalize_to_hdo:  ###!!!!
                cbar.set_label(f"{metab} / HDO", color='black', fontsize=fontsize) ###!!!!
            else:
                cbar.set_label(f"{metab} [a.u.]", color='black', fontsize=fontsize)

    # --- Legend ---
    legend_x, legend_y, legend_dx = 0.02, 0.175, 0.04

    if show_tumor_legend and (mask_t1_nn is not None):
        fig.add_artist(
            plt.Line2D(
                [legend_x, legend_x + legend_dx],
                [legend_y, legend_y],
                color=tumor_color,
                linewidth=tumor_linewidth,
                transform=fig.transFigure,
                figure=fig
            )
        )

        fig.text(
            legend_x + legend_dx + 0.01,
            legend_y,
            "Tumor mask",
            color='black',
            va='center',
            fontsize=fontsize,
            transform=fig.transFigure
        )

        legend_y2 = legend_y - 0.03

    else:
        legend_y2 = legend_y

    fig.add_artist(
        plt.Line2D(
            [legend_x, legend_x + legend_dx],
            [legend_y2, legend_y2],
            color=box_color,
            linewidth=max(box_linewidth, 1.0),
            transform=fig.transFigure,
            figure=fig
        )
    )

    plt.show()
    fig.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())

    return
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
    T1_WINDOW= (1268, 3619),#(0, 225),  #(1268, 3619),
    LINEWIDTH=1.8, figsize_scale=1.0,
    method_labels=None, rep_labels=None,
    fontsize=24,
    DMI_BOX_I=None, DMI_BOX_J=None, box_linewidth=2.0, box_color='white',
    align_t1_to_dmi=True,
    # feste Skalen optional; sonst Auto über alle Methoden/Zeitpunkte (Perzentile)
    LAC_CLIM=None, GLX_CLIM=None, clim_percentiles=(1, 99),
    # frei wählbare Metabolite (Standard: Glc & Glx)
    metabolites=("Glc", "Glx"),
    # Underlay/Overlay Look
    overlay_alpha=0.90,
    overlay_mask_mode="below",   # "below" oder "percentile"
    overlay_mask_value=0.0,      # bei mode="below": maskiert <= overlay_mask_value
    overlay_mask_percentile=5.0, # bei mode="percentile": maskiert unter diesem Perzentil
    # --- NEU: Tumor-Maske ---
    tumor_mask_filename="Tumor_1_mask.nii",  # wird relativ zu BASE/maps/ gesucht
    tumor_mask_path=None,                    # optional: absoluter/relativer Pfad (überschreibt filename)
    tumor_color="red",
    tumor_linewidth=2.0,
    show_tumor_legend=True,
):
    """
    metabolites: (metab1, metab2), z.B. ("Glc","Glx") oder ("Lac","Glx")
    Hinweis: LAC_CLIM / GLX_CLIM wirken auf metab1 / metab2 (Namenskompatibilität).

    Design:
    - Weißer Background (Figure + alle Axes)
    - T1 als Underlay, Metab-Map als (maskiertes) Overlay, damit T1 sichtbar bleibt
    - Optional: Tumor-Maske als Kontur (resampled auf T1-Raster)
    """

    assert len(BASE_LIST) >= 1
    if method_labels is not None: assert len(method_labels) == len(BASE_LIST)
    if rep_labels is not None:    assert len(rep_labels) == len(T_INDEX_LIST)
    assert isinstance(metabolites, (list, tuple)) and len(metabolites) == 2
    MET1, MET2 = metabolites[0], metabolites[1]

    # --- T1 aus erster Methode ---
    base0 = BASE_LIST[0]
    T1_PATH = os.path.join(base0, "maps", "magnitude.nii")#"flair_coreg.nii")    #"magnitude.nii")

    t1_img = nib.load(T1_PATH)
    t1 = t1_img.get_fdata()
    t1_aff, t1_shape = t1_img.affine, t1_img.shape

    # --- Tumor-Maske laden & auf T1 resamplen (NN) ---
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
        ijk_dmi = np.array([sx/2.0, sy/2.0, dmi_z, 1.0])
        xyz = dmi_img.affine @ ijk_dmi
        ijk_t1 = np.linalg.inv(t1_affine) @ xyz
        return int(np.clip(np.rint(ijk_t1[2]), 0, t1_shape[2]-1))

    # --- Markierungsbox (affine-sicher) ---
    def draw_dmi_voxel_box_via_resample(ax, dmi_img, i, j, dmi_z, t1_shape, t1_affine,
                                        t1_slice_index, edgecolor='white', lw=2.0):
        if i is None or j is None: return
        shp = dmi_img.shape
        if not (0 <= i < shp[0] and 0 <= j < shp[1] and 0 <= dmi_z < shp[2]): return
        mask_dmi = np.zeros(shp, dtype=np.uint8); mask_dmi[i, j, dmi_z] = 1
        mask_t1 = resample_from_to(nib.Nifti1Image(mask_dmi, dmi_img.affine),
                                   (t1_shape, t1_aff), order=0).get_fdata() > 0
        for c in measure.find_contours(mask_t1[..., t1_slice_index].T.astype(float), 0.5):
            ax.plot(c[:,1], c[:,0], color=edgecolor, linewidth=lw)

    # --- Tumor-Kontur plotten (falls vorhanden) ---
    def draw_tumor_contour(ax, z):
        if mask_t1_nn is None: 
            return
        sl = mask_t1_nn[..., z]
        if not np.any(sl):
            return
        for c in measure.find_contours(sl.T.astype(float), 0.5):
            ax.plot(c[:,1], c[:,0], color=tumor_color, linewidth=tumor_linewidth)

    # --- Loader für Metabolit (auf T1-Raster) ---
    def load_metab_on_t1(BASE, T_index, metab):
        p = os.path.join(BASE, "maps", f"{T_index}", "Orig", f"{metab}_amp_map.nii")
        img = nib.load(p)
        return resample_from_to(img, (t1_shape, t1_aff), order=0).get_fdata()

    # --- Overlay-Maske, damit T1 sichtbar bleibt ---
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

    # globale DMI→T1-Slice (nutze MET1 als Referenz)
    ref_img_for_z = nib.load(os.path.join(BASE_LIST[0], "maps", f"{T_INDEX_LIST[0]}", "Orig", f"{MET1}_amp_map.nii"))
    T1_Z_MATCH_GLOBAL = dmi_z_to_t1_z(ref_img_for_z, t1_aff, t1_shape, DMI_Z)

    # --- Auto-CLIMs ---
    def gather_vals(metab):
        vals = []
        for B in BASE_LIST:
            for T in T_INDEX_LIST:
                a = load_metab_on_t1(B, T, metab)
                vals.append(a[np.isfinite(a)])
        vals = np.concatenate(vals) if len(vals) > 1 else vals[0]
        p1, p99 = np.percentile(vals, clim_percentiles)
        if p1 == p99: p99 = p1 + 1.0
        return (float(p1), float(p99))

    M1_CLIM = LAC_CLIM if LAC_CLIM is not None else gather_vals(MET1)
    M2_CLIM = GLX_CLIM if GLX_CLIM is not None else gather_vals(MET2)

    # === Layout: T1 | Label | (alle MET1) | (alle MET2) | Cbar MET1 | Cbar MET2 ===
    n_times = len(T_INDEX_LIST)
    nrows = len(BASE_LIST)
    ncols = 1 + 1 + n_times + n_times + 2

    fig_w = figsize_scale * (ncols * 3.0)
    fig_h = figsize_scale * (nrows * 4.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200, constrained_layout=True)

    # weißer Hintergrund (wie in deinem healthy design)
    fig.patch.set_facecolor('white')

    width_ratios = [1, 0.2] + [1]*n_times + [1]*n_times + [0.06, 0.06]
    gs = GridSpec(nrows, ncols, figure=fig, width_ratios=width_ratios, height_ratios=[1]*nrows)

    cmap_t1  = mpl.cm.get_cmap("gray").copy()
    cmap_m1 = mpl.cm.get_cmap("magma")
    cmap_m2 = mpl.cm.get_cmap("magma")
    norm_m1 = mpl.colors.Normalize(vmin=M1_CLIM[0], vmax=M1_CLIM[1], clip=True)
    norm_m2 = mpl.colors.Normalize(vmin=M2_CLIM[0], vmax=M2_CLIM[1], clip=True)
    t1_vmin, t1_vmax = T1_WINDOW

    last_im_m1 = None
    last_im_m2 = None

    for r, BASE in enumerate(BASE_LIST):
        ref_img = nib.load(os.path.join(BASE, "maps", f"{T_INDEX_LIST[0]}", "Orig", f"{MET1}_amp_map.nii"))
        T1_Z_MATCH = dmi_z_to_t1_z(ref_img, t1_aff, t1_shape, DMI_Z)

        # T1 (nur erste Zeile)
        if r == 0:
            ax_t1 = fig.add_subplot(gs[r, 0]); ax_t1.axis('off')
            ax_t1.set_facecolor('white')
            zshow = T1_Z_MATCH_GLOBAL if align_t1_to_dmi else T1_Z
            ax_t1.imshow(t1[..., zshow].T, cmap=cmap_t1, vmin=t1_vmin, vmax=t1_vmax,
                         origin='lower', interpolation='nearest')
            # Tumor-Kontur auf T1
            draw_tumor_contour(ax_t1, zshow)

            if DMI_BOX_I is not None and DMI_BOX_J is not None:
                draw_dmi_voxel_box_via_resample(ax_t1, ref_img_for_z, DMI_BOX_I, DMI_BOX_J, DMI_Z,
                                                t1_shape, t1_aff, zshow, edgecolor=box_color, lw=box_linewidth)
        else:
            ax_blank = fig.add_subplot(gs[r, 0]); ax_blank.axis('off'); ax_blank.set_facecolor('white')

        # vertikales Methodenlabel
        ax_label = fig.add_subplot(gs[r, 1]); ax_label.axis('off'); ax_label.set_facecolor('white')
        label = method_labels[r] if method_labels is not None else os.path.basename(os.path.normpath(BASE))
        ax_label.text(0.5, 0.5, label, color='black', fontsize=fontsize,
                      rotation=90, ha='center', va='center', transform=ax_label.transAxes)

        # --- MET1 ---
        for j, T in enumerate(T_INDEX_LIST):
            ax = fig.add_subplot(gs[r, 2 + j]); ax.axis('off'); ax.set_facecolor('white')

            ax.imshow(t1[..., T1_Z_MATCH].T, cmap=cmap_t1, vmin=t1_vmin, vmax=t1_vmax,
                      origin='lower', interpolation='nearest')

            m1 = load_metab_on_t1(BASE, T, MET1)
            m1_sl = masked_overlay(m1, T1_Z_MATCH, mode=overlay_mask_mode,
                                   value=overlay_mask_value, percentile=overlay_mask_percentile)
            last_im_m1 = ax.imshow(m1_sl.T, cmap=cmap_m1, norm=norm_m1,
                                   origin='lower', interpolation='nearest', alpha=overlay_alpha)

            # Tumor-Kontur
            draw_tumor_contour(ax, T1_Z_MATCH)

            if r == 0:
                tlabel = rep_labels[j] if rep_labels is not None else f"T={T}"
                ax.set_title(f"{tlabel} — {MET1}", color='black', fontsize=fontsize, pad=4)

        # --- MET2 ---
        m2_offset = 2 + n_times
        for j, T in enumerate(T_INDEX_LIST):
            ax = fig.add_subplot(gs[r, m2_offset + j]); ax.axis('off'); ax.set_facecolor('white')

            ax.imshow(t1[..., T1_Z_MATCH].T, cmap=cmap_t1, vmin=t1_vmin, vmax=t1_vmax,
                      origin='lower', interpolation='nearest')

            m2 = load_metab_on_t1(BASE, T, MET2)
            m2_sl = masked_overlay(m2, T1_Z_MATCH, mode=overlay_mask_mode,
                                   value=overlay_mask_value, percentile=overlay_mask_percentile)
            last_im_m2 = ax.imshow(m2_sl.T, cmap=cmap_m2, norm=norm_m2,
                                   origin='lower', interpolation='nearest', alpha=overlay_alpha)

            # Tumor-Kontur
            draw_tumor_contour(ax, T1_Z_MATCH)

            if r == 0:
                tlabel = rep_labels[j] if rep_labels is not None else f"T={T}"
                ax.set_title(f"{tlabel} — {MET2}", color='black', fontsize=fontsize, pad=4)

    # --- zwei globale Colorbars ---
    cax_m1 = fig.add_subplot(gs[:, ncols-2]); cax_m1.set_facecolor('white')
    if last_im_m1 is not None:
        cbar_m1 = fig.colorbar(last_im_m1, cax=cax_m1)
        cbar_m1.ax.set_facecolor('white')
        cbar_m1.outline.set_edgecolor('black')
        plt.setp(cbar_m1.ax.get_yticklabels(), color='black', fontsize=fontsize)
        cbar_m1.set_label(f"{MET1} [a.u.]", color='black', fontsize=fontsize)

    cax_m2 = fig.add_subplot(gs[:, ncols-1]); cax_m2.set_facecolor('white')
    if last_im_m2 is not None:
        cbar_m2 = fig.colorbar(last_im_m2, cax=cax_m2)
        cbar_m2.ax.set_facecolor('white')
        cbar_m2.outline.set_edgecolor('black')
        plt.setp(cbar_m2.ax.get_yticklabels(), color='black', fontsize=fontsize)
        cbar_m2.set_label(f"{MET2} [a.u.]", color='black', fontsize=fontsize)

    # --- Legende unten links (Tumor + Marked voxel) ---
    legend_x, legend_y, legend_dx = 0.02, 0.175, 0.04

    if show_tumor_legend and (mask_t1_nn is not None):
        fig.add_artist(plt.Line2D([legend_x, legend_x + legend_dx],
                                  [legend_y, legend_y],
                                  color=tumor_color, linewidth=tumor_linewidth,
                                  transform=fig.transFigure, figure=fig))
        fig.text(legend_x + legend_dx + 0.01, legend_y, "Tumor mask",
                 color='black', va='center', fontsize=fontsize, transform=fig.transFigure)

        # Abstand nach unten für voxel-legende
        legend_y2 = legend_y - 0.03
    else:
        legend_y2 = legend_y

    # Marked voxel (falls genutzt)
    fig.add_artist(plt.Line2D([legend_x, legend_x + legend_dx],
                              [legend_y2, legend_y2],
                              color=box_color, linewidth=max(box_linewidth, 1.0),
                              transform=fig.transFigure, figure=fig))
    # optional Text (wenn du ihn willst, einfach entkommentieren)
    # fig.text(legend_x + legend_dx + 0.01, legend_y2, "Marked voxel",
    #          color='black', va='center', fontsize=fontsize, transform=fig.transFigure)

    plt.show()
    fig.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
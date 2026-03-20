import numpy as np
import matplotlib.pyplot as plt

def plot_acf_comparison(
    acf_stats_by_method,
    methods=None,
    axis_indices=None,
    axis_names=None,
    skip_lag0=True,
    show_std=True,
    title=None,
    figsize=(12, 4)
):
    """
    Compare ACFs across methods for all or selected axes.

    Parameters
    ----------
    acf_stats_by_method : dict
        {
            "Method name": (mean_acfs, std_acfs),
            ...
        }

    methods : list of str, optional
        Methods to plot
    axis_indices : list of int, optional
        Which axes to plot (default: all available)
    axis_names : list of str, optional
        Names of axes (e.g. ["x","y","z","t","T"])
    skip_lag0 : bool
    show_std : bool
    title : str, optional
    figsize : tuple
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if methods is None:
        methods = list(acf_stats_by_method.keys())

    # 👉 infer available axes from first method
    first_method = methods[0]
    mean_acfs_first, _ = acf_stats_by_method[first_method]
    available_axes = sorted(mean_acfs_first.keys())

    # 👉 default: plot ALL axes
    if axis_indices is None:
        axis_indices = available_axes

    # 👉 axis names handling
    if axis_names is None:
        axis_names = [f"axis {i}" for i in axis_indices]
    else:
        axis_names = [axis_names[i] for i in axis_indices]

    n_plots = len(axis_indices)

    # 👉 dynamic figure width
    max_width = figsize[0]
    width_per_plot = max_width / 2
    fig_width = width_per_plot * n_plots

    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, figsize[1]))

    if n_plots == 1:
        axes = [axes]

    # 🔁 plotting
    for plot_idx, ax_i in enumerate(axis_indices):
        ax = axes[plot_idx]
        axis_name = axis_names[plot_idx]

        any_plotted = False

        for method in methods:
            if method not in acf_stats_by_method:
                raise ValueError(f"Method '{method}' not found")

            mean_acfs, std_acfs = acf_stats_by_method[method]

            if ax_i not in mean_acfs:
                continue

            mean_acf = mean_acfs[ax_i]
            std_acf = std_acfs[ax_i]

            start = 1 if skip_lag0 else 0
            lags = np.arange(start, len(mean_acf))

            if len(lags) == 0:
                continue

            ax.plot(lags, mean_acf[start:], 'o-', label=method)
            any_plotted = True

            if show_std:
                ax.fill_between(
                    lags,
                    mean_acf[start:] - std_acf[start:],
                    mean_acf[start:] + std_acf[start:],
                    alpha=0.2
                )

        ax.set_xlabel(f"lag in {axis_name}")
        ax.set_ylabel("autocorrelation")
        ax.set_title(f"ACF along {axis_name}")
        ax.grid(True)

        if any_plotted:
            ax.legend()

    if title is not None:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_spatial_acf_comparison(
    spatial_stats_by_method,
    methods=None,
    axes_to_plot=("x", "y", "z"),
    skip_lag0=True,
    show_std=True,
    title=None,
    figsize=(12, 4),
    max_lag_plot=None   # 👈 NEU
):
    """
    Vergleicht räumliche ACFs (x, y, z) mehrerer Methoden.

    Parameters
    ----------
    spatial_stats_by_method : dict
    methods : list of str, optional
    axes_to_plot : tuple of str
    skip_lag0 : bool
    show_std : bool
    title : str, optional
    figsize : tuple
    max_lag_plot : int or None
        Maximale Anzahl Lags die geplottet werden (z. B. 6 oder 8).
        Wenn None → alle Lags.
    """

    if methods is None:
        methods = list(spatial_stats_by_method.keys())

    axes_to_plot = list(axes_to_plot)
    n_plots = len(axes_to_plot)

    max_width = figsize[0]
    width_per_plot = max_width / 3
    fig_width = width_per_plot * n_plots

    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, figsize[1]))

    if n_plots == 1:
        axes = [axes]

    for i, ax_name in enumerate(axes_to_plot):
        ax = axes[i]

        any_plotted = False

        for method in methods:
            if method not in spatial_stats_by_method:
                raise ValueError(f"Method '{method}' not found")

            mean_corrs, std_corrs = spatial_stats_by_method[method]

            if ax_name not in mean_corrs:
                continue

            mean_acf = mean_corrs[ax_name]
            std_acf = std_corrs[ax_name]

            start = 0 if skip_lag0 else 0

            N = len(mean_acf)

            if max_lag_plot is None:
                end = N
            else:
                end = min(max_lag_plot, N)

            lags = np.arange(start, end)

            ax.plot(lags, mean_acf[start:end], 'o-', label=method)
            any_plotted = True

            if show_std:
                ax.fill_between(
                    lags,
                    mean_acf[start:end] - std_acf[start:end],
                    mean_acf[start:end] + std_acf[start:end],
                    alpha=0.2
                )

        ax.set_xlabel(f"lag in {ax_name}")
        ax.set_ylabel("autocorrelation")
        ax.set_title(f"Spatial ACF along {ax_name}")
        ax.grid(True)

        if any_plotted:
            ax.legend()

    if title is not None:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_averaged_signal_comparison(
    results,
    methods=None,
    show_std=True,
    title="Averaged signal in all voxels vs. noise-dominated voxels",
    figsize=(12, 8),
):
    """
    Plottet für jede Methode den Betrag des gemittelten Signals
    in all voxels vs. noise-dominated voxels, entlang der in
    results["extra_axes"] angegebenen Nicht-Raum-Achsen.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    datasets_by_method = results["datasets_by_method"]
    masks_by_method = results["masks_by_method"]
    extra_axes = results["extra_axes"]

    if methods is None:
        methods = list(datasets_by_method.keys())

    n_methods = len(methods)
    n_axes = len(extra_axes)

    if n_methods == 0:
        raise ValueError("No methods to plot.")
    if n_axes == 0:
        raise ValueError("results['extra_axes'] is empty.")

    # --- FIX: fully adaptive figure size ---
    base_w, base_h = figsize

    fig_width = base_w * n_axes / 2
    fig_height = base_h * n_methods / 2

    fig, axes = plt.subplots(
        n_methods,
        n_axes,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    for row, method in enumerate(methods):
        if method not in datasets_by_method:
            raise ValueError(f"Method '{method}' not found in results['datasets_by_method']")
        if method not in masks_by_method:
            raise ValueError(f"Method '{method}' not found in results['masks_by_method']")

        dataset_list = datasets_by_method[method]
        mask_list = masks_by_method[method]

        if len(dataset_list) != len(mask_list):
            raise ValueError(f"Dataset/mask length mismatch for method '{method}'")

        per_axis_all = {ax_name: [] for ax_name in extra_axes}
        per_axis_noise = {ax_name: [] for ax_name in extra_axes}

        for data, mask_noise in zip(dataset_list, mask_list):
            if mask_noise.shape != data.shape[:3]:
                raise ValueError("mask_noise must have shape data.shape[:3]")

            noise_data = data[mask_noise]

            if noise_data.shape[0] == 0:
                continue

            if data.ndim == 4:
                if "t" not in extra_axes:
                    raise ValueError("For 4D data, extra_axes must contain 't'.")

                mean_noise_t = np.abs(np.mean(noise_data, axis=0))
                mean_all_t = np.abs(np.mean(data, axis=(0, 1, 2)))

                per_axis_noise["t"].append(mean_noise_t)
                per_axis_all["t"].append(mean_all_t)

            elif data.ndim == 5:
                if "t" in extra_axes:
                    mean_noise_t = np.abs(np.mean(noise_data, axis=(0, 2)))
                    mean_all_t = np.abs(np.mean(data, axis=(0, 1, 2, 4)))
                    per_axis_noise["t"].append(mean_noise_t)
                    per_axis_all["t"].append(mean_all_t)

                if "T" in extra_axes:
                    mean_noise_T = np.abs(np.mean(noise_data, axis=(0, 1)))
                    mean_all_T = np.abs(np.mean(data, axis=(0, 1, 2, 3)))
                    per_axis_noise["T"].append(mean_noise_T)
                    per_axis_all["T"].append(mean_all_T)

            else:
                raise ValueError(f"Unsupported data.ndim={data.ndim}. Expected 4 or 5.")

        for col, ax_name in enumerate(extra_axes):
            ax = axes[row, col]

            all_curves = per_axis_all[ax_name]
            noise_curves = per_axis_noise[ax_name]

            if len(all_curves) == 0 or len(noise_curves) == 0:
                ax.set_visible(False)
                continue

            min_len_all = min(len(a) for a in all_curves)
            min_len_noise = min(len(a) for a in noise_curves)
            min_len = min(min_len_all, min_len_noise)

            all_stack = np.stack([a[:min_len] for a in all_curves], axis=0)
            noise_stack = np.stack([a[:min_len] for a in noise_curves], axis=0)

            mean_all = np.mean(all_stack, axis=0)
            std_all = np.std(all_stack, axis=0)

            mean_noise = np.mean(noise_stack, axis=0)
            std_noise = np.std(noise_stack, axis=0)

            x = np.arange(min_len)

            ax.plot(x, mean_all, 'o-', label="all voxels")
            ax.plot(x, mean_noise, 'o-', label="noise voxels")

            if show_std:
                ax.fill_between(x, mean_all - std_all, mean_all + std_all, alpha=0.2)
                ax.fill_between(x, mean_noise - std_noise, mean_noise + std_noise, alpha=0.2)

            xlabel = "FID point t" if ax_name == "t" else ax_name
            if ax_name == "T":
                xlabel = "Repetition T"

            ax.set_xlabel(xlabel)
            ax.set_ylabel("|mean signal|")
            ax.set_title(f"{method} — along {ax_name}")
            ax.grid(True)
            ax.legend()

    if title is not None:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    plt.show()

def plot_mean_fid_with_noise_mask(
    results,
    methods=None,
    fid_axis=3,
    show_std=True,
    title="Mean FID with noise mask",
    figsize=(8, 4),
):
    """
    Plottet den gemittelten FID-Betrag aus results und markiert die verwendete Noise-Maske.

    WICHTIG:
    Es wird erst über alle anderen Dimensionen gemittelt und DANACH der Betrag genommen:
        curve(t) = | mean(data over all axes except fid_axis) |

    Parameters
    ----------
    results : dict
        Ausgabe von run_noise_analysis_pipeline(...)
    methods : list of str or None
        Welche Methoden geplottet werden sollen. Falls None: alle.
    fid_axis : int
        Index der FID-Achse, standardmäßig 3
    show_std : bool
        Ob ±1 std über Datensätze geplottet werden soll
    title : str
    figsize : tuple
        Basisgröße pro Plot
    """

    datasets_by_method = results["datasets_by_method"]
    masks_by_method = results["masks_by_method"]

    if methods is None:
        methods = list(datasets_by_method.keys())

    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_methods, 1,
        figsize=(figsize[0], figsize[1] * n_methods),
        squeeze=False
    )

    axes = axes[:, 0]

    for row, method in enumerate(methods):
        ax = axes[row]

        if method not in datasets_by_method:
            raise ValueError(f"Method '{method}' not found in results['datasets_by_method']")
        if method not in masks_by_method:
            raise ValueError(f"Method '{method}' not found in results['masks_by_method']")

        dataset_list = datasets_by_method[method]
        mask_list = masks_by_method[method]

        if len(dataset_list) != len(mask_list):
            raise ValueError(f"Dataset/mask length mismatch for method '{method}'")

        curves = []
        mask_profiles = []

        for data, mask in zip(dataset_list, mask_list):
            if data.shape != mask.shape:
                raise ValueError("mask must have same shape as data")

            if fid_axis >= data.ndim:
                raise ValueError(
                    f"fid_axis={fid_axis} invalid for data.ndim={data.ndim}"
                )

            mean_axes = tuple(ax_i for ax_i in range(data.ndim) if ax_i != fid_axis)

            # erst mitteln, dann Betrag
            curve = np.abs(np.mean(data, axis=mean_axes))
            curves.append(curve)

            # mask auf 1D entlang FID reduzieren
            # bei deinen broadcasteten Masken sollte das exakt 0/1 sein
            mask_profile = np.mean(mask.astype(float), axis=mean_axes)
            mask_profiles.append(mask_profile)

        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        mask_profiles = [m[:min_len] for m in mask_profiles]

        curve_stack = np.stack(curves, axis=0)
        mask_stack = np.stack(mask_profiles, axis=0)

        mean_curve = np.mean(curve_stack, axis=0)
        std_curve = np.std(curve_stack, axis=0)

        # mittlere Maskenbelegung über Datensätze
        mean_mask_profile = np.mean(mask_stack, axis=0)

        # robust zu kleinen numerischen Effekten
        mask_1d = mean_mask_profile > 0.5

        x = np.arange(min_len)

        # Noise-Regionen hinterlegen
        segments = _find_true_segments(mask_1d)
        first_segment = True
        for start, end in segments:
            label = "noise mask" if first_segment else None
            ax.axvspan(start, end - 1, alpha=0.18, label=label)
            ax.axvline(start, linestyle="--", alpha=0.7)
            ax.axvline(end - 1, linestyle="--", alpha=0.7)
            first_segment = False

        ax.plot(x, mean_curve, "o-", label="|mean FID|")

        if show_std:
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2
            )

        ax.set_xlabel("FID point t")
        ax.set_ylabel("|mean signal|")
        ax.set_title(method)
        ax.grid(True)
        ax.legend()

    if title is not None:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    plt.show()

def _find_true_segments(mask_1d):
    """
    Findet zusammenhängende True-Segmente in einer 1D-Bool-Maske.

    Returns
    -------
    segments : list of tuple
        Liste von (start, end)-Intervallen, wobei end exklusiv ist.
    """
    mask_1d = np.asarray(mask_1d, dtype=bool)
    segments = []

    in_segment = False
    start = None

    for i, val in enumerate(mask_1d):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(mask_1d)))

    return segments

def plot_pairwise_correlations(results, method=None, panel_size=3.0):
    """
    Plot mean 2D pairwise autocorrelation maps for all computed axis pairs.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    available_methods = list(results["pair_corr_stats_by_method"].keys())

    if len(available_methods) == 0:
        raise ValueError("No pairwise correlation results found.")

    if method is None:
        method = available_methods[0]
        print(f"Using method: {method}")

    if method not in available_methods:
        raise ValueError(
            f"Method '{method}' not found. Available: {available_methods}"
        )

    mean_pair, _ = results["pair_corr_stats_by_method"][method]

    if len(mean_pair) == 0:
        raise ValueError(f"No pairwise correlations stored for method '{method}'.")

    axis_names_list = results.get("axis_names", None)
    if axis_names_list is None:
        raise ValueError("results does not contain 'axis_names'.")

    axis_names = {i: name for i, name in enumerate(axis_names_list)}

    pairs = sorted(mean_pair.keys())

    titles = [f"({axis_names[a1]}, {axis_names[a2]})" for a1, a2 in pairs]

    axis_labels = {
        (a1, a2): (rf"$\Delta {axis_names[a2]}$", rf"$\Delta {axis_names[a1]}$")
        for a1, a2 in pairs
    }

    vmin = min(np.nanmin(mean_pair[p]) for p in pairs)
    vmax = max(np.nanmax(mean_pair[p]) for p in pairs)

    n_plots = len(pairs)

    if n_plots <= 3:
        ncols = n_plots
    elif n_plots <= 6:
        ncols = 3
    else:
        ncols = 4

    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size * ncols, panel_size * nrows * 0.9),
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.ravel()
    ims = []

    for ax, pair, title in zip(axes_flat, pairs, titles):
        corr = mean_pair[pair]
        xlabel, ylabel = axis_labels[pair]

        im = ax.imshow(
            corr,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.set_title(title, fontsize=12, pad=2)
        ax.set_xlabel(xlabel, fontsize=11, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=2)
        ax.tick_params(axis="both", pad=1)

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle(method, fontsize=14, y=0.98)
    fig.subplots_adjust(right=0.88, top=0.90, wspace=0.35, hspace=0.75)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label("Correlation", fontsize=11)

    plt.show()
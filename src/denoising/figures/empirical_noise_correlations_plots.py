import numpy as np
import matplotlib.pyplot as plt

def plot_acf_comparison(
    acf_stats_by_method,
    methods=None,
    axis_idx=0,
    axis_names=None,
    skip_lag0=True,
    show_std=True,
    title=None,
    figsize=(12, 4)
):
    """
    Vergleicht ACFs mehrerer Methoden in einem oder mehreren gemeinsamen Plots.

    Parameters
    ----------
    acf_stats_by_method : dict
        {
            "Method name": (mean_acfs, std_acfs),
            ...
        }
    methods : list of str, optional
        Welche Methoden geplottet werden sollen
    axis_idx : int or list of int
        Welche Achse(n) geplottet werden sollen
    axis_names : list of str, optional
        Namen der Achsen (z. B. ["t", "T"])
    skip_lag0 : bool
    show_std : bool
    title : str, optional
    figsize : tuple
        Maximale Figure-Größe für zwei Plots
    """

    if methods is None:
        methods = list(acf_stats_by_method.keys())

    # int → list
    if isinstance(axis_idx, int):
        axis_indices = [axis_idx]
    else:
        axis_indices = list(axis_idx)

    n_plots = len(axis_indices)

    # Default axis names
    if axis_names is None:
        axis_names = [f"axis {i}" for i in axis_indices]

    # 👉 Dynamische Breite (1 Plot = halbe Breite)
    max_width = figsize[0]
    width_per_plot = max_width / 2
    fig_width = width_per_plot * n_plots

    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, figsize[1]))

    if n_plots == 1:
        axes = [axes]

    for plot_idx, ax_i in enumerate(axis_indices):
        ax = axes[plot_idx]
        axis_name = axis_names[plot_idx] if plot_idx < len(axis_names) else f"axis {ax_i}"

        any_plotted = False

        for method in methods:
            if method not in acf_stats_by_method:
                raise ValueError(f"Method '{method}' not found")

            mean_acfs, std_acfs = acf_stats_by_method[method]

            if ax_i >= len(mean_acfs):
                continue

            mean_acf = mean_acfs[ax_i]
            std_acf = std_acfs[ax_i]

            start = 0 if skip_lag0 else 0
            lags = np.arange(start, len(mean_acf))

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

    Für 5D-Daten mit extra_axes=["t", "T"]:
        - entlang t: Mittelung über Raum + T
        - entlang T: Mittelung über Raum + t

    Für 4D-Daten mit extra_axes=["t"]:
        - entlang t: Mittelung über Raum

    Parameters
    ----------
    results : dict
        Ausgabe von run_noise_analysis_pipeline(...)
    methods : list of str, optional
        Welche Methoden geplottet werden sollen.
        Falls None, werden alle Methoden verwendet.
    show_std : bool
        Ob Standardabweichungen über Subjects als Schattierung
        dargestellt werden sollen.
    title : str
        Figure-Titel.
    figsize : tuple
        Basisgröße der Figure.
    """

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

    fig, axes = plt.subplots(
        n_methods,
        n_axes,
        figsize=(figsize[0], max(figsize[1], 3 * n_methods)),
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

            noise_data = data[mask_noise]  # shape: (N_voxels, ...)

            if noise_data.shape[0] == 0:
                continue

            # 4D: (x, y, z, t)
            # noise_data -> (N_voxels, t)
            if data.ndim == 4:
                if "t" not in extra_axes:
                    raise ValueError("For 4D data, extra_axes must contain 't'.")

                mean_noise_t = np.abs(np.mean(noise_data, axis=0))       # (t,)
                mean_all_t = np.abs(np.mean(data, axis=(0, 1, 2)))       # (t,)

                per_axis_noise["t"].append(mean_noise_t)
                per_axis_all["t"].append(mean_all_t)

            # 5D: (x, y, z, t, T)
            # noise_data -> (N_voxels, t, T)
            elif data.ndim == 5:
                if "t" in extra_axes:
                    mean_noise_t = np.abs(np.mean(noise_data, axis=(0, 2)))   # (t,)
                    mean_all_t = np.abs(np.mean(data, axis=(0, 1, 2, 4)))     # (t,)
                    per_axis_noise["t"].append(mean_noise_t)
                    per_axis_all["t"].append(mean_all_t)

                if "T" in extra_axes:
                    mean_noise_T = np.abs(np.mean(noise_data, axis=(0, 1)))   # (T,)
                    mean_all_T = np.abs(np.mean(data, axis=(0, 1, 2, 3)))     # (T,)
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
                ax.fill_between(
                    x,
                    mean_all - std_all,
                    mean_all + std_all,
                    alpha=0.2
                )
                ax.fill_between(
                    x,
                    mean_noise - std_noise,
                    mean_noise + std_noise,
                    alpha=0.2
                )

            xlabel = "FID point t" if ax_name == "t" else f"{ax_name}"
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
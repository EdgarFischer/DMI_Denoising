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
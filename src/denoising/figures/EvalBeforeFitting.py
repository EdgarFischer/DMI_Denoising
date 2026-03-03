import numpy as np
import matplotlib.pyplot as plt

def plot_z_slices(
    Data,
    Titles,
    t,
    T,
    cmap="viridis",
    share_clim="per_column",  # "none", "per_column", "global"
):
    """
    Data   : list of arrays with shape (x, y, z, t, T)
    Titles : list of strings (same length as Data)
    t, T   : indices
    """

    n_cols = len(Data)
    n_slices = Data[0].shape[2]

    # --- Figure ---
    fig, axes = plt.subplots(
        n_slices,
        n_cols,
        figsize=(3 * n_cols, 2.5 * n_slices),
        constrained_layout=True
    )

    # Edge cases (1 slice or 1 column)
    if n_slices == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    # --- Color scaling ---
    if share_clim == "global":
        all_vals = [
            np.abs(d[:, :, z, t, T])
            for d in Data
            for z in range(n_slices)
        ]
        vmin = min(np.min(v) for v in all_vals)
        vmax = max(np.max(v) for v in all_vals)
        col_limits = [(vmin, vmax)] * n_cols

    elif share_clim == "per_column":
        col_limits = []
        for d in Data:
            vals = [np.abs(d[:, :, z, t, T]) for z in range(n_slices)]
            vmin = min(np.min(v) for v in vals)
            vmax = max(np.max(v) for v in vals)
            col_limits.append((vmin, vmax))
    else:
        col_limits = None

    # --- Plot ---
    for z in range(n_slices):
        for j, d in enumerate(Data):

            img = np.abs(d[:, :, z, t, T])
            ax = axes[z, j]

            if col_limits is None:
                im = ax.imshow(img, cmap=cmap)
            else:
                vmin, vmax = col_limits[j]
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

            # Titel nur in erster Zeile
            if z == 0:
                ax.set_title(f"{Titles[j]}  (t={t}, T={T})")
            else:
                ax.set_title(f"z={z}")

            ax.axis("on")
            ax.grid(True, color="w", lw=0.5)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig, axes


def plot_voxel_spectra_over_z(
    Data_ft,
    Titles,
    x,
    y,
    T,
    z_max=None,          # optional: nur 0..z_max-1 plotten
    n_cols=2,
    freq_axis=3,         # bei dir: (x,y,z,f,T) -> f ist axis=3
    z_axis=2,            # z ist axis=2
    mag=True,            # True -> abs(), False -> raw complex
    sharex=True,
    sharey=True,
    figsize_per_row=3.0,
    legend=True,
):
    """
    Plots spectra for a fixed voxel (x,y) and timepoint T over all z-slices.
    Each subplot is one z, and within each subplot we overlay all datasets in Data_ft.

    Expected shape per dataset (default): (x, y, z, f, T)
    """

    if len(Data_ft) != len(Titles):
        raise ValueError("Data_ft and Titles must have same length.")

    Z = Data_ft[0].shape[z_axis]
    F = Data_ft[0].shape[freq_axis]

    # Optional truncate z
    if z_max is not None:
        Z = min(Z, int(z_max))

    freqs = np.arange(F)

    n_rows = int(np.ceil(Z / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, n_rows * figsize_per_row),
        sharex=sharex, sharey=sharey,
        constrained_layout=True
    )

    # axes kann 1D sein, wenn n_rows==1
    axes = np.atleast_2d(axes)

    def get_spec(d, z):
        # d shape (x,y,z,f,T) assumed
        spec = d[x, y, z, :, T]
        return np.abs(spec) if mag else spec

    for z in range(Z):
        i, j = divmod(z, n_cols)
        ax = axes[i, j]

        for d, title in zip(Data_ft, Titles):
            ax.plot(freqs, get_spec(d, z), '-', label=title, linewidth=1)

        ax.set_title(f"z={z}")
        ax.grid(True, linestyle=':', alpha=0.3)

        if legend:
            ax.legend(fontsize='small', loc='upper right')

        if i == n_rows - 1:
            ax.set_xlabel("Frequency bin")
        if j == 0:
            ax.set_ylabel("Magnitude" if mag else "Signal")

    # Leere Subplots ausblenden
    for idx in range(Z, n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        axes[i, j].axis('off')

    return fig, axes

def plot_average_spectra_over_T(
    Avg,
    Titles,
    n_cols=2,
    mag=True,
    sharey=True,
    figsize_per_row=3.0,
):
    """
    Avg: list of arrays with shape (F, T)
    Titles: list of labels (same length)
    """

    if len(Avg) != len(Titles):
        raise ValueError("Avg and Titles must have same length.")

    F, n_T = Avg[0].shape

    n_rows = int(np.ceil(n_T / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, n_rows * figsize_per_row),
        sharey=sharey,
        constrained_layout=True
    )

    axes = np.atleast_2d(axes)

    freqs = np.arange(F)

    for T in range(n_T):
        i, j = divmod(T, n_cols)
        ax = axes[i, j]

        for avg, title in zip(Avg, Titles):
            line = np.abs(avg[:, T]) if mag else avg[:, T]
            ax.plot(freqs, line, label=title)

        ax.set_title(f"T = {T}")
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Magnitude" if mag else "Signal")
        ax.grid(True)
        ax.legend()

    # Leere Subplots ausblenden
    for idx in range(n_T, n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        axes[i, j].axis("off")

    return fig, axes
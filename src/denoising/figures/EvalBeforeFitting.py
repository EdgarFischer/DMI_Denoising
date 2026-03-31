import numpy as np
import matplotlib.pyplot as plt

def plot_z_slices(
    Data,
    Titles,
    t,
    T=None,
    cmap="viridis",
    share_clim="per_column",  # "none", "per_column", "global"
):
    """
    Plottet z-Slices für 4D- oder 5D-Daten.

    Parameters
    ----------
    Data : list of np.ndarray
        Liste von Arrays mit Shape
        - 4D: (x, y, z, t)
        - 5D: (x, y, z, t, T)
    Titles : list of str
        Titel pro Datensatz, gleiche Länge wie Data.
    t : int
        Index entlang der t-Achse.
    T : int or None
        Index entlang der T-Achse für 5D-Daten.
        Wird bei 4D-Daten ignoriert.
    cmap : str
        Colormap für imshow.
    share_clim : str
        "none", "per_column", oder "global"
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if len(Data) == 0:
        raise ValueError("Data must not be empty.")
    if len(Data) != len(Titles):
        raise ValueError("Data and Titles must have the same length.")

    n_cols = len(Data)

    first_ndim = Data[0].ndim
    if first_ndim not in (4, 5):
        raise ValueError("Arrays in Data must be 4D or 5D.")

    n_slices = Data[0].shape[2]

    # Konsistenzchecks
    for i, d in enumerate(Data):
        if d.ndim not in (4, 5):
            raise ValueError(f"Data[{i}] has ndim={d.ndim}, expected 4 or 5.")
        if d.shape[2] != n_slices:
            raise ValueError("All arrays must have the same z dimension.")
        if d.ndim == 5 and T is None:
            raise ValueError("T must be provided when plotting 5D arrays.")

    def get_img(d, z, t, T):
        if d.ndim == 4:
            return np.abs(d[:, :, z, t])
        elif d.ndim == 5:
            return np.abs(d[:, :, z, t, T])
        else:
            raise ValueError(f"Unsupported ndim={d.ndim}")

    # --- Figure ---
    fig, axes = plt.subplots(
        n_slices,
        n_cols,
        figsize=(3 * n_cols, 2.5 * n_slices),
        constrained_layout=True
    )

    # Edge cases
    if n_slices == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    # --- Color scaling ---
    if share_clim == "global":
        all_vals = [
            get_img(d, z, t, T)
            for d in Data
            for z in range(n_slices)
        ]
        vmin = min(np.min(v) for v in all_vals)
        vmax = max(np.max(v) for v in all_vals)
        col_limits = [(vmin, vmax)] * n_cols

    elif share_clim == "per_column":
        col_limits = []
        for d in Data:
            vals = [get_img(d, z, t, T) for z in range(n_slices)]
            vmin = min(np.min(v) for v in vals)
            vmax = max(np.max(v) for v in vals)
            col_limits.append((vmin, vmax))

    elif share_clim == "none":
        col_limits = None

    else:
        raise ValueError("share_clim must be one of: 'none', 'per_column', 'global'")

    # --- Plot ---
    for z in range(n_slices):
        for j, d in enumerate(Data):
            img = get_img(d, z, t, T)
            ax = axes[z, j]

            if col_limits is None:
                im = ax.imshow(img, cmap=cmap)
            else:
                vmin, vmax = col_limits[j]
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

            # Titel
            if z == 0:
                if d.ndim == 4:
                    ax.set_title(f"{Titles[j]}  (t={t})")
                else:
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
    T=None,
    z_max=None,
    n_cols=2,
    freq_axis=3,
    z_axis=2,
    mag=True,
    sharex=True,
    sharey=True,
    figsize_per_row=3.0,
    legend=True,
    min_t_index=0,
    max_t_index=None,   # <-- neu
):
    import numpy as np
    import matplotlib.pyplot as plt

    if len(Data_ft) == 0:
        raise ValueError("Data_ft must not be empty.")
    if len(Data_ft) != len(Titles):
        raise ValueError("Data_ft and Titles must have same length.")

    first = Data_ft[0]
    if first.ndim not in (4, 5):
        raise ValueError("Arrays in Data_ft must be 4D or 5D.")

    Z = first.shape[z_axis]
    F = first.shape[freq_axis]

    # --- neue checks ---
    if not (0 <= min_t_index < F):
        raise ValueError(f"min_t_index must be between 0 and {F-1}, got {min_t_index}.")

    if max_t_index is None:
        max_t_index = F
    else:
        if not (0 < max_t_index <= F):
            raise ValueError(f"max_t_index must be between 1 and {F}, got {max_t_index}.")

    if min_t_index >= max_t_index:
        raise ValueError("min_t_index must be smaller than max_t_index.")

    for i, d in enumerate(Data_ft):
        if d.ndim not in (4, 5):
            raise ValueError(f"Data_ft[{i}] has ndim={d.ndim}, expected 4 or 5.")
        if d.shape[z_axis] != Z:
            raise ValueError("All arrays must have same z dimension.")
        if d.shape[freq_axis] != F:
            raise ValueError("All arrays must have same frequency dimension.")
        if d.ndim == 5 and T is None:
            raise ValueError("T must be provided for 5D arrays.")

    if z_max is not None:
        Z = min(Z, int(z_max))

    freqs = np.arange(min_t_index, max_t_index)

    n_rows = int(np.ceil(Z / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, n_rows * figsize_per_row),
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True
    )

    axes = np.atleast_2d(axes)

    def get_spec(d, z):
        if d.ndim == 4:
            spec = d[x, y, z, :]
        else:
            spec = d[x, y, z, :, T]
        spec = spec[min_t_index:max_t_index]  # <-- geändert
        return np.abs(spec) if mag else spec

    for z in range(Z):
        i, j = divmod(z, n_cols)
        ax = axes[i, j]

        for idx, (d, title) in enumerate(zip(Data_ft, Titles)):
            if idx == 0:
                ax.plot(
                    freqs,
                    get_spec(d, z),
                    color="black",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=1,
                    label=title,
                )
            else:
                ax.plot(
                    freqs,
                    get_spec(d, z),
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                    label=title,
                )

        ax.set_title(f"z={z}")
        ax.grid(True, linestyle=":", alpha=0.3)

        if legend:
            ax.legend(fontsize="small", loc="upper right")

        if i == n_rows - 1:
            ax.set_xlabel("Frequency bin")
        if j == 0:
            ax.set_ylabel("Magnitude" if mag else "Signal")

    for idx in range(Z, n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        axes[i, j].axis("off")

    return fig, axes

def plot_average_spectra_over_T(
    Avg,
    Titles,
    n_cols=2,
    mag=True,
    sharey=True,
    figsize_per_row=3.0,
):
    import numpy as np
    import matplotlib.pyplot as plt

    if len(Avg) == 0:
        raise ValueError("Avg must not be empty.")
    if len(Avg) != len(Titles):
        raise ValueError("Avg and Titles must have same length.")

    first = Avg[0]
    if first.ndim not in (1, 2):
        raise ValueError("Arrays in Avg must be 1D or 2D.")

    F = first.shape[0]
    n_T = 1 if first.ndim == 1 else first.shape[1]

    for i, avg in enumerate(Avg):
        if avg.ndim not in (1, 2):
            raise ValueError(f"Avg[{i}] invalid ndim.")
        if avg.shape[0] != F:
            raise ValueError("All arrays must have same frequency dimension.")

        if first.ndim != avg.ndim:
            raise ValueError("Do not mix 1D and 2D arrays.")

        if avg.ndim == 2 and avg.shape[1] != n_T:
            raise ValueError("All 2D arrays must have same T dimension.")

    freqs = np.arange(F)

    n_rows = int(np.ceil(n_T / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, n_rows * figsize_per_row),
        sharey=sharey,
        constrained_layout=True
    )

    axes = np.atleast_2d(axes)

    def get_line(avg, T_idx=0):
        line = avg if avg.ndim == 1 else avg[:, T_idx]
        return np.abs(line) if mag else line

    for T in range(n_T):
        i, j = divmod(T, n_cols)
        ax = axes[i, j]

        for idx, (avg, title) in enumerate(zip(Avg, Titles)):
            if idx == 0:
                # Deep → Hintergrund
                ax.plot(
                    freqs,
                    get_line(avg, T),
                    color="black",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=1,
                    label=title,
                )
            else:
                # andere → darüber
                ax.plot(
                    freqs,
                    get_line(avg, T),
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                    label=title,
                )

        if first.ndim == 1:
            ax.set_title("Average spectrum")
        else:
            ax.set_title(f"T = {T}")

        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Magnitude" if mag else "Signal")
        ax.grid(True)

        ax.legend()

    for idx in range(n_T, n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        axes[i, j].axis("off")

    return fig, axes
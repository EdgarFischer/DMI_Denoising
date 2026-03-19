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
    z_max=None,          # optional: nur 0..z_max-1 plotten
    n_cols=2,
    freq_axis=3,         # default: (x,y,z,f[,T]) -> f ist axis=3
    z_axis=2,            # z ist axis=2
    mag=True,            # True -> abs(), False -> raw complex
    sharex=True,
    sharey=True,
    figsize_per_row=3.0,
    legend=True,
):
    """
    Plots spectra for a fixed voxel (x,y) over all z-slices.
    Each subplot is one z, and within each subplot all datasets in Data_ft are overlaid.

    Supported shapes per dataset:
        4D: (x, y, z, f)
        5D: (x, y, z, f, T)

    Parameters
    ----------
    Data_ft : list of np.ndarray
        Liste von 4D- oder 5D-Arrays.
    Titles : list of str
        Titel für die Datensätze, gleiche Länge wie Data_ft.
    x, y : int
        Fester Voxelindex.
    T : int or None
        Index entlang T für 5D-Daten. Bei 4D-Daten ignoriert.
    z_max : int or None
        Falls gesetzt, werden nur z=0..z_max-1 geplottet.
    n_cols : int
        Anzahl Subplots pro Zeile.
    freq_axis : int
        Achse der Frequenzdimension.
    z_axis : int
        Achse der z-Dimension.
    mag : bool
        True -> Betrag plotten, False -> rohes Signal.
    sharex, sharey : bool
        Achsen zwischen Subplots teilen.
    figsize_per_row : float
        Höhe pro Subplot-Zeile.
    legend : bool
        Ob eine Legende angezeigt werden soll.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if len(Data_ft) == 0:
        raise ValueError("Data_ft must not be empty.")
    if len(Data_ft) != len(Titles):
        raise ValueError("Data_ft and Titles must have same length.")

    # Konsistenzchecks
    first = Data_ft[0]
    if first.ndim not in (4, 5):
        raise ValueError("Arrays in Data_ft must be 4D or 5D.")

    Z = first.shape[z_axis]
    F = first.shape[freq_axis]

    for i, d in enumerate(Data_ft):
        if d.ndim not in (4, 5):
            raise ValueError(f"Data_ft[{i}] has ndim={d.ndim}, expected 4 or 5.")
        if d.shape[z_axis] != Z:
            raise ValueError("All arrays in Data_ft must have the same z dimension.")
        if d.shape[freq_axis] != F:
            raise ValueError("All arrays in Data_ft must have the same frequency dimension.")
        if d.ndim == 5 and T is None:
            raise ValueError("T must be provided when plotting 5D arrays.")

    # Optional truncate z
    if z_max is not None:
        Z = min(Z, int(z_max))

    freqs = np.arange(F)

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
        elif d.ndim == 5:
            spec = d[x, y, z, :, T]
        else:
            raise ValueError(f"Unsupported ndim={d.ndim}")
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
    Plottet gemittelte Spektren.

    Supported shapes per entry in Avg:
        1D: (F,)
        2D: (F, T)

    Parameters
    ----------
    Avg : list of np.ndarray
        Liste von Arrays mit Shape (F,) oder (F, T).
    Titles : list of str
        Labels für die Arrays, gleiche Länge wie Avg.
    n_cols : int
        Anzahl Subplots pro Zeile.
    mag : bool
        True -> Betrag plotten, False -> rohes Signal.
    sharey : bool
        Ob die y-Achse zwischen Subplots geteilt wird.
    figsize_per_row : float
        Höhe pro Subplot-Zeile.
    """

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
            raise ValueError(f"Avg[{i}] has ndim={avg.ndim}, expected 1 or 2.")
        if avg.shape[0] != F:
            raise ValueError("All arrays in Avg must have the same frequency dimension.")

        if first.ndim == 1 and avg.ndim != 1:
            raise ValueError("Do not mix 1D and 2D arrays in Avg.")
        if first.ndim == 2 and avg.ndim != 2:
            raise ValueError("Do not mix 1D and 2D arrays in Avg.")

        if avg.ndim == 2 and avg.shape[1] != n_T:
            raise ValueError("All 2D arrays in Avg must have the same T dimension.")

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
        if avg.ndim == 1:
            line = avg
        elif avg.ndim == 2:
            line = avg[:, T_idx]
        else:
            raise ValueError(f"Unsupported ndim={avg.ndim}")
        return np.abs(line) if mag else line

    for T in range(n_T):
        i, j = divmod(T, n_cols)
        ax = axes[i, j]

        for avg, title in zip(Avg, Titles):
            ax.plot(freqs, get_line(avg, T), label=title)

        if first.ndim == 1:
            ax.set_title("Average spectrum")
        else:
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
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

from matplotlib.widgets import Slider


def interactive_spectra_viewer(
    image_volume,
    spectrum_func1,
    spectrum_func2,
    label1="Spectrum 1",
    label2="Spectrum 2",
    f_min=None,
    f_max=None,
    z_init=0,
    x_init=0,
    y_init=0,
    cmap="gray",
    figsize=(11, 4),
    legend=True,
):
    """
    Interaktiver Viewer:
    - links: 3D-Bildvolumen image_volume[x, y, z]
    - rechts: zwei 1D-Spektren aus spectrum_func1 / spectrum_func2

    Parameter
    ----------
    image_volume : np.ndarray
        Array mit shape (X, Y, Z), das links angezeigt wird.

    spectrum_func1 : callable
        Funktion f(x, y, z) -> 1D array der Länge F

    spectrum_func2 : callable
        Funktion f(x, y, z) -> 1D array der Länge F

    label1, label2 : str
        Legenden für die beiden Spektren.

    f_min, f_max : int oder None
        Frequenzindexbereich für die Anzeige.
        Default: vollständiges Spektrum.

    z_init, x_init, y_init : int
        Startposition.

    cmap : str
        Colormap für linkes Bild.

    figsize : tuple
        Figuregröße.

    legend : bool
        Ob rechts eine Legende angezeigt werden soll.
    """

    image_volume = np.asarray(image_volume)

    if image_volume.ndim != 3:
        raise ValueError("image_volume muss shape (X, Y, Z) haben.")

    X, Y, Z = image_volume.shape

    if not (0 <= z_init < Z):
        raise ValueError(f"z_init muss zwischen 0 und {Z-1} liegen.")
    if not (0 <= x_init < X):
        raise ValueError(f"x_init muss zwischen 0 und {X-1} liegen.")
    if not (0 <= y_init < Y):
        raise ValueError(f"y_init muss zwischen 0 und {Y-1} liegen.")

    spec1_init = np.asarray(spectrum_func1(x_init, y_init, z_init)).squeeze()
    spec2_init = np.asarray(spectrum_func2(x_init, y_init, z_init)).squeeze()

    if spec1_init.ndim != 1 or spec2_init.ndim != 1:
        raise ValueError("spectrum_func1 und spectrum_func2 müssen 1D-Arrays zurückgeben.")

    if len(spec1_init) != len(spec2_init):
        raise ValueError("Beide Spektren müssen die gleiche Länge haben.")

    F = len(spec1_init)

    if f_min is None:
        f_min = 0
    if f_max is None:
        f_max = F - 1

    if not (0 <= f_min <= f_max < F):
        raise ValueError(f"f_min/f_max müssen im Bereich 0 bis {F-1} liegen.")

    f_axis = np.arange(f_min, f_max + 1)

    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(bottom=0.25)

    # Linkes Bild
    img0 = image_volume[:, :, z_init]
    im = ax_img.imshow(img0.T, origin="lower", cmap=cmap)
    ax_img.set_title(f"Slice z={z_init}")
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("y")

    marker, = ax_img.plot(x_init, y_init, "ro")

    # Rechte Spektren mit deinen Wunschfarben
    line1, = ax_spec.plot(
        f_axis,
        spec1_init[f_min:f_max + 1],
        color="black",
        linewidth=1.5,
        alpha=0.8,
        zorder=1,
        label=label1,
    )

    line2, = ax_spec.plot(
        f_axis,
        spec2_init[f_min:f_max + 1],
        color="C0",
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
        label=label2,
    )

    ax_spec.set_title(f"Spectra at x={x_init}, y={y_init}, z={z_init}")
    ax_spec.set_xlabel("f")
    ax_spec.set_ylabel("signal")

    if legend:
        ax_spec.legend()

    ax_spec.relim()
    ax_spec.autoscale_view()

    current = {"x": x_init, "y": y_init, "z": z_init}

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    z_slider = Slider(
        ax=ax_slider,
        label="z",
        valmin=0,
        valmax=Z - 1,
        valinit=z_init,
        valstep=1
    )

    def redraw_spectra():
        x = current["x"]
        y = current["y"]
        z = current["z"]

        spec1 = np.asarray(spectrum_func1(x, y, z)).squeeze()
        spec2 = np.asarray(spectrum_func2(x, y, z)).squeeze()

        if spec1.ndim != 1 or spec2.ndim != 1:
            raise ValueError("Spektrumsfunktionen müssen 1D-Arrays zurückgeben.")
        if len(spec1) != F or len(spec2) != F:
            raise ValueError("Spektrumlänge darf sich nicht zwischen Voxeln ändern.")

        line1.set_data(f_axis, spec1[f_min:f_max + 1])
        line2.set_data(f_axis, spec2[f_min:f_max + 1])

        ax_spec.relim()
        ax_spec.autoscale_view()
        ax_spec.set_title(f"Spectra at x={x}, y={y}, z={z}")

    def onclick(event):
        if event.inaxes != ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if x < 0 or x >= X or y < 0 or y >= Y:
            return

        current["x"] = x
        current["y"] = y

        marker.set_data([x], [y])
        redraw_spectra()
        fig.canvas.draw_idle()

    def update_z(val):
        z = int(z_slider.val)
        current["z"] = z

        new_img = image_volume[:, :, z]
        im.set_data(new_img.T)
        ax_img.set_title(f"Slice z={z}")

        redraw_spectra()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    z_slider.on_changed(update_z)

    plt.show()

    return fig, (ax_img, ax_spec), z_slider

import numpy as np
import matplotlib.pyplot as plt


def plot_real_imag_ft(
    data_5d,
    x,
    y,
    z,
    title=None,
    save_path=None,
    clip_percentiles=(1, 99),
    cmap="gray",
    mode="grayscale",   # "grayscale" oder "signed"
    figsize=(2.2, 2.0),
    dpi=300,
):
    """
    Plot real and imaginary parts of one voxel's f x T spectrum.

    mode:
        "grayscale" -> helles Concept-Figure Mapping
        "signed"    -> symmetrische Darstellung (für diverging colormaps)
    """

    voxel = data_5d[x, y, z, ...]  # (f, T)

    real = np.real(voxel).T
    imag = np.imag(voxel).T

    data = np.concatenate([real.ravel(), imag.ravel()])

    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

    if mode == "grayscale":
        # Helles Mapping für Concept Figure
        p_low, p_high = np.percentile(data, clip_percentiles)

        def preprocess(img):
            img = np.clip(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low + 1e-12)
            return img

        real_plot = preprocess(real)
        imag_plot = preprocess(imag)

        vmin, vmax = 0, 1

    elif mode == "signed":
        # Für diverging colormaps (z.B. RdGy_r)
        vmax = np.percentile(np.abs(data), clip_percentiles[1])
        vmin = -vmax

        real_plot = real
        imag_plot = imag

    else:
        raise ValueError("mode must be 'grayscale' or 'signed'")

    for a, img, label in zip(ax, [real_plot, imag_plot], ["Real part", "Imaginary part"]):
        a.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        # dynamische Textfarbe je nach cmap
        text_color = "black" if mode == "grayscale" else "white"
        box_color = "white" if mode == "grayscale" else "black"

        a.text(
            0.03, 0.88, label,
            transform=a.transAxes,
            fontsize=6,
            fontweight="bold",
            ha="left",
            va="top",
            color=text_color,
            bbox=dict(facecolor=box_color, alpha=0.5, edgecolor="none", pad=1.2),
        )

        a.set_xticks([])
        a.set_yticks([])

        for spine in a.spines.values():
            spine.set_linewidth(0.6)

    fig.text(0.02, 0.5, "Repetition", rotation=90,
             va="center", ha="center", fontsize=7)
    fig.text(0.5, 0.04, "Frequency",
             va="center", ha="center", fontsize=7)

    if title is not None:
        fig.suptitle(title, fontsize=8, y=0.99)
        top = 0.90
    else:
        top = 0.98

    plt.subplots_adjust(left=0.12, right=0.98,
                        top=top, bottom=0.13, hspace=0.03)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)

    return fig, ax
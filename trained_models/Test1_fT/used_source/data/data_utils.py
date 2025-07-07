import os
import numpy as np

def load_and_preprocess_data(
    folder_names: list,
    base_path: str,
    fourier_axes: list = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Lädt jeweils 'data.npy' aus jedem Unterordner in base_path und
    stapelt sie entlang der letzten Achse (D). Führt optional FFT und
    Normierung durch.

    Args:
      folder_names: Liste von Ordnern unter base_path, z.B. ['P03','P04',...]
      base_path:    Pfad zu Deinem 'datasets'-Ordner
      fourier_axes: Liste von Achsen, auf denen np.fft.fft+fftshift angewandt werden soll
      normalize:    True → jede Teildatenmenge wird auf max(abs)=1 skaliert

    Returns:
      data: np.ndarray mit Shape (X, Y, Z, t, T, D)
    """
    arrays = []
    for fold in folder_names:
        fn = os.path.join(base_path, fold, 'data.npy')
        arr = np.load(fn)               # erwartet Shape (X,Y,Z,t,T)
        if arr.ndim == 5:
            arr = arr[..., np.newaxis]  # → (X,Y,Z,t,T,1)

        # 1) Normalisieren
        if normalize:
            maxv = np.max(np.abs(arr))
            if maxv > 0:
                arr = arr / maxv

        # 2) Fourieranalyse
        if fourier_axes:
            for ax in fourier_axes:
                # unge-shiftete FFT
                arr = np.fft.fft(arr, axis=ax)
                # zentrieren
                arr = np.fft.fftshift(arr, axes=ax)

        arrays.append(arr)

    # 3) Stapeln aller Runs → Shape (X,Y,Z,t,T,D)
    return np.concatenate(arrays, axis=-1)

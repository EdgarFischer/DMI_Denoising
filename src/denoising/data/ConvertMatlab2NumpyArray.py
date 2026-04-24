import numpy as np
import h5py
import os
from pathlib import Path

for SNR in [1]:
    # Arbeitsverzeichnis auf aktuelle Datei setzen
    DATA_DIRECTORY = Path(f"/workspace/Denoising/datasets/DMI/Simulations/SNRExperiments_double_radius/SNR{SNR}")

    # automatisch alle Vol*-Ordner finden
    Ordner_Liste = [f for f in DATA_DIRECTORY.glob("Noise*") if f.is_dir()]

    for Ordner in Ordner_Liste:
        input_path = Ordner / 'data_tMPPCA.mat'
        output_path = Ordner / 'data_tMPPCA.npy'

        print(f"Lade {input_path} ...")
        with h5py.File(input_path, 'r') as f:
            if 'Data' not in f:
                raise KeyError(f"'Data' nicht in {input_path} gefunden.")

            dset = f['Data']
            arr = np.array(dset)

            # Komplex gespeichert? (MATLAB v7.3 speichert als struct mit Feldern 'real'/'imag')
            if arr.dtype.names == ('real', 'imag'):
                print("→ Komplexe Daten erkannt, konvertiere zu NumPy complex.")
                # dtype der Felder beibehalten (float32/float64)
                real = arr['real']
                imag = arr['imag']
                # in komplexes NumPy-Array umwandeln
                arr = real + 1j * imag

            # Achsenreihenfolge von MATLAB (Fortran) nach NumPy (C) drehen
            Data = np.transpose(arr)

        print(f"Speichere {output_path} ... (shape={Data.shape}, dtype={Data.dtype})")
        np.save(output_path, Data)

print("Fertig!")






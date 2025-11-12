import numpy as np
import h5py
import os

# Arbeitsverzeichnis auf aktuelle Datei setzen
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Ordner_Liste = ['Tumor_1_normalized_tMPPCA_5D_On_All_Reps']
DATA_DIRECTORY = "../datasets/"

for Ordner in Ordner_Liste:
    input_path = os.path.join(DATA_DIRECTORY, Ordner, 'data.mat')
    output_path = os.path.join(DATA_DIRECTORY, Ordner, 'data.npy')

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






import numpy as np
from scipy.io import savemat
from pathlib import Path

for SNR in [1]:

    DATA_DIRECTORY = Path(f"/workspace/Denoising/datasets/DMI/Simulations/SNRExperiments_double_radius/SNR{SNR}")

    # alle Noise_* Ordner finden
    Ordner_Liste = [f for f in DATA_DIRECTORY.glob("Noise_*") if f.is_dir()]

    for folder in Ordner_Liste:
        input_file = folder / "data.npy"
        output_file = folder / "data.mat"

        if not input_file.exists():
            print(f"Überspringe {folder}, keine data.npy gefunden")
            continue

        print(f"Lade {input_file}")
        Data = np.load(input_file)

        savemat(output_file, {'Data': Data})
        print(f"Gespeichert: {output_file}")

    print("fertig")
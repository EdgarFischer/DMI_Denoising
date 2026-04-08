import numpy as np
from scipy.io import savemat
from pathlib import Path

# working directory auf aktuelle Datei setzen
#base_path = Path(__file__).resolve().parent

DATA_DIRECTORY = Path("/workspace/Denoising/datasets/Proton/B0corrected_wo_LipidMask")

# automatisch alle Vol*-Ordner finden
Ordner_Liste = [f for f in DATA_DIRECTORY.glob("Vol*") if f.is_dir()]

for folder in Ordner_Liste:
    input_file = folder / "OriginalData/data_after_walinet.npy"
    output_file = folder / "OriginalData/data_after_walinet.mat"

    if not input_file.exists():
        print(f"Überspringe {folder}, keine data_after_walinet.npy gefunden")
        continue

    print(f"Lade {input_file}")
    Data = np.load(input_file)

    savemat(output_file, {'Data': Data})
    print(f"Gespeichert: {output_file}")

print('fertig')
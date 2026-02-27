import numpy as np
from scipy.io import savemat
import os

# working directory auf aktuelle Datei setzen
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ordnernamen im datasets ordner mit data.npy dateien angeben die zu .mat konvertiert werden

Ordner_Liste = ['sf_brain_DMI_HC_pilot_normalized']

DATA_DIRECTORY = "../datasets/"

for Ordner in Ordner_Liste:

    Data = np.load(DATA_DIRECTORY+Ordner+'/data.npy')
    savemat(DATA_DIRECTORY+Ordner+'/data.mat', {'Data': Data})

print('fertig')
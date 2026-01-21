import numpy as np

import scipy.io
import numpy as np

mat = scipy.io.loadmat('CombinedCSI.mat')
mask = mat['mask']        # exakt dein Feld
np.save('mask.npy', mask)

# Ihr seid voll die Specklümmel man

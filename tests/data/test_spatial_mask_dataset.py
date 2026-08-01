import numpy as np

from denoising.data.mrsi_nd_dataset import MRSiNDataset
from denoising.data.transforms import StratifiedAxisMasking


def test_spatial_mask_restricts_sampling_and_effective_loss_mask():
    data = np.zeros((8, 8, 3, 16, 1, 2), dtype=np.complex64)
    support = np.zeros((8, 8, 3, 1, 1, 2), dtype=bool)
    support[2:6, 2:6, 1, 0, 0, 0] = True
    data[2:6, 2:6, 1, :, 0, 0] = 1 + 2j

    dataset = MRSiNDataset(
        data=data,
        spatial_mask=support,
        image_axes=(0, 1, 3),
        masked_axes=(2,),
        transform=StratifiedAxisMasking(mask_fraction=0.25, window_size=3),
        num_samples=4,
        patching_enabled=True,
        patch_sizes=(4, 4, None),
    )

    for index in range(len(dataset)):
        _, target, effective_mask = dataset[index]
        assert effective_mask.sum() > 0
        # Every effective N2V target must belong to the non-zero support.
        assert np.all(target.numpy()[effective_mask.numpy().astype(bool)] != 0)

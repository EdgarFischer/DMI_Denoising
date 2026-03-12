import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict

class MRSiNDataset(Dataset):
    """
    Flexible MRSI dataset for 2D/3D networks.

    Internal logic:
    - data is assumed to use GLOBAL axes
    - image_axes define the spatial/network axes
    - channel_axis is optional and stays explicit until after masking
    - transform sees structured input:
        without channel_axis: img shape = (2, *spatial)
        with channel_axis   : img shape = (2, C, *spatial)
      where axis 0 is [real, imag]

    The transform may optionally accept:
        transform(img, masked_axes_local=...)

    Returned tensors for the network:
    - without channel_axis: (2, *spatial)
    - with channel_axis   : (2*C, *spatial)
    """

    def __init__(
        self,
        data: np.ndarray,
        image_axes: Tuple[int, ...],
        channel_axis: Optional[int] = None,
        masked_axes: Optional[Tuple[int, ...]] = None,   # GLOBAL axes from config
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,
        num_samples: int = 10000,
        phase_prob: float = 1.0,
    ):
        self.data = data
        self.image_axes = tuple(image_axes)
        self.channel_axis = channel_axis
        self.masked_axes = tuple(masked_axes) if masked_axes is not None else tuple()
        self.fixed = fixed_indices or {}
        self.transform = transform
        self.num_samples = int(num_samples)
        self.phase_prob = float(phase_prob)

        if len(self.image_axes) not in (2, 3):
            raise ValueError("Only 2D and 3D image_axes are supported.")

        if self.channel_axis is not None and self.channel_axis in self.image_axes:
            raise ValueError("channel_axis must not be part of image_axes.")

        # Axes that are kept when extracting a sample
        self.network_axes = list(self.image_axes)
        if self.channel_axis is not None:
            self.network_axes.append(self.channel_axis)

        self.other_axes = [
            ax for ax in range(self.data.ndim)
            if ax not in self.network_axes
        ]

        # masked_axes must refer only to axes visible in the extracted sample
        invalid_mask_axes = [ax for ax in self.masked_axes if ax not in self.network_axes]
        if invalid_mask_axes:
            raise ValueError(
                f"masked_axes {invalid_mask_axes} are not part of visible network axes "
                f"{self.network_axes}."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Sample all axes that are NOT passed to the network
        sel = {
            ax: (self.fixed[ax] if ax in self.fixed
                 else np.random.randint(self.data.shape[ax]))
            for ax in self.other_axes
        }

        slicer = [
            slice(None) if ax in self.network_axes else sel[ax]
            for ax in range(self.data.ndim)
        ]
        arr = self.data[tuple(slicer)]

        # 2) Optional global phase augmentation
        if self.phase_prob > 0 and np.random.rand() < self.phase_prob:
            theta = np.random.rand() * 2 * np.pi
            arr = arr * np.exp(1j * theta)

        # 3) Reorder extracted sample into a STRUCTURED local representation
        #    without channel_axis: (*image_shape)
        #    with channel_axis   : (C, *image_shape)
        kept_axes = [ax for ax in range(self.data.ndim) if ax in self.network_axes]

        if self.channel_axis is None:
            perm = [kept_axes.index(ax) for ax in self.image_axes]
            arr = np.transpose(arr, perm)   # (*spatial)
            real = np.real(arr).astype(np.float32)
            imag = np.imag(arr).astype(np.float32)
            img = np.stack([real, imag], axis=0)   # (2, *spatial)

            # local axes seen by transform:
            # axis 0 = real/imag
            # axes 1.. = image_axes
            local_axis_map = {ax: i + 1 for i, ax in enumerate(self.image_axes)}

        else:
            perm = [kept_axes.index(self.channel_axis)] + [
                kept_axes.index(ax) for ax in self.image_axes
            ]
            arr = np.transpose(arr, perm)   # (C, *spatial)
            real = np.real(arr).astype(np.float32)
            imag = np.imag(arr).astype(np.float32)
            img = np.stack([real, imag], axis=0)   # (2, C, *spatial)

            # local axes seen by transform:
            # axis 0 = real/imag
            # axis 1 = channel_axis
            # axes 2.. = image_axes
            local_axis_map = {self.channel_axis: 1}
            local_axis_map.update({ax: i + 2 for i, ax in enumerate(self.image_axes)})

        masked_axes_local = tuple(local_axis_map[ax] for ax in self.masked_axes)

        # 4) Apply transform on STRUCTURED representation
        if self.transform is not None:
            try:
                inp, tgt, mask = self.transform(img, masked_axes_local=masked_axes_local)
            except TypeError:
                # fallback for older transforms that do not yet accept masked_axes_local
                inp, tgt, mask = self.transform(img)

            # Normalize mask shape to match structured img shape
            expected_shape = img.shape

            if mask.shape != expected_shape:
                # Example: mask only over non-real/imag dims -> broadcast to full tensor
                if mask.ndim == len(expected_shape) - 1:
                    mask = np.broadcast_to(mask[None], expected_shape).copy()
                elif mask.ndim == len(expected_shape) and mask.shape[0] == 1:
                    mask = np.broadcast_to(mask, expected_shape).copy()
                else:
                    raise ValueError(
                        f"Mask shape {mask.shape} cannot be broadcast to expected shape {expected_shape}."
                    )
        else:
            inp = img
            tgt = img
            mask = np.ones_like(img, dtype=bool)

        # 5) Flatten structured representation to network representation
        #    without channel_axis:
        #       (2, *spatial) -> unchanged
        #    with channel_axis:
        #       (2, C, *spatial) -> (2*C, *spatial)
        if self.channel_axis is not None:
            inp = inp.reshape(inp.shape[0] * inp.shape[1], *inp.shape[2:])
            tgt = tgt.reshape(tgt.shape[0] * tgt.shape[1], *tgt.shape[2:])
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])

        return (
            torch.from_numpy(inp.astype(np.float32, copy=False)),
            torch.from_numpy(tgt.astype(np.float32, copy=False)),
            torch.from_numpy(mask.astype(np.float32, copy=False)),
        )
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict
from denoising.data.patching import get_random_patch_slices, extract_patch
from denoising.data.augmentations import PatchAugmentationPipeline


class MRSiNDataset(Dataset):
    """
    Flexible MRSI dataset for 2D/3D networks.

    Internal logic:
    - data is assumed to use GLOBAL axes
    - image_axes / view_sampling.views define GLOBAL input axes
    - channel_axis is optional and stays explicit until after masking
    - masking axes are LOCAL within the currently selected input view
    - transform sees structured input:
        without channel_axis: img shape = (2, *spatial)
        with channel_axis   : img shape = (2, C, *spatial)
      where axis 0 is [real, imag]

    Returned tensors for the network:
    - without channel_axis: (2, *spatial)
    - with channel_axis   : (2*C, *spatial)
    """

    def __init__(
        self,
        data: np.ndarray,
        image_axes: Tuple[int, ...],
        channel_axis: Optional[int] = None,
        masked_axes: Optional[Tuple[int, ...]] = None,   # LOCAL axes within current input view
        fixed_indices: Optional[Dict[int, int]] = None,
        transform=None,
        num_samples: int = 10000,
        augmentation=None,
        patching_enabled: bool = False,
        patch_sizes: Optional[Tuple[Optional[int], ...]] = None,
        view_sampling=None,
        spatial_mask: Optional[np.ndarray] = None,
    ):
        self.data = data
        self.spatial_mask = spatial_mask
        self.image_axes = tuple(image_axes)
        self.channel_axis = channel_axis
        self.masked_axes = tuple(masked_axes) if masked_axes is not None else tuple()
        self.fixed = fixed_indices or {}
        self.transform = transform
        self.num_samples = int(num_samples)
        self.augmentation = (
            PatchAugmentationPipeline(augmentation)
            if augmentation is not None else None
        )
        self.patching_enabled = bool(patching_enabled)
        self.patch_sizes = tuple(patch_sizes) if patch_sizes is not None else tuple()
        self.view_sampling = view_sampling

        if len(self.image_axes) not in (2, 3):
            raise ValueError("Only 2D and 3D image_axes are supported.")

        if self.spatial_mask is not None:
            if self.spatial_mask.ndim != self.data.ndim:
                raise ValueError(
                    "spatial_mask must have the same number of axes as data."
                )
            for axis, (mask_size, data_size) in enumerate(
                zip(self.spatial_mask.shape, self.data.shape)
            ):
                if mask_size not in (1, data_size):
                    raise ValueError(
                        f"spatial_mask axis {axis} has size {mask_size}; "
                        f"expected 1 or {data_size}."
                    )
            if self.view_sampling is not None and getattr(
                self.view_sampling, "enabled", False
            ):
                raise ValueError(
                    "spatial_mask with view_sampling is not yet supported."
                )

            network_axes = set(self.image_axes)
            if self.channel_axis is not None:
                network_axes.add(self.channel_axis)
            self._masked_other_axes = tuple(
                axis for axis in range(self.data.ndim) if axis not in network_axes
            )
            nonzero = np.argwhere(self.spatial_mask)
            valid_other = np.unique(
                nonzero[:, self._masked_other_axes], axis=0
            )
            if self.fixed:
                valid_other = np.asarray([
                    values for values in valid_other
                    if all(
                        values[self._masked_other_axes.index(axis)] == value
                        for axis, value in self.fixed.items()
                        if axis in self._masked_other_axes
                    )
                ])
            if len(valid_other) == 0:
                raise ValueError("spatial_mask contains no valid samples.")
            self._valid_other_indices = valid_other

        if self.channel_axis is not None and self.channel_axis in self.image_axes:
            raise ValueError("channel_axis must not be part of image_axes.")

        if any(ax < 0 or ax >= len(self.image_axes) for ax in self.masked_axes):
            raise ValueError(
                f"masked_axes must refer to local axes within the current input view "
                f"(0..{len(self.image_axes)-1}), got {self.masked_axes}."
            )

        if self.patching_enabled:
            expected_num_patch_axes = len(self.image_axes) + (1 if self.channel_axis is not None else 0)

            if len(self.patch_sizes) != expected_num_patch_axes:
                raise ValueError(
                    f"patch_sizes must have length {expected_num_patch_axes}, "
                    f"but got {len(self.patch_sizes)}."
                )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.view_sampling is not None and getattr(self.view_sampling, "enabled", False):
            image_axes = tuple(
                self.view_sampling.views[np.random.randint(len(self.view_sampling.views))]
            )
        else:
            image_axes = self.image_axes

        network_axes = list(image_axes)
        if self.channel_axis is not None:
            network_axes.append(self.channel_axis)

        other_axes = [
            ax for ax in range(self.data.ndim)
            if ax not in network_axes
        ]

        # 1) Sample all axes that are NOT passed to the network
        if self.spatial_mask is not None:
            if tuple(other_axes) != self._masked_other_axes:
                raise RuntimeError("Mask axes changed after dataset initialization.")
            chosen = self._valid_other_indices[
                np.random.randint(len(self._valid_other_indices))
            ]
            sel = {axis: int(value) for axis, value in zip(other_axes, chosen)}
        else:
            sel = {
                ax: (self.fixed[ax] if ax in self.fixed
                     else np.random.randint(self.data.shape[ax]))
                for ax in other_axes
            }

        slicer = [
            slice(None) if ax in network_axes else sel[ax]
            for ax in range(self.data.ndim)
        ]
        arr = self.data[tuple(slicer)]
        support = None
        if self.spatial_mask is not None:
            mask_slicer = [
                slice(None) if axis in network_axes else min(sel[axis], self.spatial_mask.shape[axis] - 1)
                for axis in range(self.data.ndim)
            ]
            support = self.spatial_mask[tuple(mask_slicer)]

        # 2) Reorder extracted sample into a STRUCTURED local representation
        kept_axes = [ax for ax in range(self.data.ndim) if ax in network_axes]

        if self.channel_axis is None:
            perm = [kept_axes.index(ax) for ax in image_axes]
            arr = np.transpose(arr, perm)   # (*spatial)
            if support is not None:
                support = np.transpose(support, perm)
                support = np.broadcast_to(support, arr.shape)
            real = np.real(arr).astype(np.float32)
            imag = np.imag(arr).astype(np.float32)
            img = np.stack([real, imag], axis=0)   # (2, *spatial)

            local_axis_map = {ax: i + 1 for i, ax in enumerate(image_axes)}

        else:
            perm = [kept_axes.index(self.channel_axis)] + [
                kept_axes.index(ax) for ax in image_axes
            ]
            arr = np.transpose(arr, perm)   # (C, *spatial)
            if support is not None:
                support = np.transpose(support, perm)
                support = np.broadcast_to(support, arr.shape)
            real = np.real(arr).astype(np.float32)
            imag = np.imag(arr).astype(np.float32)
            img = np.stack([real, imag], axis=0)   # (2, C, *spatial)

            local_axis_map = {self.channel_axis: 1}
            local_axis_map.update({ax: i + 2 for i, ax in enumerate(image_axes)})

        if self.patching_enabled:
            patch_slices = None
            for _ in range(100):
                candidate = get_random_patch_slices(img.shape, self.patch_sizes)
                if support is None or np.any(support[candidate[1:]]):
                    patch_slices = candidate
                    break
            if patch_slices is None:
                raise RuntimeError("Could not sample a patch intersecting spatial_mask.")
            img = extract_patch(img, patch_slices)
            if support is not None:
                support = support[patch_slices[1:]]

        if self.augmentation is not None:
            img, local_axis_map = self.augmentation(img, local_axis_map)

        if self.channel_axis is None:
            masked_axes_local = tuple(ax + 1 for ax in self.masked_axes)
        else:
            masked_axes_local = tuple(ax + 2 for ax in self.masked_axes)

        # 3) Apply transform on STRUCTURED representation
        if self.transform is not None:
            try:
                inp, tgt, mask = self.transform(img, masked_axes_local=masked_axes_local)
            except TypeError:
                inp, tgt, mask = self.transform(img)

            expected_shape = img.shape

            if mask.shape != expected_shape:
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

        if support is not None:
            support_structured = np.broadcast_to(support[None], mask.shape)
            mask = np.asarray(mask, dtype=bool) & support_structured

        # 4) Flatten structured representation to network representation
        if self.channel_axis is not None:
            inp = inp.reshape(inp.shape[0] * inp.shape[1], *inp.shape[2:])
            tgt = tgt.reshape(tgt.shape[0] * tgt.shape[1], *tgt.shape[2:])
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])

        return (
            torch.from_numpy(inp.astype(np.float32, copy=False)),
            torch.from_numpy(tgt.astype(np.float32, copy=False)),
            torch.from_numpy(mask.astype(np.float32, copy=False)),
        )

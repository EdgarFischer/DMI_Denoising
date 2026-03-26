import numpy as np
from typing import Dict, Tuple


def apply_global_phase(img: np.ndarray) -> np.ndarray:
    """
    Multiply full structured patch by exp(i * theta), theta ~ U(0, 2*pi).

    Input:
      img:
        - without channel_axis: (2, *spatial)
        - with channel_axis   : (2, C, *spatial)

    Returns:
      img_aug with same shape
    """
    theta = np.random.rand() * 2.0 * np.pi

    z = img[0] + 1j * img[1]
    z = z * np.exp(1j * theta)

    out = np.stack(
        [z.real.astype(np.float32, copy=False),
         z.imag.astype(np.float32, copy=False)],
        axis=0,
    )
    return out


def apply_global_scale(img: np.ndarray, scale_min: float, scale_max: float) -> np.ndarray:
    """
    Multiply full structured patch by scalar ~ U(scale_min, scale_max).
    """
    scale = np.random.uniform(scale_min, scale_max)
    return (img * scale).astype(np.float32, copy=False)


def apply_inversion(
    img: np.ndarray,
    local_axis_map: Dict[int, int],
    axes_global: Tuple[int, ...],
) -> np.ndarray:
    """
    Randomly flip independently along the specified GLOBAL axes
    that are visible in the current structured patch.

    local_axis_map maps:
      global axis -> local structured axis

    Example:
      without channel_axis:
        img.shape = (2, X, Y, F)
        local axes start at 1

      with channel_axis:
        img.shape = (2, C, X, Y, F)
        local axes:
          0 = real/imag
          1 = channel_axis
          2.. = image_axes
    """
    out = img

    for ax_global in axes_global:
        if ax_global not in local_axis_map:
            continue

        ax_local = local_axis_map[ax_global]

        if np.random.rand() < 0.5:
            out = np.flip(out, axis=ax_local).copy()

    return out.astype(np.float32, copy=False)


def apply_permutation(
    img: np.ndarray,
    local_axis_map: Dict[int, int],
    axes_global: Tuple[int, ...],
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Randomly permute the specified GLOBAL axes among each other.

    Important:
      - only axes visible in the current structured patch are considered
      - only axes with matching sizes may be permuted together
      - returns updated local_axis_map

    Example:
      axes_global = (0, 1, 2)
      visible local axes might be (2, 3, 4)
      then we may randomly permute those three axes

    If fewer than 2 eligible axes are visible, img is returned unchanged.
    """
    visible_axes_global = [ax for ax in axes_global if ax in local_axis_map]

    if len(visible_axes_global) < 2:
        return img, dict(local_axis_map)

    visible_axes_local = [local_axis_map[ax] for ax in visible_axes_global]
    visible_sizes = [img.shape[ax] for ax in visible_axes_local]

    # Only permute if all selected axes have same size
    if len(set(visible_sizes)) != 1:
        return img, dict(local_axis_map)

    permuted_globals = list(np.random.permutation(visible_axes_global))

    # Build transpose order over all axes of img
    axes_order = list(range(img.ndim))

    # Old local positions of the selected globals
    old_local_positions = [local_axis_map[gax] for gax in visible_axes_global]

    # New local positions are taken from the permuted globals
    # meaning: the content that used to belong to permuted_globals[k]
    # will move into old_local_positions[k]
    new_local_positions = [local_axis_map[gax] for gax in permuted_globals]

    for dst_pos, src_pos in zip(old_local_positions, new_local_positions):
        axes_order[dst_pos] = src_pos

    out = np.transpose(img, axes=axes_order).copy()

    # Update global -> local map:
    # after permutation, each global axis in visible_axes_global keeps its role,
    # but now lives at the local position where its data was moved to.
    new_local_axis_map = dict(local_axis_map)
    for gax, new_pos in zip(permuted_globals, old_local_positions):
        new_local_axis_map[gax] = new_pos

    return out.astype(np.float32, copy=False), new_local_axis_map


class PatchAugmentationPipeline:
    """
    Patch-level augmentation pipeline operating on STRUCTURED patches.

    Input:
      img:
        - without channel_axis: (2, *spatial)
        - with channel_axis   : (2, C, *spatial)

      local_axis_map:
        dict mapping GLOBAL axes -> local structured axes

    Returns:
      img_aug, local_axis_map_aug
    """

    def __init__(self, cfg_aug):
        self.cfg = cfg_aug

    def __call__(
        self,
        img: np.ndarray,
        local_axis_map: Dict[int, int],
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        out = img.astype(np.float32, copy=False)
        axis_map = dict(local_axis_map)

        if self.cfg is None:
            return out, axis_map

        if not getattr(self.cfg, "enabled", True):
            return out, axis_map

        # 1) global phase
        if (
            self.cfg.global_phase.enabled
            and np.random.rand() < self.cfg.global_phase.p
        ):
            out = apply_global_phase(out)

        # 2) global scale
        if (
            self.cfg.global_scale.enabled
            and np.random.rand() < self.cfg.global_scale.p
        ):
            out = apply_global_scale(
                out,
                scale_min=self.cfg.global_scale.min,
                scale_max=self.cfg.global_scale.max,
            )

        # 3) inversion
        if (
            self.cfg.inversion.enabled
            and np.random.rand() < self.cfg.inversion.p
        ):
            out = apply_inversion(
                out,
                local_axis_map=axis_map,
                axes_global=tuple(self.cfg.inversion.axes),
            )

        # 4) permutation
        if (
            self.cfg.permutation.enabled
            and np.random.rand() < self.cfg.permutation.p
        ):
            out, axis_map = apply_permutation(
                out,
                local_axis_map=axis_map,
                axes_global=tuple(self.cfg.permutation.axes),
            )

        return out.astype(np.float32, copy=False), axis_map
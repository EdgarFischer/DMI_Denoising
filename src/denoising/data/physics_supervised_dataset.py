"""Supervised spatial patches paired with classical-forD parameter maps."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PhysicsSupervisedDataset(Dataset):
    """Load globally normalized WALINET FIDs and aligned forD parameter maps."""

    def __init__(
        self,
        *,
        base_dir: str | Path,
        subjects: list[str],
        data_filename: str,
        mask_filename: str,
        target_dirname: str,
        metabolite_names: tuple[str, ...],
        num_samples: int,
        patch_size: tuple[int, int],
        hz_per_ppm: float = 297.22,
        ppm_reference: float = 4.65,
    ) -> None:
        super().__init__()
        if not subjects:
            raise ValueError("subjects must not be empty.")
        if len(metabolite_names) == 0:
            raise ValueError("metabolite_names must not be empty.")
        self.num_samples = int(num_samples)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.metabolite_names = tuple(metabolite_names)
        self.subjects = []

        base = Path(base_dir)
        for subject_name in subjects:
            subject = base / subject_name
            fid = np.load(subject / data_filename).astype(np.complex64, copy=False)
            if fid.ndim != 4:
                raise ValueError(f"Expected 4D FID for {subject_name}, got {fid.shape}.")
            scale = float(np.max(np.abs(fid)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid FID max-abs for {subject_name}: {scale}.")
            normalized = fid / scale
            spectra = np.fft.fftshift(
                np.fft.fft(normalized, axis=-1), axes=-1
            ).astype(np.complex64)
            brain_mask = np.load(subject / mask_filename).astype(bool)
            if brain_mask.shape != fid.shape[:3]:
                raise ValueError(
                    f"Mask shape {brain_mask.shape} does not match {fid.shape[:3]}."
                )
            target_dir = subject / target_dirname
            targets = self._load_targets(
                target_dir,
                self.metabolite_names,
                hz_per_ppm=float(hz_per_ppm),
                ppm_reference=float(ppm_reference),
            )
            if targets.shape[1:] != brain_mask.shape:
                raise ValueError(
                    f"Target shape {targets.shape[1:]} does not match mask "
                    f"{brain_mask.shape}: {target_dir}"
                )
            finite = np.all(np.isfinite(targets), axis=0)
            valid_mask = brain_mask & finite
            valid_coordinates = np.argwhere(valid_mask)
            if len(valid_coordinates) == 0:
                raise ValueError(f"No finite in-mask targets found: {target_dir}")
            self.subjects.append(
                (spectra, np.nan_to_num(targets), valid_mask, valid_coordinates)
            )

    @staticmethod
    def _squeeze_spatial(array: np.ndarray, expected_ndim: int) -> np.ndarray:
        while array.ndim > expected_ndim and array.shape[0] == 1:
            array = array[0]
        return array

    @classmethod
    def _load_targets(
        cls,
        target_dir: Path,
        metabolite_names: tuple[str, ...],
        *,
        hz_per_ppm: float,
        ppm_reference: float,
    ) -> np.ndarray:
        metadata_path = target_dir / "parameter_maps_metadata.json"
        combined_path = target_dir / "metabolite_maps" / "metabolite_maps.npy"
        amplitudes = None
        if metadata_path.is_file() and combined_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_names = tuple(metadata["metabolite_axis_order"])
            combined = cls._squeeze_spatial(np.load(combined_path), 4)
            if combined.ndim != 4 or combined.shape[-1] != len(source_names):
                raise ValueError(f"Invalid combined metabolite map: {combined_path}")
            missing = [name for name in metabolite_names if name not in source_names]
            if missing:
                raise KeyError(
                    f"Teacher fit lacks requested metabolites {missing}: {target_dir}"
                )
            amplitudes = np.stack(
                [combined[..., source_names.index(name)] for name in metabolite_names],
                axis=0,
            )
        else:
            amplitudes = np.stack(
                [
                    cls._squeeze_spatial(
                        np.load(target_dir / "metabolite_maps" / f"{name}.npy"), 3
                    )
                    for name in metabolite_names
                ],
                axis=0,
            )

        nuisance = target_dir / "nuisance_parameter_maps"
        load = lambda name: cls._squeeze_spatial(np.load(nuisance / f"{name}.npy"), 3)
        ford_delta_f = load("delta_f")
        ford_phase0 = load("delta_phi_0")
        ford_phase1_ppm = load("delta_phi_1")
        ford_lorentz = load("lorentzian_damping")
        ford_gaussian = load("gaussian_damping")

        # Convert the classical-forD SignalModel convention to the WALINET
        # decoder convention used by PhysicsConv3D.
        frequency_shift_hz = -ford_delta_f
        lorentzian_fwhm_hz = ford_lorentz / math.pi
        gaussian_fwhm_hz = np.sqrt(
            np.maximum(ford_gaussian, 0.0) * (4.0 * math.log(2.0))
        ) / math.pi
        phase1_rad_per_hz = ford_phase1_ppm / hz_per_ppm
        phase0_radians = ford_phase0 + ford_phase1_ppm * ppm_reference

        return np.concatenate(
            (
                amplitudes,
                frequency_shift_hz[None],
                lorentzian_fwhm_hz[None],
                gaussian_fwhm_hz[None],
                phase0_radians[None],
                phase1_rad_per_hz[None],
            ),
            axis=0,
        ).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        spectra, targets, valid_mask, coordinates = self.subjects[
            np.random.randint(len(self.subjects))
        ]
        x, y, z = coordinates[np.random.randint(len(coordinates))]
        patch_x, patch_y = self.patch_size
        x0 = int(np.clip(x - np.random.randint(patch_x), 0, spectra.shape[0] - patch_x))
        y0 = int(np.clip(y - np.random.randint(patch_y), 0, spectra.shape[1] - patch_y))
        spectrum = spectra[x0:x0 + patch_x, y0:y0 + patch_y, z]
        target = targets[:, x0:x0 + patch_x, y0:y0 + patch_y, z]
        mask = valid_mask[x0:x0 + patch_x, y0:y0 + patch_y, z]
        network_input = np.stack((spectrum.real, spectrum.imag), axis=0)
        return (
            torch.from_numpy(network_input.astype(np.float32, copy=False)),
            torch.from_numpy(target.astype(np.float32, copy=False)),
            torch.from_numpy(mask.astype(np.float32, copy=False)),
        )

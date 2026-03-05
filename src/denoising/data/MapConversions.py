from pathlib import Path
from typing import Iterable, Mapping, Sequence, Optional, Tuple, List, Dict

import numpy as np
import nibabel as nib

def convert_metabmaps(
    data_dir: str = "MetabMaps",
    model_base: str = "",
    *,
    # --- Metabolite maps (amp/sd) ---
    do_metabolites: bool = True,
    metabolites: Iterable[str] = ("water", "Glc", "Glx", "Lac"),
    variants: Mapping[str, str] = None,          # folder_name -> suffix
    amp_timepoints: Sequence[int] = tuple(range(1, 9)),  # 1..8
    save_sd_only_for_orig: bool = True,

    # --- Other maps (FWHM/SNR) + Spectra ---
    do_fwhm: bool = True,
    do_snr: bool = True,
    do_spectra: bool = True,
    other_timepoints: Sequence[int] = tuple(range(1, 11)),  # 1..10

    # --- IO / consistency ---
    canonical: bool = True,          # nib.as_closest_canonical
    check_affines: bool = True,
    affine_atol: float = 1e-5,
    out_dtype: np.dtype = np.float32,

    # --- Cleanup ---
    delete_sources: bool = False,
    delete_all_mnc_in_data_dir: bool = False,    # careful: deletes ALL *.mnc in data_dir

    # --- Logging ---
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Convert various MetabMaps files into stacked .npy arrays.

    Returns a dict with keys like 'amp_water', 'sd_water', 'FWHM', 'SNR', 'Spectra'
    mapped to the saved .npy output paths (only for outputs that were written).
    """
    if variants is None:
        variants = {
            "Orig": "Orig",
            "QualityAndOutlier_Clip": "QualityClip",
            "Outlier_Clip": "OutlierClip",
        }

    data_dir = Path(data_dir)
    output_dir = data_dir / model_base
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        if verbose:
            print(msg)

    def load_img(path: Path):
        img = nib.load(str(path))
        return nib.as_closest_canonical(img) if canonical else img

    def affine_equal(a, b) -> bool:
        return np.allclose(a, b, atol=affine_atol)

    written: Dict[str, Path] = {}
    to_delete: List[Path] = []

    # -------------------------
    # Metabolites: AMP + SD
    # -------------------------
    if do_metabolites:
        for folder, suffix in variants.items():
            model = f"{model_base}_{suffix}" if model_base else f"{suffix}"

            for metabolite in metabolites:
                amp_list: List[np.ndarray] = []
                sd_list: List[np.ndarray] = []
                amp_srcs: List[Path] = []
                sd_srcs: List[Path] = []

                # Option: only load SD for Orig
                load_sd = (folder == "Orig") if save_sd_only_for_orig else True

                for i in amp_timepoints:
                    amp_file = data_dir / f"{metabolite}_amp_map_{i}_{suffix}.mnc"
                    sd_file  = data_dir / f"{metabolite}_sd_map_{i}_{suffix}.mnc"

                    if amp_file.exists():
                        amp_img = load_img(amp_file)
                        amp_list.append(amp_img.get_fdata())
                        amp_srcs.append(amp_file)
                    else:
                        _log(f"❌ Datei fehlt: {amp_file}")

                    if load_sd:
                        if sd_file.exists():
                            sd_img = load_img(sd_file)
                            sd_list.append(sd_img.get_fdata())
                            sd_srcs.append(sd_file)
                        else:
                            _log(f"❌ Datei fehlt: {sd_file}")

                # Save AMP
                if amp_list:
                    amp_stacked = np.stack(amp_list, axis=-1).astype(out_dtype)
                    _log(f"{metabolite} AMP shape ({suffix}): {amp_stacked.shape}")
                    out_path = output_dir / f"{metabolite}_amp_{model}.npy"
                    np.save(out_path, amp_stacked)
                    written[f"amp_{metabolite}_{suffix}"] = out_path
                    if delete_sources:
                        to_delete.extend(amp_srcs)

                # Save SD
                if load_sd and sd_list:
                    sd_stacked = np.stack(sd_list, axis=-1).astype(out_dtype)
                    _log(f"{metabolite} SD shape ({suffix}): {sd_stacked.shape}")
                    out_path = output_dir / f"{metabolite}_sd_{model}.npy"
                    np.save(out_path, sd_stacked)
                    written[f"sd_{metabolite}_{suffix}"] = out_path
                    if delete_sources:
                        to_delete.extend(sd_srcs)

    # -------------------------
    # FWHM / SNR helpers
    # -------------------------
    def _stack_mnc_series(prefix: str, timepoints: Sequence[int], out_name: str) -> Optional[Path]:
        arrs: List[np.ndarray] = []
        srcs: List[Path] = []
        aff0 = None

        for t in timepoints:
            fp = data_dir / f"{prefix}_map_{t}.mnc"
            if not fp.exists():
                _log(f"❌ fehlt: {fp}")
                continue
            img = load_img(fp)

            if check_affines:
                if aff0 is None:
                    aff0 = img.affine
                elif not affine_equal(img.affine, aff0):
                    _log(f"⚠️  AFFINE-Mismatch ({prefix}) bei t={t}")

            arrs.append(img.get_fdata())
            srcs.append(fp)

        if not arrs:
            return None

        stacked = np.stack(arrs, axis=-1).astype(out_dtype)
        _log(f"{prefix} shape: {stacked.shape}")
        out_path = output_dir / f"{out_name}_{model_base}.npy"
        np.save(out_path, stacked)

        if delete_sources:
            to_delete.extend(srcs)

        return out_path

    # -------------------------
    # FWHM
    # -------------------------
    if do_fwhm:
        p = _stack_mnc_series("FWHM", other_timepoints, "FWHM")
        if p is not None:
            written["FWHM"] = p

    # -------------------------
    # SNR
    # -------------------------
    if do_snr:
        p = _stack_mnc_series("SNR", other_timepoints, "SNR")
        if p is not None:
            written["SNR"] = p

    # -------------------------
    # Spectra (nii.gz)
    # -------------------------
    if do_spectra:
        spec_list: List[np.ndarray] = []
        spec_srcs: List[Path] = []
        aff0 = None
        shapes: List[Tuple[int, ...]] = []

        for t in other_timepoints:
            fp = data_dir / f"SpecMap_LCMInput_{t}.nii.gz"
            if not fp.exists():
                _log(f"❌ fehlt: {fp}")
                continue
            img = load_img(fp)
            data = img.get_fdata()
            shapes.append(tuple(data.shape))

            if check_affines:
                if aff0 is None:
                    aff0 = img.affine
                elif not affine_equal(img.affine, aff0):
                    _log(f"⚠️  AFFINE-Mismatch (Spec) bei t={t}")

            spec_list.append(data)
            spec_srcs.append(fp)

        if spec_list:
            if len(set(shapes)) != 1:
                _log(f"⚠️  Uneinheitliche Spektren-Shapes: {shapes}")

            spec5d = np.stack(spec_list, axis=-1).astype(out_dtype)
            _log(f"Spectra 5D shape: {spec5d.shape}")
            out_path = output_dir / f"Spectra_{model_base}.npy"
            np.save(out_path, spec5d)
            written["Spectra"] = out_path

            if delete_sources:
                to_delete.extend(spec_srcs)

    # -------------------------
    # Cleanup
    # -------------------------
    # Delete only collected sources
    if delete_sources and to_delete:
        for p in to_delete:
            try:
                p.unlink()
                _log(f"🗑️ gelöscht: {p}")
            except Exception as e:
                _log(f"⚠️  Konnte nicht löschen: {p} ({e})")

    # Delete ALL .mnc in data_dir (dangerous)
    if delete_all_mnc_in_data_dir:
        for p in data_dir.glob("*.mnc"):
            try:
                p.unlink()
                _log(f"🗑️ Gelöscht: {p}")
            except Exception as e:
                _log(f"⚠️  Konnte nicht löschen: {p} ({e})")

    _log("✅ Fertig.")
    return written
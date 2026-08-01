"""Build the fixed decoder basis from a WALINET simulation config."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from .basis_decoder import BaselineFreeBasisDecoder


def _active_profile_basis_names(config) -> set[str]:
    """Return the union of enabled basis components across all profiles."""
    active: set[str] = set()
    for profile in config.metabolites.profiles:
        profile_path = Path(profile.config)
        with profile_path.open("r", encoding="utf-8") as file:
            profile_raw = yaml.safe_load(file)
        metabolites = profile_raw.get("metabolites")
        if not isinstance(metabolites, dict):
            raise TypeError(
                f"Metabolite profile has no 'metabolites' mapping: {profile_path}"
            )
        for entry in metabolites.values():
            if not isinstance(entry, dict) or not bool(entry.get("enabled", True)):
                continue
            basis_component = entry.get("basis_component")
            if basis_component is None or not str(basis_component).strip():
                raise ValueError(
                    "Enabled metabolite has no basis_component in "
                    f"{profile_path}."
                )
            active.add(str(basis_component).strip())
    if not active:
        raise ValueError("No active metabolites found in configured profiles.")
    return active


def decoder_from_walinet_simulation_config(
    simulation_config_path: str | Path,
    *,
    dataset_name: str = "clean_fid",
    active_metabolites_only: bool = True,
    basis_components: tuple[str, ...] | None = None,
) -> tuple[BaselineFreeBasisDecoder, tuple[str, ...]]:
    """Prepare the acquisition-matched LCModel basis used by WALINET."""
    path = Path(simulation_config_path).expanduser().resolve()

    # A virtual environment called ``walinet`` does not necessarily mean the
    # sibling project was installed into it. The simulation-config path is an
    # unambiguous anchor: locate that project's ``src/walinet`` directory and
    # make it importable without requiring a separate editable installation.
    try:
        from walinet.config.build_simulation import build_simulation_config
        from walinet.training_data.lcmodel_basis.acquisition import (
            prepare_basis_for_acquisition,
        )
    except ModuleNotFoundError as error:
        if error.name != "walinet":
            raise
        walinet_source = next(
            (
                ancestor / "src"
                for ancestor in path.parents
                if (ancestor / "src" / "walinet").is_dir()
            ),
            None,
        )
        if walinet_source is None:
            raise ImportError(
                "Could not locate src/walinet from simulation config "
                f"{path}. Install WALINET or use a config inside its project."
            ) from error
        sys.path.insert(0, str(walinet_source))
        from walinet.config.build_simulation import build_simulation_config
        from walinet.training_data.lcmodel_basis.acquisition import (
            prepare_basis_for_acquisition,
        )

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    config = build_simulation_config(raw, config_dir=path.parent)
    prepared = prepare_basis_for_acquisition(
        config.basis.library,
        target_bandwidth=config.acquisition.bandwidth_hz,
        target_n_timepoints=config.acquisition.n_timepoints,
        dataset_name=dataset_name,
    )
    prepared_names = tuple(str(name) for name in prepared.names)
    selected_indices = list(range(len(prepared_names)))
    selected_names = prepared_names
    if basis_components is not None:
        requested = set(basis_components)
        missing = sorted(requested.difference(prepared_names))
        if missing:
            raise KeyError(
                "Requested basis components missing from prepared basis: "
                + ", ".join(missing)
            )
        selected_indices = [prepared_names.index(name) for name in basis_components]
        selected_names = tuple(basis_components)
        print(
            "Selected explicit basis components: "
            f"{len(selected_names)}/{len(prepared_names)}\n  "
            + ", ".join(selected_names)
        )
    elif active_metabolites_only:
        active_names = _active_profile_basis_names(config)
        missing = sorted(active_names.difference(prepared_names))
        if missing:
            raise KeyError(
                "Active simulation metabolites missing from prepared basis: "
                + ", ".join(missing)
            )
        selected_indices = [
            index for index, name in enumerate(prepared_names)
            if name in active_names
        ]
        selected_names = tuple(prepared_names[index] for index in selected_indices)
        print(
            "Selected active WALINET metabolites: "
            f"{len(selected_names)}/{len(prepared_names)}\n  "
            + ", ".join(selected_names)
        )

    basis = torch.from_numpy(
        np.ascontiguousarray(
            prepared.fids[selected_indices], dtype=np.complex64
        )
    )
    return (
        BaselineFreeBasisDecoder(basis, prepared.dwell_time),
        selected_names,
    )

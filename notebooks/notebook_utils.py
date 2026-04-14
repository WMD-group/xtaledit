"""Shared utilities for analysis notebooks."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from notebook_constants import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    EHULL_RELAXED_FILE,
    EHULL_RELAXED_INFO_FILE,
    GEN_FILE,
    METASTABLE_EHULL_MAX,
    RELAXED_SM_ANON_ENTRIES_FILE,
    RELAXED_SM_ANON_MATCHES_FILE,
    RELAXED_WYCKOFF_ENTRIES_FILE,
    RELAXED_WYCKOFF_MATCHES_FILE,
    SM_ANON_MATCH_FILE,
    SM_FIT_FILE,
    SMACT_VALIDITY_FILE,
    SOURCE_ORDER,
    TOP3_SM_ANON_FILE,
    TOP3_WYCKOFF_FILE,
    TRAIN_FILE,
    WYCKOFF_MATCH_FILE,
    WYCKOFF_REPR_FILE,
)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or shell working directory."""
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


def load_pickle_gz(path: Path) -> Any:
    """Load a gzip-compressed pickle file."""
    with gzip.open(path, "rb") as file:
        return pickle.load(file)  # noqa: S301


def validate_1d_length(values: np.ndarray, expected: int, path: Path) -> np.ndarray:
    """Validate that an array is one-dimensional and has the expected length."""
    if values.ndim != 1 or len(values) != expected:
        raise ValueError(
            f"{path} has unexpected shape {values.shape}; expected ({expected},)"
        )
    return values


def load_relaxed_ehulls(path: Path, expected: int) -> np.ndarray:
    """Load relaxed energy-above-hull values."""
    values = np.asarray(load_pickle_gz(path), dtype=float)
    return validate_1d_length(values, expected, path)


def load_relax_convergence(path: Path, expected: int) -> np.ndarray:
    """Load per-structure relaxation convergence flags."""
    infos = load_pickle_gz(path)
    if len(infos) != expected:
        raise ValueError(
            f"{path} has unexpected length {len(infos)}; expected {expected}"
        )
    return np.array(
        [info is not None and bool(info.get("converged", False)) for info in infos],
        dtype=bool,
    )


def load_smact_validity(path: Path, expected: int) -> np.ndarray:
    """Load SMACT validity flags from an npz file."""
    with np.load(path) as data:
        if "valid" not in data.files:
            raise ValueError(f"{path} does not contain a 'valid' array")
        values = np.asarray(data["valid"], dtype=bool)
    return validate_1d_length(values, expected, path)


def load_direct_matches(path: Path) -> pd.DataFrame:
    """Load direct StructureMatcher match pairs."""
    with np.load(path) as data:
        if "matches" not in data.files:
            raise ValueError(f"{path} does not contain a 'matches' array")
        matches = data["matches"]

    if matches.ndim != 2 or matches.shape[1] != 2:
        raise ValueError(
            f"{path} has unexpected shape {matches.shape}; expected (n, 2)"
        )
    return pd.DataFrame(matches, columns=["gen_idx", "train_idx"]).astype(int)


def load_direct_match_gen_indices(path: Path) -> set[int]:
    """Load generated-structure indices with a direct match."""
    return set(int(i) for i in load_direct_matches(path)["gen_idx"])


def load_relaxed_match_gen_indices(path: Path) -> set[int]:
    """Load generated-structure indices with a relaxed substituted match."""
    records = load_pickle_gz(path)
    return {
        int(record["gen_idx"]) for record in records if bool(record.get("match", False))
    }


def load_wyckoff_data(path: Path, expected: int) -> list[Any]:
    """Load Wyckoff representation data and validate its length."""
    data = load_pickle_gz(path)
    if len(data) != expected:
        raise ValueError(
            f"{path} has unexpected length {len(data)}; expected {expected}"
        )
    return data


def load_generated_count(model: str, input_dir: Path) -> int:
    """Return the number of generated structures for a model."""
    path = input_dir / "gen" / "preprocessed" / model / GEN_FILE
    return len(load_pickle_gz(path))


def crystal_system_from_spg_num(spg_num: int) -> str:
    """Map an international space-group number to a crystal system."""
    match spg_num:
        case n if 1 <= n <= 2:
            return "triclinic"
        case n if 3 <= n <= 15:
            return "monoclinic"
        case n if 16 <= n <= 74:
            return "orthorhombic"
        case n if 75 <= n <= 142:
            return "tetragonal"
        case n if 143 <= n <= 167:
            return "trigonal"
        case n if 168 <= n <= 194:
            return "hexagonal"
        case n if 195 <= n <= 230:
            return "cubic"
        case _:
            raise ValueError(f"Invalid space group number: {spg_num}")


def simple_category(category: str) -> str:
    """Collapse detailed category labels into categories 1, 2, and 3."""
    if category == "1":
        return "1"
    if category.startswith("2"):
        return "2"
    if category == "3":
        return "3"
    raise ValueError(f"Unknown category: {category!r}")


def required_paths(
    model: str,
    input_dir: Path,
    results_dir: Path,
    keys: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Build required input/result paths for a model."""
    gen_dir = input_dir / "gen" / "preprocessed" / model
    result_dir = results_dir / model
    train_dir = input_dir / "train" / "preprocessed"
    all_paths = {
        "generated_structures": gen_dir / GEN_FILE,
        "training_structures": train_dir / TRAIN_FILE,
        "relax_infos": gen_dir / EHULL_RELAXED_INFO_FILE,
        "relaxed_ehull": gen_dir / EHULL_RELAXED_FILE,
        "smact_validity": gen_dir / SMACT_VALIDITY_FILE,
        "direct_sm": result_dir / SM_FIT_FILE,
        "sm_anon_matches": result_dir / SM_ANON_MATCH_FILE,
        "wyckoff_matches": result_dir / WYCKOFF_MATCH_FILE,
        "top3_sm_anon": result_dir / TOP3_SM_ANON_FILE,
        "top3_wyckoff": result_dir / TOP3_WYCKOFF_FILE,
        "relaxed_sm_anon_entries": result_dir / RELAXED_SM_ANON_ENTRIES_FILE,
        "relaxed_wyckoff_entries": result_dir / RELAXED_WYCKOFF_ENTRIES_FILE,
        "relaxed_sm_anon_matches": result_dir / RELAXED_SM_ANON_MATCHES_FILE,
        "relaxed_wyckoff_matches": result_dir / RELAXED_WYCKOFF_MATCHES_FILE,
        "wyckoff_repr": result_dir / WYCKOFF_REPR_FILE,
    }
    if keys is None:
        return all_paths
    return {key: all_paths[key] for key in keys}


def missing_required_paths(
    paths: dict[str, Path],
    *,
    model: str | None = None,
) -> list[dict[str, str]]:
    """Return records for required paths that do not exist."""
    return [
        {"model": model, "kind": kind, "path": str(path)}
        if model is not None
        else {"kind": kind, "path": str(path)}
        for kind, path in paths.items()
        if not path.exists()
    ]


def check_required_paths(paths: dict[str, Path]) -> None:
    """Raise a FileNotFoundError when any required paths are missing."""
    missing = {kind: path for kind, path in paths.items() if not path.exists()}
    if missing:
        missing_lines = "\n".join(f"- {kind}: {path}" for kind, path in missing.items())
        raise FileNotFoundError(f"Missing required input files:\n{missing_lines}")


def classify_model(
    model: str,
    paths: dict[str, Path],
    *,
    include_category_label: bool = False,
    include_crystal_system: bool = False,
    include_simple_category: bool = False,
    simple_category_labels: dict[str, str] | None = None,
    simple_category_order: list[str] | None = None,
) -> pd.DataFrame:
    """Classify generated structures into novelty categories."""
    n_generated = len(load_pickle_gz(paths["generated_structures"]))
    relaxed_ehulls = load_relaxed_ehulls(paths["relaxed_ehull"], n_generated)
    relax_converged = load_relax_convergence(
        paths.get(
            "relax_infos",
            paths["relaxed_ehull"].with_name(EHULL_RELAXED_INFO_FILE),
        ),
        n_generated,
    )
    smact_validity = load_smact_validity(paths["smact_validity"], n_generated)
    direct = load_direct_match_gen_indices(paths["direct_sm"])
    relaxed_sm_anon = load_relaxed_match_gen_indices(paths["relaxed_sm_anon_matches"])
    relaxed_wyckoff = load_relaxed_match_gen_indices(paths["relaxed_wyckoff_matches"])
    wyckoff_data = (
        load_wyckoff_data(paths["wyckoff_repr"], n_generated)
        if include_crystal_system
        else None
    )

    rows = []
    for gen_idx in range(n_generated):
        is_direct = gen_idx in direct
        has_sm_anon = gen_idx in relaxed_sm_anon
        has_wyckoff = gen_idx in relaxed_wyckoff
        ehull_relaxed = float(relaxed_ehulls[gen_idx])
        is_relax_converged = bool(relax_converged[gen_idx])
        is_metastable = is_relax_converged and ehull_relaxed <= METASTABLE_EHULL_MAX
        is_smact_valid = bool(smact_validity[gen_idx])

        if is_direct:
            category = "1"
        elif has_sm_anon and has_wyckoff:
            category = "2-1"
        elif has_sm_anon:
            category = "2-2"
        elif has_wyckoff:
            category = "2-3"
        else:
            category = "3"

        row = {
            "model": model,
            "gen_idx": gen_idx,
            "category": category,
            "is_direct_sm_match": is_direct,
            "has_relaxed_sm_anon_match": has_sm_anon,
            "has_relaxed_wyckoff_match": has_wyckoff,
            "ehull_relaxed": ehull_relaxed,
            "is_relax_converged": is_relax_converged,
            "is_metastable": is_metastable,
            "is_smact_valid": is_smact_valid,
            "is_metastable_smact_valid": is_metastable and is_smact_valid,
        }
        if include_category_label:
            row["category_label"] = CATEGORY_LABELS[category]
        if include_simple_category:
            simplified = simple_category(category)
            row["simple_category"] = simplified
            if simple_category_labels is not None:
                row["simple_category_label"] = simple_category_labels[simplified]
        if include_crystal_system:
            assert wyckoff_data is not None
            row["crystal_system"] = crystal_system_from_spg_num(
                int(wyckoff_data[gen_idx].spg_num)
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["category"] = pd.Categorical(
        frame["category"], categories=CATEGORY_ORDER, ordered=True
    )
    if include_simple_category and simple_category_order is not None:
        frame["simple_category"] = pd.Categorical(
            frame["simple_category"], categories=simple_category_order, ordered=True
        )
    return frame


def entries_to_frame(
    entries: list[Any],
    records: list[dict[str, Any]],
    source: str,
    *,
    include_structure: bool = False,
    add_match_label: bool = False,
) -> pd.DataFrame:
    """Join substituted entries with relaxed-match records."""
    if len(entries) != len(records):
        raise ValueError(
            f"{source}: entries length {len(entries)} != records length {len(records)}"
        )

    rows: list[dict[str, Any]] = []
    for entry_idx, (entry, record) in enumerate(zip(entries, records, strict=True)):
        if int(record["entry_idx"]) != entry_idx:
            raise ValueError(
                f"{source}: record entry_idx={record['entry_idx']} at position "
                f"{entry_idx}"
            )
        for attr in ("gen_idx", "train_idx", "rank"):
            if int(getattr(entry, attr)) != int(record[attr]):
                raise ValueError(f"{source}: {attr} mismatch at entry {entry_idx}")
        for attr in ("cost_uniform", "cost_mod_petti"):
            if not np.isclose(float(getattr(entry, attr)), float(record[attr])):
                raise ValueError(f"{source}: {attr} mismatch at entry {entry_idx}")

        row = {
            "source": source,
            "source_order": SOURCE_ORDER.index(source),
            "entry_idx": entry_idx,
            "gen_idx": int(entry.gen_idx),
            "train_idx": int(entry.train_idx),
            "rank": int(entry.rank),
            "cost_uniform": float(entry.cost_uniform),
            "cost_mod_petti": float(entry.cost_mod_petti),
            "match": bool(record["match"]),
        }
        if include_structure:
            row["relaxed_substituted_structure"] = entry.structure
        rows.append(row)

    frame = pd.DataFrame(rows)
    if add_match_label:
        frame["match_label"] = np.where(
            frame["match"], "relaxed match", "no relaxed match"
        )
    return frame

from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from notebook_utils import (  # noqa: E402
    check_required_paths,
    classify_model,
    crystal_system_from_spg_num,
    entries_to_frame,
    load_direct_match_gen_indices,
    load_direct_matches,
    load_pickle_gz,
    load_relax_convergence,
    load_relaxed_match_gen_indices,
    load_smact_validity,
    required_paths,
    simple_category,
    substituted_relax_info_is_complete,
    validate_1d_length,
)


def dump_pickle_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as file:
        pickle.dump(obj, file)


def test_loaders_validate_shapes(tmp_path: Path) -> None:
    pickle_path = tmp_path / "values.pkl.gz"
    dump_pickle_gz(pickle_path, [1, 2, 3])
    assert load_pickle_gz(pickle_path) == [1, 2, 3]

    values = validate_1d_length(np.array([1, 2, 3]), 3, pickle_path)
    np.testing.assert_array_equal(values, np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="unexpected shape"):
        validate_1d_length(np.array([[1, 2, 3]]), 3, pickle_path)

    smact_path = tmp_path / "smact_validity.npz"
    np.savez(smact_path, valid=np.array([1, 0, 1]))
    np.testing.assert_array_equal(
        load_smact_validity(smact_path, 3), np.array([True, False, True])
    )

    direct_path = tmp_path / "sm_fit.npz"
    np.savez(direct_path, matches=np.array([[0, 10], [2, 12]]))
    pd.testing.assert_frame_equal(
        load_direct_matches(direct_path),
        pd.DataFrame({"gen_idx": [0, 2], "train_idx": [10, 12]}),
    )
    assert load_direct_match_gen_indices(direct_path) == {0, 2}

    relax_infos_path = tmp_path / "relax_infos.pkl.gz"
    dump_pickle_gz(
        relax_infos_path,
        [{"converged": True}, {"converged": False}, None],
    )
    np.testing.assert_array_equal(
        load_relax_convergence(relax_infos_path, 3),
        np.array([True, False, False]),
    )

    relaxed_path = tmp_path / "substituted_relaxed_gen_matches.pkl.gz"
    relaxed_infos_path = tmp_path / "substituted_relaxed_infos.pkl.gz"
    dump_pickle_gz(
        relaxed_path,
        [
            {"entry_idx": 0, "gen_idx": 0, "match": True},
            {"entry_idx": 1, "gen_idx": 1, "match": True},
            {"entry_idx": 2, "gen_idx": 2, "match": True},
        ],
    )
    dump_pickle_gz(
        relaxed_infos_path,
        [
            {"converged": True, "n_steps": 10, "energy_per_atom_eV": -1.0},
            {"converged": False, "n_steps": 1000, "energy_per_atom_eV": -2.0},
            None,
        ],
    )
    assert load_relaxed_match_gen_indices(relaxed_path) == {0, 1}
    assert substituted_relax_info_is_complete(
        {"converged": False, "n_steps": 1000, "energy_per_atom_eV": -2.0}
    )
    assert not substituted_relax_info_is_complete({"converged": True})


def test_required_paths_and_missing_check(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    paths = required_paths(
        "mattergen",
        input_dir,
        results_dir,
        ["generated_structures", "training_structures", "direct_sm"],
    )

    assert paths == {
        "generated_structures": (
            input_dir / "gen" / "preprocessed" / "mattergen" / "relaxed_niggli.pkl.gz"
        ),
        "training_structures": input_dir / "train" / "preprocessed" / "train.pkl.gz",
        "direct_sm": results_dir / "mattergen" / "sm_fit.npz",
    }
    with pytest.raises(FileNotFoundError, match="Missing required input files"):
        check_required_paths(paths)


@pytest.mark.parametrize(
    ("spg_num", "expected"),
    [
        (1, "triclinic"),
        (15, "monoclinic"),
        (74, "orthorhombic"),
        (142, "tetragonal"),
        (167, "trigonal"),
        (194, "hexagonal"),
        (230, "cubic"),
    ],
)
def test_crystal_system_from_spg_num(spg_num: int, expected: str) -> None:
    assert crystal_system_from_spg_num(spg_num) == expected


def test_category_helpers_reject_unknown_values() -> None:
    assert simple_category("2-3") == "2"
    with pytest.raises(ValueError, match="Unknown category"):
        simple_category("x")
    with pytest.raises(ValueError, match="Invalid space group number"):
        crystal_system_from_spg_num(0)


def test_classify_model_uses_shared_category_logic(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    paths = required_paths("model", input_dir, results_dir)

    dump_pickle_gz(paths["generated_structures"], list(range(5)))
    dump_pickle_gz(paths["relaxed_ehull"], [0.0, 0.2, 0.03, 0.04, 0.05])
    dump_pickle_gz(
        paths["relax_infos"],
        [
            {"converged": True},
            {"converged": True},
            {"converged": True},
            {"converged": False},
            None,
        ],
    )
    np.savez(paths["smact_validity"], valid=np.array([1, 0, 1, 1, 1]))
    paths["direct_sm"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths["direct_sm"], matches=np.array([[0, 10]]))
    dump_pickle_gz(
        paths["relaxed_sm_anon_matches"],
        [
            {"entry_idx": 0, "gen_idx": 1, "match": True},
            {"entry_idx": 1, "gen_idx": 2, "match": True},
            {"entry_idx": 2, "gen_idx": 4, "match": True},
        ],
    )
    dump_pickle_gz(
        paths["relaxed_sm_anon_infos"],
        [
            {"converged": False, "n_steps": 1000, "energy_per_atom_eV": -1.0},
            {"converged": True, "n_steps": 10, "energy_per_atom_eV": -1.0},
            None,
        ],
    )
    dump_pickle_gz(
        paths["relaxed_wyckoff_matches"],
        [
            {"entry_idx": 0, "gen_idx": 1, "match": True},
            {"entry_idx": 1, "gen_idx": 3, "match": True},
        ],
    )
    dump_pickle_gz(
        paths["relaxed_wyckoff_infos"],
        [
            {"converged": True, "n_steps": 10, "energy_per_atom_eV": -1.0},
            {"converged": True, "n_steps": 10, "energy_per_atom_eV": -1.0},
        ],
    )
    dump_pickle_gz(
        paths["wyckoff_repr"],
        [SimpleNamespace(spg_num=n) for n in [1, 3, 16, 75, 230]],
    )

    frame = classify_model(
        "model",
        paths,
        include_category_label=True,
        include_crystal_system=True,
        include_simple_category=True,
        simple_category_labels={
            "1": "Exact match",
            "2": "Substituted match",
            "3": "No match",
        },
        simple_category_order=["1", "2", "3"],
    )

    assert frame["category"].astype(str).tolist() == ["1", "2-1", "2-2", "2-3", "3"]
    assert frame["simple_category"].astype(str).tolist() == ["1", "2", "2", "2", "3"]
    assert frame["crystal_system"].tolist() == [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "cubic",
    ]
    assert frame["is_relax_converged"].tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert frame["is_metastable"].tolist() == [True, False, True, False, False]
    assert frame["is_smact_valid"].tolist() == [True, False, True, True, True]


def test_entries_to_frame_options() -> None:
    entries = [
        SimpleNamespace(
            gen_idx=0,
            train_idx=10,
            rank=1,
            cost_uniform=0.1,
            cost_mod_petti=0.2,
            cost_cs=0.3,
            structure="structure",
        )
    ]
    records = [
        {
            "entry_idx": 0,
            "gen_idx": 0,
            "train_idx": 10,
            "rank": 1,
            "cost_uniform": 0.1,
            "cost_mod_petti": 0.2,
            "cost_cs": 0.3,
            "match": True,
        }
    ]

    frame = entries_to_frame(
        entries,
        records,
        "anon",
        infos=[{"converged": False, "n_steps": 1000, "energy_per_atom_eV": -1.0}],
        include_structure=True,
        add_match_label=True,
    )

    assert frame.loc[0, "source_order"] == 0
    assert frame.loc[0, "cost_cs"] == 0.3
    assert frame.loc[0, "relaxed_substituted_structure"] == "structure"
    assert frame.loc[0, "match_label"] == "relaxed match"
    assert not frame.loc[0, "relax_failed"]
    assert not frame.loc[0, "relax_converged"]

    with pytest.raises(ValueError, match="infos length"):
        entries_to_frame(entries, records, "anon", infos=[])

    records[0]["rank"] = 2
    with pytest.raises(ValueError, match="rank mismatch"):
        entries_to_frame(entries, records, "anon")


def test_entries_to_frame_excludes_hard_relaxation_failures() -> None:
    entry = SimpleNamespace(
        gen_idx=0,
        train_idx=10,
        rank=1,
        cost_uniform=0.1,
        cost_mod_petti=0.2,
        structure="structure",
    )
    record = {
        "entry_idx": 0,
        "gen_idx": 0,
        "train_idx": 10,
        "rank": 1,
        "cost_uniform": 0.1,
        "cost_mod_petti": 0.2,
        "match": True,
    }

    frame = entries_to_frame([entry], [record], "anon", infos=[None])

    assert not frame.loc[0, "match"]
    assert np.isnan(frame.loc[0, "cost_cs"])
    assert frame.loc[0, "relax_failed"]
    assert not frame.loc[0, "relax_converged"]

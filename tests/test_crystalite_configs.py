from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from src.config import ROOT

CONFIG_DIR = ROOT / "configs" / "crystalite"
SCRIPT_DIR = ROOT / "scripts"


def _load_script(name: str) -> ModuleType:
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_config_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(monkeypatch, script: str, config: str):
    module = _load_script(script)
    config_path = CONFIG_DIR / config
    monkeypatch.setattr(sys, "argv", [str(module.__file__), str(config_path)])
    return module.parse_args()


def test_crystalite_preprocess_config(monkeypatch) -> None:
    args = _parse(monkeypatch, "preprocess", "01_preprocess.yaml")

    assert args.gen_file == ROOT / "input/gen/raw/crystalite.pkl.gz"


def test_crystalite_exact_match_config(monkeypatch) -> None:
    args = _parse(monkeypatch, "exact_match", "02_exact_match.yaml")

    assert args.model == "crystalite"
    assert args.output == "sm_fit"
    assert args.matcher_kwargs["ltol"] == 0.2
    assert args.matcher_kwargs["primitive_cell"] is False
    assert args.matcher_kwargs["scale"] is False
    assert args.matcher_kwargs["attempt_supercell"] is True
    assert args.fit_kwargs == {
        "symmetric": False,
        "skip_structure_reduction": True,
    }


def test_crystalite_anonymous_match_config(monkeypatch) -> None:
    args = _parse(monkeypatch, "anonymous_match", "03_anonymous_match.yaml")

    assert args.model == "crystalite"
    assert args.output == "sm_anon"
    assert args.jobs == 5
    assert args.timeout_sec == 3600.0
    assert args.matcher_kwargs["scale"] is False


def test_crystalite_wyckoff_match_config(monkeypatch) -> None:
    args = _parse(monkeypatch, "wyckoff_match", "04_wyckoff_match.yaml")

    assert args.model == "crystalite"
    assert args.cost == "mod_petti"
    assert args.symprec == 0.01


def test_crystalite_substitution_config(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "substitute_structures",
        "05_substitute_structures.yaml",
    )

    assert args.model == "crystalite"
    assert args.top_k == 3
    assert args.cost == "mod_petti"
    assert args.sm_anon_input == ROOT / "results/raw/crystalite/sm_anon.pkl.gz"
    assert args.force is False


def test_crystalite_relaxation_config(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "relax_substituted",
        "06_relax_substituted.yaml",
    )

    assert len(args.paths) == 2
    assert args.device == "cuda"
    assert args.fmax == 0.001
    assert args.max_steps == 1000
    assert args.forward_batch_size == 64
    assert args.force is False


def test_crystalite_relaxed_match_config(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "check_relaxed_substituted_matches",
        "07_check_relaxed_matches.yaml",
    )

    assert len(args.paths) == 2
    assert args.gen_path == (
        ROOT / "input/gen/preprocessed/crystalite/relaxed_niggli.pkl.gz"
    )
    assert args.matcher_kwargs["ltol"] == 0.2
    assert args.fit_kwargs["skip_structure_reduction"] is False
    assert args.output is None
    assert args.force is False


def test_crystalite_configs_cover_existing_artifact_names(monkeypatch) -> None:
    exact = _parse(monkeypatch, "exact_match", "02_exact_match.yaml")
    anonymous = _parse(monkeypatch, "anonymous_match", "03_anonymous_match.yaml")
    wyckoff = _parse(monkeypatch, "wyckoff_match", "04_wyckoff_match.yaml")
    substitution = _parse(
        monkeypatch,
        "substitute_structures",
        "05_substitute_structures.yaml",
    )
    relaxation = _parse(
        monkeypatch,
        "relax_substituted",
        "06_relax_substituted.yaml",
    )
    relaxed_match = _parse(
        monkeypatch,
        "check_relaxed_substituted_matches",
        "07_check_relaxed_matches.yaml",
    )
    relax_module = _load_script("relax_substituted")
    check_module = _load_script("check_relaxed_substituted_matches")

    artifact_names = {
        f"{exact.output}.npz",
        f"{anonymous.output}.pkl.gz",
        f"wyckoff_match_s={wyckoff.symprec}_c={wyckoff.cost}.pkl.gz",
        f"wyckoff_repr_s={wyckoff.symprec}.pkl.gz",
        substitution.sm_anon_output.name,
        substitution.wyckoff_output.name,
    }
    for path in relaxation.paths:
        entries_out, infos_out = relax_module.derive_output_paths(path)
        artifact_names.update({entries_out.name, infos_out.name})
    artifact_names.update(
        check_module.derive_output_path(path, relaxed_match.output).name
        for path in relaxed_match.paths
    )

    assert artifact_names == {
        "sm_anon.pkl.gz",
        "sm_fit.npz",
        "substituted_relaxed_niggli_top3_sm_anon.pkl.gz",
        "substituted_relaxed_niggli_top3_sm_anon_gen_matches.pkl.gz",
        "substituted_relaxed_niggli_top3_sm_anon_infos.pkl.gz",
        "substituted_relaxed_niggli_top3_wyckoff_match_s=0.01_c=mod_petti.pkl.gz",
        (
            "substituted_relaxed_niggli_top3_wyckoff_match_s=0.01_c="
            "mod_petti_gen_matches.pkl.gz"
        ),
        (
            "substituted_relaxed_niggli_top3_wyckoff_match_s=0.01_c="
            "mod_petti_infos.pkl.gz"
        ),
        "substituted_top3_sm_anon.pkl.gz",
        "substituted_top3_wyckoff_match_s=0.01_c=mod_petti.pkl.gz",
        "wyckoff_match_s=0.01_c=mod_petti.pkl.gz",
        "wyckoff_repr_s=0.01.pkl.gz",
    }

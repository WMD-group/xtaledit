from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "relax_matched_train.py"
)
SPEC = importlib.util.spec_from_file_location("relax_matched_train", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
relax_matched_train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relax_matched_train)


def test_process_model_deduplicates_and_resumes(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "results" / "raw" / "mattergen"
    model_dir.mkdir(parents=True)
    np.savez_compressed(
        model_dir / "sm_fit.npz",
        matches=np.array([[4, 2], [1, 0], [8, 2]], dtype=np.int32),
    )

    train = [
        Structure(Lattice.cubic(length), ["Si"], [[0, 0, 0]])
        for length in (3.0, 4.0, 5.0)
    ]
    train_path = tmp_path / "train.pkl.gz"
    relax_matched_train._save(train, train_path)
    monkeypatch.setattr(relax_matched_train, "RAW_RESULTS_DIR", model_dir.parent)
    monkeypatch.setattr(relax_matched_train, "TRAIN_PATH", train_path)
    monkeypatch.setattr(relax_matched_train, "_get_ppd", object)
    monkeypatch.setattr(relax_matched_train, "_reduce", lambda structure: structure)

    calls: list[list[Structure]] = []

    def fake_relax(structures, **kwargs):
        calls.append(structures)
        infos = [
            {
                "energy_per_atom_eV_initial": float(index),
                "energy_per_atom_eV": float(index) + 0.5,
            }
            for index in range(len(structures))
        ]
        return structures, infos

    monkeypatch.setattr(relax_matched_train, "relax_structures", fake_relax)
    monkeypatch.setattr(
        relax_matched_train,
        "compute_ehulls",
        lambda structures, energies, ppd: list(energies),
    )
    args = argparse.Namespace(
        model="mattergen",
        device="cpu",
        fmax=0.01,
        max_steps=10,
        forward_batch_size=2,
        force=False,
    )

    relax_matched_train.process_model(args)

    with np.load(model_dir / "matched_train_indices.npz") as data:
        assert data["train_indices"].tolist() == [0, 2]
    assert calls == [[train[0], train[2]]]
    assert relax_matched_train._load(
        model_dir / "matched_train_ehull_unrelaxed.pkl.gz"
    ) == [0.0, 1.0]
    assert relax_matched_train._load(
        model_dir / "matched_train_ehull_relaxed.pkl.gz"
    ) == [0.5, 1.5]
    assert relax_matched_train._load(
        model_dir / "matched_train_relaxed_niggli.pkl.gz"
    ) == [train[0], train[2]]

    (model_dir / "matched_train_ehull_unrelaxed.pkl.gz").unlink()
    relax_matched_train.process_model(args)
    assert len(calls) == 1
    assert relax_matched_train._load(
        model_dir / "matched_train_ehull_unrelaxed.pkl.gz"
    ) == [0.0, 1.0]

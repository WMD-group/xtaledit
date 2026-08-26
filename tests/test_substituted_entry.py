from __future__ import annotations

import math
import pickle

from pymatgen.core import Lattice, Structure

from src.substituted_entry import SubstitutedEntry


def test_pickle_round_trip_uses_src_module() -> None:
    entry = SubstitutedEntry(
        gen_idx=1,
        train_idx=2,
        rank=3,
        cost_uniform=0.5,
        cost_mod_petti=1.5,
        structure=Structure(Lattice.cubic(3.0), ["Na"], [[0, 0, 0]]),
        cost_cs=0.25,
    )

    payload = pickle.dumps(entry)
    restored = pickle.loads(payload)

    assert b"src.substituted_entry" in payload
    assert type(restored) is SubstitutedEntry
    assert restored == entry


def test_unpickle_legacy_entry_defaults_cost_cs_to_nan() -> None:
    entry = SubstitutedEntry(
        gen_idx=1,
        train_idx=2,
        rank=3,
        cost_uniform=0.5,
        cost_mod_petti=1.5,
        structure=Structure(Lattice.cubic(3.0), ["Na"], [[0, 0, 0]]),
    )
    del entry.__dict__["cost_cs"]

    restored = pickle.loads(pickle.dumps(entry))

    assert math.isnan(restored.cost_cs)
    assert "cost_cs" in restored.__dict__

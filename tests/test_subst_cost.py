from __future__ import annotations

import pytest

from src._subst_cost import subst_cost_cs
from src.chemical_scale import CHEMICAL_SCALE


def test_subst_cost_cs_uses_table_2_values() -> None:
    assert len(CHEMICAL_SCALE) == 96
    assert CHEMICAL_SCALE["Fr"] == 0.0
    assert CHEMICAL_SCALE["F"] == 3.080
    assert subst_cost_cs("Na", "K") == pytest.approx(0.432)
    assert subst_cost_cs("Cl", "Cl") == 0.0


def test_subst_cost_cs_rejects_unsupported_elements() -> None:
    with pytest.raises(KeyError, match="Bk"):
        subst_cost_cs("Bk", "Na")

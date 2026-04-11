"""Element substitution cost functions shared across matching modules."""

from __future__ import annotations

from .mod_petti import MOD_PETTI


def subst_cost_uniform(e1: str, e2: str) -> float:
    """Return 0 if *e1* and *e2* are the same element, 1 otherwise."""
    return 0.0 if e1 == e2 else 1.0


def subst_cost_mod_petti(e1: str, e2: str) -> float:
    """Return the absolute difference of modified Pettifor numbers for *e1* and *e2*.

    Returns 102.0 if either element has no entry in the modified Pettifor table.
    """
    p1 = MOD_PETTI.get(e1)
    p2 = MOD_PETTI.get(e2)
    if p1 is None or p2 is None:
        return 102.0
    return float(abs(p1 - p2))

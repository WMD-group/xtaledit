"""Data model for substituted training structures."""

from dataclasses import dataclass

from pymatgen.core import Structure


@dataclass
class SubstitutedEntry:
    """Training structure with elements replaced by those of a matched gen structure.

    Attributes:
        gen_idx: Index of the generated structure in the preprocessed gen list.
        train_idx: Index of the training structure in the preprocessed train list.
        rank: 1-based rank of this training match among all matches for this gen
            structure (ranked by the cost used for selection).
        cost_uniform: Uniform substitution cost of the match (0 = identical chemistry).
        cost_mod_petti: Modified Pettifor substitution cost of the match.
        structure: Training structure with element types replaced by those of the
            generated structure according to the atom mapping.
    """

    gen_idx: int
    train_idx: int
    rank: int
    cost_uniform: float
    cost_mod_petti: float
    structure: Structure

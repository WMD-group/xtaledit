---
pretty_name: xtaledit
license: other
---

# xtaledit

Artifacts for **"Substitution-Based Analysis of Structural Novelty for Generative Models of Materials."**

This dataset supports the [`xtaledit`](https://github.com/masahiro-negishi/xtaledit) analysis of whether inorganic crystal generative models explore structures beyond elemental substitution in known MP20 training structures.

## Dataset contents

Each generated crystal is assigned one of three classes:

- **Duplicate:** directly matches an MP20 training structure with `pymatgen.analysis.structure_matcher.StructureMatcher`.
- **Substituted:** does not directly match, but is reproduced by substituting elements in a selected training structure and relaxing it once with MACE-MPA-0.
- **Unmatched:** is not reproduced by the substitution workflow.

The release contains generated structures and analysis artifacts for MatterGen, DiffCSP++, WyckoffTransformer, Crystalite, Chemeleon2, and an MP20 test-set reference. 
It also contains an 80,000-structure MatterGen collection used for selected symmetry analyses.

```text
input/
├── gen/
│   ├── raw/              Generated and MP20 test structures
│   └── preprocessed/     Relaxed, filtered, and Niggli-reduced structures
└── train/
    ├── raw/              MP20 training CSV
    └── preprocessed/     Reduced MP20 training structures

results/
├── raw/                  Matching, substitution, relaxation, and sensitivity data
└── analysis/
    ├── ai4am/            AI4AM figures and example CIF structures
    └── journal/          Manuscript figures, tables, and example structures
```

The files preserve the paths expected by the code repository and `notebooks/journal.ipynb`.

## Download and use

Clone the code repository, install its Python 3.12 environment, then download the artifacts into the repository root:

```bash
uvx hf download masahiro-negishi/xtaledit \
  --repo-type dataset \
  --include "input/**" \
  --include "results/**" \
  --local-dir .
```

Open `notebooks/journal.ipynb` to reproduce the manuscript figures and tables.
Rerunning preprocessing or relaxation requires CUDA-capable hardware for MACE-MPA-0.
Recomputing energies above the Materials Project convex hull also requires a Materials Project API key.

## Excluded data

ICSD records are not redistributed under the standard ICSD license. 
Therefore, the release excludes:

- `input/icsd/**`
- `results/raw/icsd/**`

The final ICSD-dependent prototype-coverage figure is included as a derived manuscript artifact. 
Regenerating it requires licensed ICSD access and a locally prepared `results/raw/icsd/wyckoff_repr_s=0.01.pkl.gz`.

The release also excludes `input/ppd_cache.pkl.gz`, a Materials Project-derived phase-diagram cache whose redistribution terms have not been confirmed, and the recomputable `notebooks/.cache/` directory.

## Licensing and attribution

The repository uses `license: other` because it combines generated structures, MP20-derived inputs, and derived scientific artifacts. Upstream data retain their original licenses, terms, and citation requirements.
Users are responsible for checking the applicable terms before redistribution or commercial use.
See the paper for the complete model and method citations.

## Citation

Citation information for the xtaledit paper will be added when a preprint or publication is available.

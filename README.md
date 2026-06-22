# Substitution-Based Analysis of Structural Novelty for Generative Models of Materials

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Formatter: Ruff](https://img.shields.io/badge/formatter-Ruff-D7FF64.svg?logo=ruff)](https://docs.astral.sh/ruff/)
[![Linter: Ruff](https://img.shields.io/badge/linter-Ruff-D7FF64.svg?logo=ruff)](https://docs.astral.sh/ruff/)
[![Packaging: uv](https://img.shields.io/badge/packaging-uv-DE5FE9.svg?logo=uv)](https://docs.astral.sh/uv/)
[![GitHub issues](https://img.shields.io/github/issues-raw/masahiro-negishi/xtaledit)](https://github.com/masahiro-negishi/xtaledit/issues)

Code and analysis for the paper **"Substitution-Based Analysis of Structural Novelty for Generative Models of Materials."**

`xtaledit` evaluates whether crystals from generative models go beyond elemental substitution in known MP20 training structures.

## Novelty classification

Each generated crystal is assigned to one of three classes:

| Class | Meaning |
|---|---|
| **Duplicate** | Matches an MP20 training structure with `pymatgen`'s `StructureMatcher`. |
| **Substituted** | Is reproduced by substituting elements in a selected training structure and relaxing it. |
| **Unmatched** | Is not reproduced by the substitution-based workflow. |

## Workflow

For each generated crystal, the pipeline:

1. Relaxes the generated structure with MACE-MPA-0, applies primitive-cell and Niggli reduction, checks composition validity with SMACT, and computes energy above the Materials Project convex hull.
2. Searches the MP20 training set for direct `StructureMatcher` matches.
3. For non-Duplicates, selects structurally similar training crystals using two complementary criteria:
   - anonymous lattice-and-site matching with `StructureMatcher`;
   - matching space groups and Wyckoff-label multisets.
4. Ranks candidates by mean modified Pettifor-scale distance between mapped elements and retains the top `k=3` from each structural criterion.
5. Substitutes the candidate elements, relaxes each candidate once with MACE-MPA-0, and compares the relaxed structures with the generated crystal.

The seven pipeline scripts under `scripts/` (excluding `upload_hf_dataset.py`) implement preprocessing, direct matching, the two candidate-selection methods, substitution, relaxation, and the final match check.
The ordered YAML files in `configs/crystalite/` provide a complete Crystalite example.

## Repository layout

```text
configs/                  Pipeline configurations
scripts/                  Executable preprocessing and matching stages
src/                      Matching, substitution, energy, and utility code
notebooks/journal.ipynb   All manuscript figures and analysis tables

input/
├── gen/
│   ├── raw/              Generated structures
│   └── preprocessed/     Relaxed, filtered, and reduced structures
├── train/
│   ├── raw/              MP20 training data
│   └── preprocessed/     Reduced training structures
└── icsd/                 Licensed ICSD-derived inputs; not publicly distributed

results/
├── raw/                  Pipeline outputs and intermediate artifacts
└── analysis/journal/     Manuscript figures, tables, and example structures
```

`input/`, `results/`, and notebook caches are intentionally excluded from the Git repository.

## Setup

Python 3.12 is required.

```bash
uv venv
source .venv/bin/activate
uv sync --group dev
```

The lock file selects CUDA 12.8 PyTorch wheels. The configured pipeline is intended for CUDA-capable hardware because preprocessing and substituted structure relaxation use MACE-MPA-0.
The first MACE run may download model weights.

Preprocessing also needs a Materials Project API key to construct the phase diagram cache:

```bash
export MP_API_KEY="<your Materials Project API key>"
```

Project paths default to `input/` and `results/`. 
They can be redirected in a gitignored `.env` file:

```dotenv
INPUT_DIR=/path/to/input
RESULTS_DIR=/path/to/results
```

## Reproduce the manuscript figures

The public paper artifacts are hosted in the Hugging Face dataset repository [`masahiro-negishi/xtaledit`](https://huggingface.co/datasets/masahiro-negishi/xtaledit).
The dataset card is maintained in [`DATASET_CARD.md`](DATASET_CARD.md).
Download them into the repository root while preserving their paths:

```bash
uvx hf download masahiro-negishi/xtaledit \
  --repo-type dataset \
  --include "input/**" \
  --include "results/**" \
  --local-dir .
```

Then open the journal notebook:

```bash
jupyter notebook notebooks/journal.ipynb
```

All figures used in the manuscript are produced by [`notebooks/journal.ipynb`](notebooks/journal.ipynb).
Outputs are written to `results/analysis/journal/`.
The notebook also recreates the main classification table and supporting analysis tables.

The public artifacts support all notebook sections except the final ICSD-based Wyckoff-multiset coverage analysis. 
Run all preceding cells normally. 
Running the final section requires a locally prepared, licensed `results/raw/icsd/wyckoff_repr_s=0.01.pkl.gz` artifact.
The released `results/analysis/journal/` directory includes the final derived figure for reference.

Apart from the ICSD exception above, the downloaded artifacts support figure reproduction for MatterGen, DiffCSP++, WyckoffTransformer, Crystalite, Chemeleon2, and the MP20 test set. 
The checked-in end-to-end pipeline configurations currently cover Crystalite; configurations for rerunning every model are not included.

## Run the Crystalite pipeline

Place the raw Crystalite structures at `input/gen/raw/crystalite.pkl.gz` and the MP20 training CSV at `input/train/raw/train.csv`.
Run the stages in order:

```bash
python scripts/preprocess.py configs/crystalite/01_preprocess.yaml
python scripts/exact_match.py configs/crystalite/02_exact_match.yaml
python scripts/anonymous_match.py configs/crystalite/03_anonymous_match.yaml
python scripts/wyckoff_match.py configs/crystalite/04_wyckoff_match.yaml
python scripts/substitute_structures.py configs/crystalite/05_substitute_structures.yaml
python scripts/relax_substituted.py configs/crystalite/06_relax_substituted.yaml
python scripts/check_relaxed_substituted_matches.py configs/crystalite/07_check_relaxed_matches.yaml
```

Outputs are written to `input/gen/preprocessed/crystalite/` and `results/raw/crystalite/`.
Existing artifacts are skipped unless the relevant configuration supports and enables `force: true`.

## Data sharing and licensing

The Hugging Face release contains:

- redistributable generated structures and MP20 train/test inputs;
- preprocessed structures and filtering metadata;
- raw matching, substitution, relaxation, and sensitivity-analysis artifacts;
- figures, tables, and example structures from `results/analysis/ai4am/` and
  `results/analysis/journal/`.

ICSD records must not be redistributed under the standard ICSD license.
Therefore, `input/icsd/` and record-level `results/raw/icsd/` artifacts are excluded from the public dataset.
The final ICSD-dependent prototype-coverage figure can be shared as a derived manuscript artifact, but regenerating it requires licensed ICSD access and locally prepared Wyckoff representations.

The Materials Project-derived phase-diagram cache should also be excluded unless its redistribution terms have been confirmed.
Without that cache, rerunning preprocessing requires `MP_API_KEY`.

Validate the public manifest without uploading:

```bash
python scripts/upload_hf_dataset.py
```

Publish the artifacts and dataset card after authenticating with Hugging Face:

```bash
uvx hf auth login
python scripts/upload_hf_dataset.py --upload
```

The uploader includes `input/gen/`, `input/train/`, `results/raw/`, `results/analysis/ai4am/`, and `results/analysis/journal/`, while explicitly excluding other analysis directories, ICSD data, the Materials Project-derived phase-diagram cache, and `.gitkeep` files.
Uploads are resumable.
To update only the dataset card, pass `--card-only --upload`.

The MIT license badge applies to the code repository. 
The Hugging Face dataset uses `license: other` because it combines generated structures, MP20-derived
inputs, and derived artifacts.
Upstream data retain their original terms and citation requirements; see [`DATASET_CARD.md`](DATASET_CARD.md).

Generate a version-specific DOI for a stable release and add it to this README, the dataset card, and the paper's Data Availability statement.

## Citation

Citation information will be added when the paper preprint or publication is available.

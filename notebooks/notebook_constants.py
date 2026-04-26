"""Shared constants for analysis notebooks."""

CATEGORY_ORDER = ["1", "2-1", "2-2", "2-3", "3"]
CATEGORY_LABELS = {
    "1": "Duplicate",
    "2-1": "Substituted (both)",
    "2-2": "Substituted (SM-anon only)",
    "2-3": "Substituted (Wyckoff only)",
    "3": "Unmatched",
}
SIMPLE_CATEGORY_ORDER = ["1", "2", "3"]
SIMPLE_CATEGORY_LABELS = {
    "1": "Duplicate",
    "2": "Substituted",
    "3": "Unmatched",
}
CRYSTAL_SYSTEM_ORDER = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]

SUBSET_OPTIONS = {"all", "metastable", "metastable_smact_valid"}
METASTABLE_EHULL_MAX = 0.1

GEN_FILE = "relaxed_niggli.pkl.gz"
TRAIN_FILE = "train.pkl.gz"
EHULL_RELAXED_FILE = "ehull_relaxed.pkl.gz"
EHULL_RELAXED_INFO_FILE = "relax_infos.pkl.gz"
SMACT_VALIDITY_FILE = "smact_validity.npz"

SM_FIT_FILE = "sm_fit.npz"
SM_ANON_MATCH_FILE = "sm_anon.pkl.gz"
WYCKOFF_MATCH_FILE = "wyckoff_match_s=0.01_c=mod_petti.pkl.gz"
WYCKOFF_REPR_FILE = "wyckoff_repr_s=0.01.pkl.gz"

TOP3_SM_ANON_FILE = "substituted_top3_sm_anon.pkl.gz"
TOP3_WYCKOFF_FILE = "substituted_top3_wyckoff_match_s=0.01_c=mod_petti.pkl.gz"
RELAXED_SM_ANON_ENTRIES_FILE = "substituted_relaxed_niggli_top3_sm_anon.pkl.gz"
RELAXED_WYCKOFF_ENTRIES_FILE = (
    "substituted_relaxed_niggli_top3_wyckoff_match_s=0.01_c=mod_petti.pkl.gz"
)
RELAXED_SM_ANON_MATCHES_FILE = (
    "substituted_relaxed_niggli_top3_sm_anon_gen_matches.pkl.gz"
)
RELAXED_WYCKOFF_MATCHES_FILE = (
    "substituted_relaxed_niggli_top3_wyckoff_match_s=0.01_c="
    "mod_petti_gen_matches.pkl.gz"
)

SOURCE_ORDER = ["anon", "wyckoff"]
MATCH_SETTING_ORDER = ["match_anon", "match_wyckoff", "match_concat"]

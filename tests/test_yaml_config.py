from pathlib import Path

import pytest

from src.config import ROOT
from src.yaml_config import (
    get_bool,
    get_float,
    get_int,
    get_mapping,
    get_path,
    get_paths,
    load_yaml_config,
)


def test_load_yaml_config_preserves_native_types(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "enabled: true\ncount: 3\ntolerance: 0.2\noptions:\n  scale: false\n"
    )

    config = load_yaml_config(
        path,
        allowed_keys={"enabled", "count", "tolerance", "options"},
        required_keys={"enabled"},
    )

    assert get_bool(config, "enabled") is True
    assert get_int(config, "count") == 3
    assert get_float(config, "tolerance") == 0.2
    assert get_mapping(config, "options") == {"scale": False}


def test_paths_resolve_relative_to_repository_root(tmp_path: Path) -> None:
    config = {
        "path": "results/raw/model/output.pkl.gz",
        "paths": ["input/a.pkl.gz", str(tmp_path / "absolute.pkl.gz")],
    }

    assert get_path(config, "path") == ROOT / "results/raw/model/output.pkl.gz"
    assert get_paths(config, "paths") == [
        ROOT / "input/a.pkl.gz",
        tmp_path / "absolute.pkl.gz",
    ]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("[one, two]\n", "top-level mapping"),
        ("known: true\nunknown: false\n", "unknown config key"),
        ("known: true\n", "missing required config key"),
        ("known: [\n", "invalid YAML"),
    ],
)
def test_load_yaml_config_rejects_invalid_documents(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents)

    with pytest.raises(SystemExit, match=message):
        load_yaml_config(
            path,
            allowed_keys={"known", "required"},
            required_keys={"required"},
        )


@pytest.mark.parametrize(
    ("getter", "config"),
    [
        (get_bool, {"value": "true"}),
        (get_int, {"value": True}),
        (get_float, {"value": "0.2"}),
        (get_mapping, {"value": []}),
        (get_paths, {"value": "input/file.pkl.gz"}),
    ],
)
def test_getters_reject_wrong_types(getter, config: dict[str, object]) -> None:
    with pytest.raises(SystemExit, match="config key 'value'"):
        getter(config, "value")

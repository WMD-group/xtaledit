"""Helpers for loading and validating script YAML configurations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from src.config import ROOT

_MISSING = object()


def load_yaml_config(
    path: Path,
    *,
    allowed_keys: Iterable[str],
    required_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Load a YAML mapping and validate its top-level keys."""
    try:
        with path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except FileNotFoundError:
        raise SystemExit(f"error: config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise SystemExit(f"error: invalid YAML in {path}: {exc}") from None

    if not isinstance(loaded, dict):
        raise SystemExit(f"error: config must contain a top-level mapping: {path}")
    if not all(isinstance(key, str) for key in loaded):
        raise SystemExit(f"error: config keys must be strings: {path}")

    allowed = set(allowed_keys)
    unknown = sorted(set(loaded) - allowed)
    if unknown:
        raise SystemExit(
            f"error: unknown config key(s) in {path}: {', '.join(unknown)}"
        )

    missing = sorted(set(required_keys) - set(loaded))
    if missing:
        raise SystemExit(
            f"error: missing required config key(s) in {path}: {', '.join(missing)}"
        )
    return loaded


def _get(config: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    if key in config:
        return config[key]
    if default is _MISSING:
        raise SystemExit(f"error: missing required config key: {key}")
    return default


def get_string(
    config: Mapping[str, Any],
    key: str,
    *,
    default: str | None | object = _MISSING,
) -> str | None:
    """Return a string config value."""
    value = _get(config, key, default)
    if value is None and default is None:
        return None
    if not isinstance(value, str):
        raise SystemExit(f"error: config key '{key}' must be a string")
    return value


def get_bool(
    config: Mapping[str, Any],
    key: str,
    *,
    default: bool | object = _MISSING,
) -> bool:
    """Return a boolean config value."""
    value = _get(config, key, default)
    if not isinstance(value, bool):
        raise SystemExit(f"error: config key '{key}' must be a boolean")
    return value


def get_int(
    config: Mapping[str, Any],
    key: str,
    *,
    default: int | object = _MISSING,
) -> int:
    """Return an integer config value."""
    value = _get(config, key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"error: config key '{key}' must be an integer")
    return value


def get_float(
    config: Mapping[str, Any],
    key: str,
    *,
    default: float | object = _MISSING,
) -> float:
    """Return a numeric config value as a float."""
    value = _get(config, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"error: config key '{key}' must be a number")
    return float(value)


def resolve_path(value: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_path(
    config: Mapping[str, Any],
    key: str,
    *,
    default: str | None | object = _MISSING,
) -> Path | None:
    """Return a repository-root-relative or absolute path."""
    value = get_string(config, key, default=default)
    return None if value is None else resolve_path(value)


def get_paths(config: Mapping[str, Any], key: str) -> list[Path]:
    """Return a non-empty list of paths."""
    value = _get(config, key)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"error: config key '{key}' must be a non-empty list")
    if not all(isinstance(item, str) for item in value):
        raise SystemExit(f"error: config key '{key}' must contain only strings")
    return [resolve_path(item) for item in value]


def get_mapping(
    config: Mapping[str, Any],
    key: str,
    *,
    default: Mapping[str, Any] | object = _MISSING,
) -> dict[str, Any]:
    """Return a mapping with string keys."""
    value = _get(config, key, default)
    if not isinstance(value, dict) or not all(
        isinstance(item_key, str) for item_key in value
    ):
        raise SystemExit(
            f"error: config key '{key}' must be a mapping with string keys"
        )
    return dict(value)

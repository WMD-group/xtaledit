"""Validate and upload the public xtaledit artifacts to Hugging Face.

The command is a dry run unless ``--upload`` is passed. Public artifacts keep
their repository-relative ``input/`` and ``results/`` paths.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "masahiro-negishi/xtaledit"
DATASET_CARD = ROOT / "DATASET_CARD.md"

INCLUDED_DIRS = (
    Path("input/gen"),
    Path("input/train"),
    Path("results/raw"),
    Path("results/analysis/ai4am"),
    Path("results/analysis/journal"),
)
EXCLUDED_PATHS = (
    Path("input/icsd"),
    Path("input/ppd_cache.pkl.gz"),
    Path("results/raw/icsd"),
)
REQUIRED_PATHS = (
    Path("input/gen/raw/crystalite.pkl.gz"),
    Path("input/train/raw/train.csv"),
)

HF_INCLUDE_PATTERNS = (
    "input/gen/**",
    "input/train/**",
    "results/raw/**",
    "results/analysis/ai4am/**",
    "results/analysis/journal/**",
)
HF_EXCLUDE_PATTERNS = (
    "**/.gitkeep",
    "input/icsd/**",
    "input/ppd_cache.pkl.gz",
    "results/raw/icsd/**",
)


@dataclass(frozen=True)
class Manifest:
    """Files selected for public release."""

    files: tuple[Path, ...]
    total_bytes: int


def _is_excluded(relative_path: Path) -> bool:
    if relative_path.name == ".gitkeep":
        return True
    return any(
        relative_path == excluded or excluded in relative_path.parents
        for excluded in EXCLUDED_PATHS
    )


def build_manifest(root: Path = ROOT) -> Manifest:
    """Build and validate the public artifact manifest."""
    missing = [path for path in REQUIRED_PATHS if not (root / path).is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"error: required release files are missing:\n{formatted}")

    files: list[Path] = []
    for directory in INCLUDED_DIRS:
        absolute_dir = root / directory
        if not absolute_dir.is_dir():
            raise SystemExit(f"error: release directory not found: {directory}")
        files.extend(
            path.relative_to(root)
            for path in absolute_dir.rglob("*")
            if path.is_file() and not _is_excluded(path.relative_to(root))
        )

    unique_files = tuple(sorted(set(files)))
    total_bytes = sum((root / path).stat().st_size for path in unique_files)
    return Manifest(files=unique_files, total_bytes=total_bytes)


def format_size(size: int) -> str:
    """Format a byte count with a binary unit."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.3f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def large_folder_command(
    repo_id: str,
    num_workers: int | None = None,
    root: Path = ROOT,
) -> list[str]:
    """Return the resilient artifact-upload command."""
    command = [
        "uvx",
        "--from",
        "huggingface_hub",
        "hf",
        "upload-large-folder",
        repo_id,
        str(root),
        "--repo-type",
        "dataset",
    ]
    for pattern in HF_INCLUDE_PATTERNS:
        command.extend(["--include", pattern])
    for pattern in HF_EXCLUDE_PATTERNS:
        command.extend(["--exclude", pattern])
    if num_workers is not None:
        command.extend(["--num-workers", str(num_workers)])
    return command


def card_upload_command(repo_id: str, card: Path = DATASET_CARD) -> list[str]:
    """Return the command that publishes the dataset card as README.md."""
    return [
        "uvx",
        "--from",
        "huggingface_hub",
        "hf",
        "upload",
        repo_id,
        str(card),
        "README.md",
        "--repo-type",
        "dataset",
        "--commit-message",
        "Update dataset card",
    ]


def run_commands(commands: Sequence[Sequence[str]]) -> None:
    """Run upload commands with high-performance Xet enabled by default."""
    env = os.environ.copy()
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Perform the upload. Without this flag, only validate and report.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Destination dataset repository (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker count passed to hf upload-large-folder.",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload only DATASET_CARD.md as the dataset README.",
    )
    args = parser.parse_args()
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if not DATASET_CARD.is_file():
        raise SystemExit(f"error: dataset card not found: {DATASET_CARD}")

    commands: list[list[str]] = []
    if args.card_only:
        print(f"Dataset card: {DATASET_CARD.relative_to(ROOT)}")
    else:
        manifest = build_manifest()
        print(
            f"Release manifest: {len(manifest.files)} files, "
            f"{format_size(manifest.total_bytes)}"
        )
        commands.append(
            large_folder_command(
                repo_id=args.repo_id,
                num_workers=args.num_workers,
            )
        )

    commands.append(card_upload_command(args.repo_id))
    if not args.upload:
        print("Commands:")
        for command in commands:
            print(f"  {shlex.join(command)}")
        print("Dry run complete. Pass --upload to publish these files.")
        return

    run_commands(commands)
    print(f"Published dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

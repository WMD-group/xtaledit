from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

UPLOAD_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "upload_hf_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("upload_hf_dataset", UPLOAD_PATH)
assert SPEC is not None and SPEC.loader is not None
upload = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = upload
SPEC.loader.exec_module(upload)


def _create_release_tree(root: Path) -> None:
    paths = {
        "input/gen/raw/crystalite.pkl.gz": b"crystalite",
        "input/gen/raw/model.pkl.gz": b"model",
        "input/gen/raw/.gitkeep": b"",
        "input/train/raw/train.csv": b"cif\nexample",
        "input/icsd/raw/licensed.pkl.gz": b"licensed",
        "input/ppd_cache.pkl.gz": b"cache",
        "results/raw/model/result.pkl.gz": b"result",
        "results/raw/icsd/wyckoff_repr_s=0.01.pkl.gz": b"licensed",
        "results/analysis/ai4am/poster.pdf": b"poster",
        "results/analysis/journal/figure.pdf": b"figure",
        "results/analysis/embedding_projections/figure.pdf": b"excluded",
        "results/analysis/mattergen/structure.cif": b"excluded",
        "results/analysis/sm_sensitivity/figure.pdf": b"excluded",
    }
    for relative, content in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_build_manifest_includes_public_files_and_excludes_restricted(
    tmp_path: Path,
) -> None:
    _create_release_tree(tmp_path)

    manifest = upload.build_manifest(tmp_path)

    assert manifest.files == (
        Path("input/gen/raw/crystalite.pkl.gz"),
        Path("input/gen/raw/model.pkl.gz"),
        Path("input/train/raw/train.csv"),
        Path("results/analysis/ai4am/poster.pdf"),
        Path("results/analysis/journal/figure.pdf"),
        Path("results/raw/model/result.pkl.gz"),
    )
    assert manifest.total_bytes == sum(
        (tmp_path / path).stat().st_size for path in manifest.files
    )


def test_build_manifest_requires_crystalite_raw_input(tmp_path: Path) -> None:
    _create_release_tree(tmp_path)
    (tmp_path / "input/gen/raw/crystalite.pkl.gz").unlink()

    with pytest.raises(SystemExit, match="crystalite.pkl.gz"):
        upload.build_manifest(tmp_path)


def test_format_size_uses_binary_units() -> None:
    assert upload.format_size(1536) == "1.500 KiB"
    assert upload.format_size(2 * 1024**3) == "2.000 GiB"


def test_large_folder_command_contains_release_filters(tmp_path: Path) -> None:
    command = upload.large_folder_command(
        "owner/dataset",
        num_workers=4,
        root=tmp_path,
    )

    assert command[:7] == [
        "uvx",
        "--from",
        "huggingface_hub",
        "hf",
        "upload-large-folder",
        "owner/dataset",
        str(tmp_path),
    ]
    assert ["--num-workers", "4"] == command[-2:]
    for pattern in upload.HF_INCLUDE_PATTERNS:
        index = command.index(pattern)
        assert command[index - 1] == "--include"
    for pattern in upload.HF_EXCLUDE_PATTERNS:
        index = command.index(pattern)
        assert command[index - 1] == "--exclude"


def test_card_upload_targets_remote_readme() -> None:
    command = upload.card_upload_command("owner/dataset", Path("CARD.md"))

    assert command[4:9] == [
        "upload",
        "owner/dataset",
        "CARD.md",
        "README.md",
        "--repo-type",
    ]
    assert command[9] == "dataset"


def test_run_commands_enables_xet_and_checks_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.setattr(upload.subprocess, "run", fake_run)

    upload.run_commands([["first"], ["second"]])

    assert [call[0] for call in calls] == [["first"], ["second"]]
    assert all(call[1]["check"] is True for call in calls)
    assert all(call[1]["env"]["HF_XET_HIGH_PERFORMANCE"] == "1" for call in calls)


def test_main_dry_run_prints_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = upload.Manifest(files=(Path("file"),), total_bytes=4)
    monkeypatch.setattr(upload, "build_manifest", lambda: manifest)
    monkeypatch.setattr(
        upload,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "upload": False,
                "repo_id": "owner/dataset",
                "num_workers": None,
                "card_only": False,
            },
        )(),
    )

    upload.main()

    output = capsys.readouterr().out
    assert "hf upload-large-folder owner/dataset" in output
    assert "hf upload owner/dataset" in output
    assert "results/raw/icsd/**" in output

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "package_week4_submission.py"


def load_module():
    spec = importlib.util.spec_from_file_location("package_week4_submission", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builds_deterministic_week4_only_archive_and_manifest(tmp_path: Path):
    module = load_module()
    source = tmp_path / "Week4"
    (source / "Day17").mkdir(parents=True)
    (source / "README.md").write_text("# Week 4\n", encoding="utf-8")
    (source / "Day17" / "result.json").write_text('{"valid": true}\n', encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = module.package_week4(source, first)
    second_result = module.package_week4(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["archive_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_result["file_count"] == 3
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "Week4/Day17/result.json",
            "Week4/README.md",
            "Week4/SHA256SUMS.txt",
        ]
        assert not any("__MACOSX" in name or name.startswith(("/", "../")) for name in archive.namelist())


@pytest.mark.parametrize("name", ["model.safetensors", "shutdown_evidence.json", "private_mapping.json"])
def test_rejects_teacher_forbidden_artifacts(tmp_path: Path, name: str):
    module = load_module()
    source = tmp_path / "Week4"
    source.mkdir()
    (source / name).write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        module.package_week4(source, tmp_path / "submission.zip")


def test_rejects_symlink_and_repository_only_markdown_path(tmp_path: Path):
    module = load_module()
    source = tmp_path / "Week4"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    os.symlink(outside, source / "linked.txt")
    with pytest.raises(ValueError, match="symlink"):
        module.package_week4(source, tmp_path / "symlink.zip")

    (source / "linked.txt").unlink()
    (source / "README.md").write_text("See ../../deliverables/week4/day21\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repository-only"):
        module.package_week4(source, tmp_path / "path.zip")

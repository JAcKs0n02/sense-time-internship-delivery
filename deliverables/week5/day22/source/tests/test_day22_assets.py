from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


SOURCE_ROOT = Path(__file__).parents[1]
BUILD_SCRIPT = SOURCE_ROOT / "scripts" / "build_image_manifest.py"
VALIDATE_SCRIPT = SOURCE_ROOT / "scripts" / "validate_day22.py"
REMOTE_EVIDENCE = SOURCE_ROOT / "results" / "remote_evidence"
IMAGE_TYPES = (
    "table_screenshot",
    "natural_scene",
    "logo",
    "handwritten_formula",
    "ui_interface",
)


def write_fixture_bundle(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir()
    sources: list[dict[str, str]] = []
    for index, image_type in enumerate(IMAGE_TYPES, start=1):
        filename = f"fixture-{index}.png"
        Image.new("RGB", (index + 2, index + 3), color=(10 * index, 20, 30)).save(
            images / filename
        )
        sources.append(
            {
                "image_id": f"w5-fixture-{index:02d}",
                "filename": filename,
                "image_type": image_type,
                "source": f"https://example.invalid/source/{index}",
                "license": "CC0-1.0",
            }
        )
    source_path = root / "image_sources.json"
    source_path.write_text(json.dumps(sources), encoding="utf-8")
    return images, source_path


def test_build_manifest_cli_freezes_exact_five_images(tmp_path: Path):
    images, sources = write_fixture_bundle(tmp_path)
    output = tmp_path / "image_manifest.csv"

    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--images-dir",
            str(images),
            "--sources",
            str(sources),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {row["image_type"] for row in rows} == set(IMAGE_TYPES)
    assert len({row["image_id"] for row in rows}) == 5
    for index, row in enumerate(rows, start=1):
        path = images / row["relative_path"]
        assert int(row["width"]) == index + 2
        assert int(row["height"]) == index + 3
        assert int(row["file_bytes"]) == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def run_builder(images: Path, sources: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--images-dir",
            str(images),
            "--sources",
            str(sources),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def write_ground_truth(source_path: Path, output: Path) -> None:
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "1.0",
        "images": [
            {
                "image_id": row["image_id"],
                "summary": f"Verified fixture for {row['image_type']}",
                "visible_text": [],
                "key_facts": [f"category={row['image_type']}"],
                "unknown_or_not_inferable": ["fixture creation context"],
                "privacy_review": "PASS",
                "license_review": "PASS",
            }
            for row in sources
        ],
    }
    output.write_text(json.dumps(payload), encoding="utf-8")


def test_builder_rejects_paths_outside_the_image_bundle(tmp_path: Path):
    images, sources = write_fixture_bundle(tmp_path)
    payload = json.loads(sources.read_text(encoding="utf-8"))
    payload[0]["filename"] = "../outside.png"
    sources.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "image_manifest.csv"
    output.write_text("stale ready output", encoding="utf-8")

    completed = run_builder(images, sources, output)

    assert completed.returncode == 1
    assert "direct child" in completed.stderr
    assert not output.exists()


def test_validator_recomputes_manifest_and_ground_truth_gates(tmp_path: Path):
    images, sources = write_fixture_bundle(tmp_path)
    manifest = tmp_path / "image_manifest.csv"
    assert run_builder(images, sources, manifest).returncode == 0
    ground_truth = tmp_path / "image_ground_truth.json"
    write_ground_truth(sources, ground_truth)
    output = tmp_path / "day22_validation.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--images-dir",
            str(images),
            "--manifest",
            str(manifest),
            "--ground-truth",
            str(ground_truth),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["summary"] == {
        "image_count": 5,
        "unique_image_ids": 5,
        "image_types": sorted(IMAGE_TYPES),
        "ground_truth_count": 5,
    }
    assert all(check["status"] == "PASS" for check in result["checks"])


def test_validator_fails_closed_when_an_image_changes(tmp_path: Path):
    images, sources = write_fixture_bundle(tmp_path)
    manifest = tmp_path / "image_manifest.csv"
    assert run_builder(images, sources, manifest).returncode == 0
    ground_truth = tmp_path / "image_ground_truth.json"
    write_ground_truth(sources, ground_truth)
    output = tmp_path / "day22_validation.json"
    output.write_text('{"status":"PASS"}', encoding="utf-8")
    changed_image = images / "fixture-1.png"
    changed_image.write_bytes(changed_image.read_bytes() + b"changed")

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--images-dir",
            str(images),
            "--manifest",
            str(manifest),
            "--ground-truth",
            str(ground_truth),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "FAIL"
    assert "file_bytes mismatch" in " ".join(result["errors"])


def load_validator_module():
    sys.path.insert(0, str(VALIDATE_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("validate_day22", VALIDATE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_evidence_validator_accepts_the_frozen_autodl_run():
    module = load_validator_module()

    errors, summary = module.validate_remote_evidence(REMOTE_EVIDENCE)

    assert errors == []
    assert summary["repository"] == "Qwen/Qwen2-VL-7B-Instruct"
    assert summary["revision"] == "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    assert summary["weight_shards"] == 5
    assert summary["offline_smoke_status"] == "PASS"


def test_remote_evidence_validator_fails_closed_on_a_changed_smoke(tmp_path: Path):
    module = load_validator_module()
    copied = tmp_path / "remote_evidence"
    shutil.copytree(REMOTE_EVIDENCE, copied)
    smoke_path = copied / "offline_load_smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["local_files_only"] = False
    smoke["image"]["sha256"] = "0" * 64
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")

    errors, _ = module.validate_remote_evidence(copied)

    assert any("local_files_only" in error for error in errors)
    assert any("image sha256" in error for error in errors)

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


REPO = Path(__file__).parents[5]
SUBMISSION_ROOT = REPO / "Submission"
WEEK4_SUBMISSION = REPO / "Submission" / "Week4"
SUBMISSION = WEEK4_SUBMISSION / "Day21_Weekly_Report_and_Final_Model_Archive"


def test_day21_teacher_submission_has_exact_required_artifacts():
    required = {
        "README.md",
        "Week4_DPO_Preference_Alignment_Report.md",
        "DPO_Rewards_Curves.png",
        "DPO_Training_Metrics.csv",
        "Week4_Acceptance_Matrix.md",
        "Final_DPO_Model_Archive.json",
    }
    assert {path.name for path in SUBMISSION.iterdir() if path.is_file()} == required
    archive = json.loads((SUBMISSION / "Final_DPO_Model_Archive.json").read_text())
    assert archive["status"] == "ready"
    assert archive["selected_model"] == "reward_corrective_40step_merged"
    assert archive["evaluation"]["teacher_refusal_rate"] == 1.0
    assert archive["remediation_candidate_promoted"] is True
    assert archive["model_weights_in_submission"] is False
    with (SUBMISSION / "DPO_Training_Metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert [int(row["step"]) for row in rows] == list(range(1, 41))
    with Image.open(SUBMISSION / "DPO_Rewards_Curves.png") as image:
        assert image.format == "PNG"
        assert image.width >= 1800 and image.height >= 1200


def test_teacher_report_states_all_results_without_overclaiming():
    report = (SUBMISSION / "Week4_DPO_Preference_Alignment_Report.md").read_text(encoding="utf-8")
    assert "710" in report and "870" in report and "783" in report and "87" in report
    assert "40/40" in report
    assert "10/10 = 100%" in report
    assert "4/10 = 40%" in report
    assert "3.595" in report and "3.640" in report
    assert "不据此声称统计显著" in report
    assert "reward_corrective_40step_merged" in report
    assert "DPO_Rewards_Curves.png" in report
    assert "source/results/dpo_rewards_curves.png" not in report


def test_week4_teacher_directory_excludes_weights_private_mapping_and_shutdown_evidence():
    files = [path for path in WEEK4_SUBMISSION.rglob("*") if path.is_file()]
    assert not any("mapping" in path.name.lower() or "shutdown" in path.name.lower() for path in files)
    assert not any(path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"} for path in files)
    assert not any(path.stat().st_size > 50 * 1024 * 1024 for path in files)
    assert not any(path.is_symlink() for path in WEEK4_SUBMISSION.rglob("*"))


def test_all_week4_teacher_markdown_links_are_self_contained_and_resolve():
    markdown_link = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
    for markdown in WEEK4_SUBMISSION.rglob("*.md"):
        content = markdown.read_text(encoding="utf-8")
        assert "deliverables/week4" not in content, f"repository-only path in teacher file: {markdown}"
        for target in markdown_link.findall(content):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            resolved = (markdown.parent / target).resolve()
            assert resolved.is_relative_to(WEEK4_SUBMISSION.resolve()), f"link escapes Week4: {markdown}: {target}"
            assert resolved.exists(), f"broken link: {markdown}: {target}"


def test_week4_has_self_contained_sha256_manifest():
    manifest_path = WEEK4_SUBMISSION / "SHA256SUMS.txt"
    rows = [line.split("  ", 1) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    recorded = {relative: digest for digest, relative in rows}
    expected_files = {
        path.relative_to(WEEK4_SUBMISSION).as_posix(): path
        for path in WEEK4_SUBMISSION.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert set(recorded) == set(expected_files)
    for relative, path in expected_files.items():
        assert recorded[relative] == hashlib.sha256(path.read_bytes()).hexdigest(), relative

#!/usr/bin/env python3
"""Validate Week 3 evidence and build the exact Week 4 model handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


TERMINAL_STATUSES = {
    "completed",
    "failed",
    "invalid",
    "invalid_nonfinite_loss",
    "interrupted",
}
EXPECTED_GROUPS = {"rank", "learning_rate", "epoch"}
EXPECTED_OPENCOMPASS_PAIRS = {
    ("base", "ceval"),
    ("base", "cmmlu"),
    ("best_sft", "ceval"),
    ("best_sft", "cmmlu"),
}
EXPECTED_OPENCOMPASS_SUBJECTS = {"ceval": 52, "cmmlu": 67}
EXPECTED_WEIGHTS = {
    "accuracy": 0.30,
    "completeness": 0.25,
    "logic": 0.20,
    "safety": 0.15,
    "format": 0.10,
}
BASE_MODEL_PATH = "/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct"
MERGED_MODEL_PATH = "/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged"
REQUIRED_WEEK3_SUBMISSION_DIRECTORIES = {
    "Week3",
    "Week3/Day11_Experiment_Design",
    "Week3/Day12_13_Batch_Experiments",
    "Week3/Day12_13_Batch_Experiments/Raw_Logs",
    "Week3/Day12_13_Batch_Experiments/Results",
    "Week3/Day14_Model_Evaluation",
    "Week3/Day15_OpenCompass",
    "Week3/Day16_Weekly_Report_and_Model_Archive",
}
EXPECTED_SUBMISSION_RUNS = {
    "rank-r8",
    "rank-r32",
    "rank-r64",
    "lr-1e-4",
    "lr-2e-4",
    "lr-5e-5",
    "epoch-e2",
    "epoch-e3",
    "epoch-e5",
}
EXPECTED_RAW_LOG_FILES = {
    "adapter_files.sha256",
    "config_diff.json",
    "environment.txt",
    "gpu_memory.csv",
    "preflight.json",
    "preflight.txt",
    "runtime_config.yaml",
    "source_config.yaml",
    "status.json",
    "train.log",
    "train_results.json",
    "trainer_state.json",
}
EXPECTED_WEEK3_FIXED_FILES = {
    "Week3/README.md",
    "Week3/Day11_Experiment_Design/README.md",
    "Week3/Day12_13_Batch_Experiments/README.md",
    "Week3/Day12_13_Batch_Experiments/Raw_Logs.sha256",
    "Week3/Day12_13_Batch_Experiments/Results/Experiment_Summary.csv",
    "Week3/Day12_13_Batch_Experiments/Results/Experiment_Summary.json",
    "Week3/Day14_Model_Evaluation/All_Responses.jsonl",
    "Week3/Day14_Model_Evaluation/Automatic_Evidence.json",
    "Week3/Day14_Model_Evaluation/Automatic_Evidence_Summary.csv",
    "Week3/Day14_Model_Evaluation/Best_Model.json",
    "Week3/Day14_Model_Evaluation/Eval_Harness.py",
    "Week3/Day14_Model_Evaluation/Evaluation_Questions.json",
    "Week3/Day14_Model_Evaluation/Evaluation_Rubric.json",
    "Week3/Day14_Model_Evaluation/Formal_Evaluation_Manifest.json",
    "Week3/Day14_Model_Evaluation/Formal_Validation.json",
    "Week3/Day14_Model_Evaluation/Human_Review_Status.json",
    "Week3/Day14_Model_Evaluation/Human_Score_Summary.csv",
    "Week3/Day14_Model_Evaluation/Human_Scoring_Guide.md",
    "Week3/Day14_Model_Evaluation/Model_Scores_Radar.png",
    "Week3/Day14_Model_Evaluation/README.md",
    "Week3/Day14_Model_Evaluation/Reviewer_1_Scores.csv",
    "Week3/Day14_Model_Evaluation/Reviewer_2_Scores.csv",
    "Week3/Day15_OpenCompass/OpenCompass_Scores.csv",
    "Week3/Day15_OpenCompass/README.md",
    "Week3/Day16_Weekly_Report_and_Model_Archive/Best_Model_Path.json",
    "Week3/Day16_Weekly_Report_and_Model_Archive/README.md",
    "Week3/Day16_Weekly_Report_and_Model_Archive/Week3_Acceptance_Matrix.md",
    "Week3/Day16_Weekly_Report_and_Model_Archive/Week3_SFT_Optimization_and_Evaluation_Report.md",
}
EXPECTED_WEEK3_SUBMISSION_FILES = frozenset(
    EXPECTED_WEEK3_FIXED_FILES
    | {
        f"Week3/Day12_13_Batch_Experiments/Raw_Logs/{run_id}/attempt-001/{filename}"
        for run_id in EXPECTED_SUBMISSION_RUNS
        for filename in EXPECTED_RAW_LOG_FILES
    }
)
FORBIDDEN_SUBMISSION_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".pyc",
    ".pt",
    ".pth",
    ".safetensors",
}
LOCAL_LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
MANIFEST_LINE_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")
DUPLICATE_COPY_PATTERN = re.compile(r"(?: \d+| \(\d+\)| copy|副本)$", re.IGNORECASE)


def validate_experiments(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    if len(rows) != 9:
        errors.append(f"expected 9 formal runs, found {len(rows)}")
    groups = {str(row.get("group", "")) for row in rows}
    if groups != EXPECTED_GROUPS:
        errors.append(f"expected experiment groups {sorted(EXPECTED_GROUPS)}, found {sorted(groups)}")
    run_ids = [str(row.get("run_id", "")) for row in rows]
    if len(set(run_ids)) != len(run_ids) or any(not run_id for run_id in run_ids):
        errors.append("run IDs must be present and unique")
    for row in rows:
        run_id = str(row.get("run_id", "<missing>"))
        status = str(row.get("status", ""))
        if status not in TERMINAL_STATUSES:
            errors.append(f"{run_id} has no terminal status: {status!r}")
        elif status != "completed":
            errors.append(f"{run_id} did not complete successfully: {status!r}")
        if status == "completed":
            for key in (
                "final_loss",
                "last_100_loss_mean",
                "train_runtime_seconds",
                "sampled_peak_gpu_memory_mib",
                "adapter_path",
            ):
                if row.get(key) in (None, ""):
                    errors.append(f"{run_id} completed without {key}")
    return errors


def validate_human_review(status: dict, rubric: dict) -> list[str]:
    errors: list[str] = []
    reviewers = status.get("reviewers", [])
    if len(reviewers) != 2:
        errors.append(f"expected exactly 2 human reviewers, found {len(reviewers)}")
    for index, reviewer in enumerate(reviewers, start=1):
        if reviewer.get("status") != "completed":
            errors.append(f"reviewer {index} is not completed")
        if reviewer.get("row_count") != 200:
            errors.append(f"reviewer {index} must contain 200 score rows")
    weights = rubric.get("weights", {})
    if weights != EXPECTED_WEIGHTS:
        errors.append(f"human score weights differ from the frozen rubric: {weights}")
    try:
        total = sum(float(value) for value in weights.values())
    except (TypeError, ValueError):
        total = -1.0
    if abs(total - 1.0) > 1e-9:
        errors.append(f"human score weights must sum to 1.0, found {total}")
    return errors


def validate_opencompass(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    pairs = {(str(row.get("model", "")), str(row.get("dataset", ""))) for row in rows}
    if pairs != EXPECTED_OPENCOMPASS_PAIRS or len(rows) != 4:
        errors.append(
            "OpenCompass table must contain exactly base/best_sft × ceval/cmmlu"
        )
    for row in rows:
        pair = (str(row.get("model", "")), str(row.get("dataset", "")))
        if row.get("status") != "completed":
            errors.append(f"OpenCompass result is not completed: {pair}")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"OpenCompass score is invalid: {pair}")
            continue
        if not 0.0 <= score <= 100.0:
            errors.append(f"OpenCompass score is out of range: {pair}={score}")
        dataset = pair[1]
        try:
            subject_count = int(row["subject_count"])
            question_count = int(row["question_count"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"OpenCompass coverage fields are invalid: {pair}")
            continue
        expected_subjects = EXPECTED_OPENCOMPASS_SUBJECTS.get(dataset)
        if subject_count != expected_subjects:
            errors.append(
                f"OpenCompass subject coverage is incomplete: {pair}="
                f"{subject_count}, expected {expected_subjects}"
            )
        if question_count <= 0:
            errors.append(f"OpenCompass question coverage is empty: {pair}")
        if row.get("aggregation") != "sample_weighted_accuracy":
            errors.append(f"OpenCompass aggregation differs from the frozen rule: {pair}")
    return errors


def contains_opencompass_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "opencompass" in str(key).lower() or contains_opencompass_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_opencompass_key(item) for item in value)
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _name_without_suffixes(name: str) -> str:
    suffixes = Path(name).suffixes
    if not suffixes:
        return name
    return name[: -sum(len(suffix) for suffix in suffixes)]


def _validate_submission_manifest(submission_root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = submission_root / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        return [f"missing Submission SHA-256 manifest: {manifest_path}"]

    manifest_rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = MANIFEST_LINE_PATTERN.fullmatch(line)
        if match is None:
            errors.append(f"invalid SHA-256 manifest line {line_number}")
            continue
        manifest_rows.append((match.group(1), match.group(2)))

    manifest_paths = [path for _, path in manifest_rows]
    if manifest_paths != sorted(manifest_paths):
        errors.append("Submission SHA-256 manifest paths are not sorted")
    if len(set(manifest_paths)) != len(manifest_paths):
        errors.append("Submission SHA-256 manifest contains duplicate paths")
    if "SHA256SUMS.txt" in manifest_paths:
        errors.append("Submission SHA-256 manifest must not list itself")
    if any(Path(path).name == ".DS_Store" for path in manifest_paths):
        errors.append("Submission SHA-256 manifest must not list .DS_Store")

    actual_paths = sorted(
        path.relative_to(submission_root).as_posix()
        for path in submission_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path != manifest_path
        and path.name != ".DS_Store"
        and path.suffix.lower() != ".zip"
    )
    missing = sorted(set(actual_paths) - set(manifest_paths))
    extra = sorted(set(manifest_paths) - set(actual_paths))
    errors.extend(f"Submission file missing from SHA-256 manifest: {path}" for path in missing)
    errors.extend(f"SHA-256 manifest path missing from Submission: {path}" for path in extra)

    for expected_digest, relative_path in manifest_rows:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"unsafe SHA-256 manifest path: {relative_path}")
            continue
        file_path = submission_root / candidate
        if not file_path.is_file() or file_path.is_symlink():
            continue
        actual_digest = sha256_file(file_path)
        if actual_digest != expected_digest:
            errors.append(f"SHA-256 mismatch: {relative_path}")
    return errors


def _validate_submission_links(submission_root: Path) -> list[str]:
    errors: list[str] = []
    for markdown_path in submission_root.rglob("*.md"):
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in LOCAL_LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target_path = unquote(target.split("#", 1)[0])
            if not target_path:
                continue
            if not (markdown_path.parent / target_path).resolve().exists():
                relative_markdown = markdown_path.relative_to(submission_root).as_posix()
                errors.append(f"broken Submission link: {relative_markdown} -> {target}")
    return errors


def validate_submission(submission_root: Path) -> list[str]:
    errors: list[str] = []
    if not submission_root.is_dir():
        return [f"missing Submission directory: {submission_root}"]

    for path in submission_root.rglob("*"):
        relative = path.relative_to(submission_root)
        relative_text = relative.as_posix()
        if not relative_text.isascii():
            errors.append(f"Submission path is not ASCII: {relative_text}")
        for component in relative.parts:
            if DUPLICATE_COPY_PATTERN.search(_name_without_suffixes(component)):
                errors.append(f"duplicate-copy path rejected: {relative_text}")
                break
        if path.is_symlink():
            errors.append(f"Submission symlink rejected: {relative_text}")
            continue
        if path.name == ".DS_Store":
            errors.append(f"Submission metadata rejected: {relative_text}")
        if path.is_dir() and (
            path.name in {"__pycache__", ".pytest_cache"}
            or path.name.startswith("checkpoint-")
        ):
            errors.append(f"Submission cache/checkpoint directory rejected: {relative_text}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUBMISSION_SUFFIXES:
            errors.append(f"Submission model weight or runtime artifact rejected: {relative_text}")

    for relative_path in sorted(REQUIRED_WEEK3_SUBMISSION_DIRECTORIES):
        if not (submission_root / relative_path).is_dir():
            errors.append(f"missing required Week 3 Submission directory: {relative_path}")

    week3_root = submission_root / "Week3"
    if week3_root.is_dir():
        actual_week3_files = {
            path.relative_to(submission_root).as_posix()
            for path in week3_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        missing_week3_files = sorted(
            EXPECTED_WEEK3_SUBMISSION_FILES - actual_week3_files
        )
        unexpected_week3_files = sorted(
            actual_week3_files - EXPECTED_WEEK3_SUBMISSION_FILES
        )
        errors.extend(
            f"missing expected Week 3 Submission file: {path}"
            for path in missing_week3_files
        )
        errors.extend(
            f"unexpected Week 3 Submission file: {path}"
            for path in unexpected_week3_files
        )

    raw_logs_root = (
        submission_root / "Week3/Day12_13_Batch_Experiments/Raw_Logs"
    )
    if raw_logs_root.is_dir():
        actual_runs = {path.name for path in raw_logs_root.iterdir() if path.is_dir()}
        if actual_runs != EXPECTED_SUBMISSION_RUNS:
            errors.append(
                "Submission raw-log run set differs from the frozen nine runs: "
                f"{sorted(actual_runs)}"
            )
        for run_id in sorted(EXPECTED_SUBMISSION_RUNS):
            attempt = raw_logs_root / run_id / "attempt-001"
            if not attempt.is_dir():
                errors.append(f"missing Submission raw-log attempt: {run_id}/attempt-001")
                continue
            actual_files = {path.name for path in attempt.iterdir() if path.is_file()}
            if actual_files != EXPECTED_RAW_LOG_FILES:
                errors.append(
                    f"Submission raw-log files differ for {run_id}: {sorted(actual_files)}"
                )

    questions_path = (
        submission_root / "Week3/Day14_Model_Evaluation/Evaluation_Questions.json"
    )
    if questions_path.is_file():
        questions = json.loads(questions_path.read_text(encoding="utf-8")).get("questions", [])
        if len(questions) != 20:
            errors.append(f"Submission must contain exactly 20 evaluation questions, found {len(questions)}")

    for reviewer_id in (1, 2):
        reviewer_path = (
            submission_root
            / f"Week3/Day14_Model_Evaluation/Reviewer_{reviewer_id}_Scores.csv"
        )
        if reviewer_path.is_file() and len(read_csv(reviewer_path)) != 200:
            errors.append(f"Submission reviewer {reviewer_id} must contain 200 score rows")

    opencompass_path = submission_root / "Week3/Day15_OpenCompass/OpenCompass_Scores.csv"
    if opencompass_path.is_file():
        errors.extend(validate_opencompass(read_csv(opencompass_path)))

    errors.extend(_validate_submission_manifest(submission_root))
    errors.extend(_validate_submission_links(submission_root))
    return errors


def validate_archive_files(archive: dict, hashes_path: Path, inventory_path: Path) -> list[str]:
    errors: list[str] = []
    for label, path, metadata_key in (
        ("merged-model hash manifest", hashes_path, "merged_model_manifest_sha256"),
        ("merged-model inventory", inventory_path, "merged_model_inventory_sha256"),
    ):
        if not path.is_file():
            errors.append(f"missing local {label}: {path}")
            continue
        actual = sha256_file(path)
        if actual != archive.get(metadata_key):
            errors.append(f"local {label} SHA-256 differs from archive metadata")
    if inventory_path.is_file():
        row_count = len(
            [line for line in inventory_path.read_text(encoding="utf-8").splitlines() if line]
        )
        if row_count != archive.get("merged_model_file_count"):
            errors.append(
                "local merged-model inventory count differs from archive metadata: "
                f"{row_count} != {archive.get('merged_model_file_count')}"
            )
    return errors


def build_best_model_metadata(best: dict, frozen: dict, archive: dict) -> dict:
    run_id = best.get("model_id") or best.get("run_id")
    if not run_id:
        raise ValueError("Day 14 best model has no run ID")
    if contains_opencompass_key(best):
        raise ValueError("Day 14 selection contains OpenCompass evidence")
    if frozen.get("selected_run_id") != run_id or archive.get("selected_run_id") != run_id:
        raise ValueError("selected run differs across Day 14, frozen input, and archive")
    if frozen.get("opencompass_used_for_selection") is not False:
        raise ValueError("OpenCompass must not be a selection input")
    if archive.get("merged_model_path") != MERGED_MODEL_PATH:
        raise ValueError("merged model path differs from the frozen Week 4 handoff")
    if archive.get("merged_model_load") != "PASS":
        raise ValueError("merged model did not pass independent loading")

    return {
        "status": "ready_for_week4_dpo",
        "selected_run_id": run_id,
        "base_model_path": BASE_MODEL_PATH,
        "adapter_path": frozen["adapter_path"],
        "adapter_model_sha256": frozen["adapter_model_sha256"],
        "merged_model_path": archive["merged_model_path"],
        "merged_model_manifest_sha256": archive["merged_model_manifest_sha256"],
        "merged_model_load": archive["merged_model_load"],
        "dpo_input_form": "merged_model",
        "selection_source": "Day 14 two-rater weighted human score",
        "human_weighted_total": best.get("weighted_total"),
        "opencompass_used_for_selection": False,
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[5],
    )
    parser.add_argument("--best-model-output", type=Path)
    return parser.parse_args()


def render_check(check_id: str, errors: Iterable[str]) -> tuple[dict, list[str]]:
    messages = list(errors)
    return (
        {
            "check": check_id,
            "status": "PASS" if not messages else "FAIL",
            "details": messages or ["all assertions passed"],
        },
        messages,
    )


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    day11 = root / "deliverables/week3/day11/source/manifests/experiment_matrix.json"
    summary_path = root / "deliverables/week3/day12_13/source/results/experiment_summary.json"
    review_status_path = root / "deliverables/week3/day14/source/results/human_review_status.json"
    rubric_path = root / "deliverables/week3/day14/source/data/evaluation_rubric.json"
    best_path = root / "deliverables/week3/day14/source/results/best_model.json"
    frozen_path = root / "deliverables/week3/day15/source/results/selected_model_input.json"
    scores_path = root / "deliverables/week3/day15/source/results/opencompass_scores.csv"
    archive_path = root / "deliverables/week3/day15/source/results/best_model_archive.json"
    model_hashes_path = (
        root / "deliverables/week3/day15/source/results/best_merged_model_files.sha256"
    )
    model_inventory_path = (
        root / "deliverables/week3/day15/source/results/best_merged_model_inventory.csv"
    )

    matrix = json.loads(day11.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    review_status = json.loads(review_status_path.read_text(encoding="utf-8"))
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    scores = read_csv(scores_path)
    archive = json.loads(archive_path.read_text(encoding="utf-8"))

    checks: list[dict] = []
    all_errors: list[str] = []
    matrix_errors = []
    if len(matrix.get("runs", [])) != 9:
        matrix_errors.append("Day 11 matrix must contain 9 runs")
    if {row.get("group") for row in matrix.get("runs", [])} != EXPECTED_GROUPS:
        matrix_errors.append("Day 11 matrix must contain the three experiment groups")
    for check_id, errors in (
        ("day11_matrix", matrix_errors),
        ("day12_13_experiments", validate_experiments(summary)),
        ("day14_human_review", validate_human_review(review_status, rubric)),
        ("day15_opencompass", validate_opencompass(scores)),
    ):
        rendered, messages = render_check(check_id, errors)
        checks.append(rendered)
        all_errors.extend(messages)

    model_errors = validate_archive_files(archive, model_hashes_path, model_inventory_path)
    try:
        metadata = build_best_model_metadata(best, frozen, archive)
    except (KeyError, TypeError, ValueError) as error:
        metadata = None
        model_errors.append(str(error))
    rendered, messages = render_check("best_model_archive", model_errors)
    checks.append(rendered)
    all_errors.extend(messages)

    rendered, messages = render_check(
        "submission_package", validate_submission(root / "Submission")
    )
    checks.append(rendered)
    all_errors.extend(messages)

    payload = {
        "status": "PASS" if not all_errors else "FAIL",
        "checks": checks,
        "error_count": len(all_errors),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if metadata is not None and args.best_model_output is not None:
        args.best_model_output.parent.mkdir(parents=True, exist_ok=True)
        args.best_model_output.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

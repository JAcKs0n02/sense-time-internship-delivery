import importlib.util
import hashlib
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_week3.py"
REPO_ROOT = SCRIPT.resolve().parents[5]


def load_module():
    spec = importlib.util.spec_from_file_location("validate_week3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def completed_run(run_id: str, group: str) -> dict:
    return {
        "run_id": run_id,
        "group": group,
        "status": "completed",
        "final_loss": 1.0,
        "last_100_loss_mean": 1.1,
        "train_runtime_seconds": 100.0,
        "sampled_peak_gpu_memory_mib": 20000,
        "adapter_path": f"runs/{run_id}/attempt-001/adapter",
    }


def test_experiment_gate_requires_nine_terminal_runs_and_three_groups():
    module = load_module()
    groups = ["rank"] * 3 + ["learning_rate"] * 3 + ["epoch"] * 3
    rows = [completed_run(f"run-{index}", group) for index, group in enumerate(groups)]

    assert module.validate_experiments(rows) == []

    rows[0]["status"] = "running"
    errors = module.validate_experiments(rows)
    assert any("terminal status" in error for error in errors)


def test_experiment_gate_requires_all_nine_formal_runs_to_complete():
    module = load_module()
    groups = ["rank"] * 3 + ["learning_rate"] * 3 + ["epoch"] * 3
    rows = [completed_run(f"run-{index}", group) for index, group in enumerate(groups)]
    rows[4]["status"] = "failed"

    errors = module.validate_experiments(rows)

    assert any("did not complete" in error for error in errors)


def test_human_gate_requires_two_completed_reviewers_and_fixed_weights():
    module = load_module()
    status = {
        "reviewers": [
            {"status": "completed", "row_count": 200},
            {"status": "completed", "row_count": 200},
        ]
    }
    rubric = {
        "weights": {
            "accuracy": 0.30,
            "completeness": 0.25,
            "logic": 0.20,
            "safety": 0.15,
            "format": 0.10,
        }
    }

    assert module.validate_human_review(status, rubric) == []

    status["reviewers"].pop()
    assert module.validate_human_review(status, rubric)


def test_opencompass_gate_requires_four_completed_pairs():
    module = load_module()
    rows = [
        {
            "model": model,
            "dataset": dataset,
            "score": score,
            "status": "completed",
            "subject_count": str(52 if dataset == "ceval" else 67),
            "question_count": str(1346 if dataset == "ceval" else 11528),
            "aggregation": "sample_weighted_accuracy",
        }
        for model, score in (("base", "70"), ("best_sft", "68"))
        for dataset in ("ceval", "cmmlu")
    ]

    assert module.validate_opencompass(rows) == []

    rows.pop()
    assert module.validate_opencompass(rows)


def test_opencompass_gate_rejects_partial_subject_coverage():
    module = load_module()
    rows = [
        {
            "model": model,
            "dataset": dataset,
            "score": "60",
            "status": "completed",
            "subject_count": str(52 if dataset == "ceval" else 67),
            "question_count": "1",
            "aggregation": "sample_weighted_accuracy",
        }
        for model in ("base", "best_sft")
        for dataset in ("ceval", "cmmlu")
    ]
    rows[0]["subject_count"] = "1"

    errors = module.validate_opencompass(rows)

    assert any("subject coverage" in error for error in errors)


def test_best_model_metadata_is_day14_selected_and_dpo_ready():
    module = load_module()
    best = {
        "model_id": "epoch-e5",
        "adapter_path": "runs/epoch-e5/attempt-001/adapter",
        "weighted_total": 3.78875,
    }
    frozen = {
        "selected_run_id": "epoch-e5",
        "adapter_path": "/root/autodl-tmp/qwen25-week3/runs/epoch-e5/attempt-001/adapter",
        "adapter_model_sha256": "a" * 64,
        "opencompass_used_for_selection": False,
    }
    archive = {
        "selected_run_id": "epoch-e5",
        "merged_model_path": "/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged",
        "merged_model_load": "PASS",
        "merged_model_manifest_sha256": "b" * 64,
    }

    metadata = module.build_best_model_metadata(best, frozen, archive)

    assert metadata["selected_run_id"] == "epoch-e5"
    assert metadata["dpo_input_form"] == "merged_model"
    assert metadata["opencompass_used_for_selection"] is False
    assert metadata["merged_model_load"] == "PASS"


def test_default_repo_root_points_to_repository(monkeypatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["validate_week3.py"])

    args = module.parse_args()

    assert args.repo_root == SCRIPT.resolve().parents[5]


def test_archive_gate_hashes_local_inventory_and_manifest(tmp_path):
    module = load_module()
    hashes = tmp_path / "best_merged_model_files.sha256"
    inventory = tmp_path / "best_merged_model_inventory.csv"
    hashes.write_text("a" * 64 + "  /models/best/config.json\n", encoding="utf-8")
    inventory.write_text("config.json,10\n", encoding="utf-8")
    archive = {
        "merged_model_manifest_sha256": module.sha256_file(hashes),
        "merged_model_inventory_sha256": module.sha256_file(inventory),
        "merged_model_file_count": 1,
    }

    assert module.validate_archive_files(archive, hashes, inventory) == []

    inventory.write_text("config.json,11\n", encoding="utf-8")
    assert module.validate_archive_files(archive, hashes, inventory)


def test_repository_submission_package_passes_contract():
    module = load_module()

    assert module.validate_submission(REPO_ROOT / "Submission") == []


def test_submission_gate_rejects_hidden_metadata_duplicate_paths_and_weights(tmp_path):
    module = load_module()
    submission = tmp_path / "Submission"
    submission.mkdir()
    (submission / ".DS_Store").write_bytes(b"finder metadata")
    (submission / "Week3" / "Day14_Model_Evaluation 2").mkdir(parents=True)
    (submission / "Week3" / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    (submission / "model.safetensors").write_bytes(b"weights")
    (submission / "README.md").write_text("submission\n", encoding="utf-8")
    (submission / "README-link.md").symlink_to(submission / "README.md")
    (submission / "SHA256SUMS.txt").write_text("", encoding="utf-8")

    errors = module.validate_submission(submission)

    assert any(".DS_Store" in error for error in errors)
    assert any("duplicate-copy" in error for error in errors)
    assert any("unexpected Week 3" in error for error in errors)
    assert any("model weight" in error for error in errors)
    assert any("symlink" in error for error in errors)


def test_submission_gate_rejects_stale_or_incomplete_manifest(tmp_path):
    module = load_module()
    submission = tmp_path / "Submission"
    submission.mkdir()
    readme = submission / "README.md"
    readme.write_text("submission\n", encoding="utf-8")
    wrong_digest = hashlib.sha256(b"different content").hexdigest()
    (submission / "SHA256SUMS.txt").write_text(
        f"{wrong_digest}  README.md\n",
        encoding="utf-8",
    )

    errors = module.validate_submission(submission)

    assert any("SHA-256 mismatch" in error for error in errors)


def test_submission_manifest_ignores_generated_zip_archives(tmp_path):
    module = load_module()
    submission = tmp_path / "Submission"
    submission.mkdir()
    readme = submission / "README.md"
    readme.write_text("submission\n", encoding="utf-8")
    (submission / "Week3.zip").write_bytes(b"generated transport archive")
    digest = hashlib.sha256(readme.read_bytes()).hexdigest()
    (submission / "SHA256SUMS.txt").write_text(
        f"{digest}  README.md\n",
        encoding="utf-8",
    )

    errors = module._validate_submission_manifest(submission)

    assert errors == []

import json
import sys
from pathlib import Path


DAY12_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY12_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from collect_experiment_evidence import collect_evidence  # noqa: E402


def write_attempt(root: Path, status: str = "completed") -> Path:
    attempt = root / "rank-r8" / "attempt-001"
    adapter = attempt / "adapter"
    adapter.mkdir(parents=True)
    required = {
        "train.log": "real training log\n",
        "gpu_memory.csv": "timestamp, 8000, 90\n",
        "source_config.yaml": "seed: 42\n",
        "runtime_config.yaml": "seed: 42\n",
        "config_diff.json": "{}\n",
        "preflight.txt": "ok\n",
        "environment.txt": "python\n",
        "adapter_files.sha256": "hash  adapter_model.safetensors\n",
    }
    for name, content in required.items():
        (attempt / name).write_text(content, encoding="utf-8")
    (attempt / "status.json").write_text(
        json.dumps({"status": status, "exit_code": 0 if status == "completed" else 1}) + "\n",
        encoding="utf-8",
    )
    (adapter / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    (adapter / "train_results.json").write_text("{}\n", encoding="utf-8")
    return attempt


def test_collector_copies_allowlisted_logs_and_trainer_results(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    write_attempt(run_root)
    output = tmp_path / "evidence" / "raw_logs"

    errors = collect_evidence(run_root, output, ["rank-r8"])

    assert errors == []
    collected = output / "rank-r8" / "attempt-001"
    assert (collected / "train.log").read_text(encoding="utf-8") == "real training log\n"
    assert (collected / "trainer_state.json").is_file()
    assert (collected / "train_results.json").is_file()


def test_collector_never_copies_weights_or_checkpoints(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    attempt = write_attempt(run_root)
    (attempt / "adapter" / "adapter_model.safetensors").write_bytes(b"weight")
    checkpoint = attempt / "adapter" / "checkpoint-500"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    output = tmp_path / "evidence" / "raw_logs"

    errors = collect_evidence(run_root, output, ["rank-r8"])

    assert errors == []
    assert not list(output.rglob("*.safetensors"))
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("checkpoint-*"))


def test_collector_rejects_symlinks(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    attempt = write_attempt(run_root)
    target = tmp_path / "outside.log"
    target.write_text("outside\n", encoding="utf-8")
    (attempt / "linked.log").symlink_to(target)

    errors = collect_evidence(run_root, tmp_path / "evidence" / "raw_logs", ["rank-r8"])

    assert any("symlink" in error for error in errors)


def test_collector_reports_missing_run(tmp_path: Path) -> None:
    errors = collect_evidence(tmp_path / "runs", tmp_path / "evidence" / "raw_logs", ["rank-r8"])

    assert errors == ["run has no attempts: rank-r8"]

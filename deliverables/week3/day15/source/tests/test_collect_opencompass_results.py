import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_opencompass_results.py"


def load_module():
    spec = importlib.util.spec_from_file_location("collect_opencompass_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_result(path: Path, accuracy: float, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"accuracy": accuracy, "details": {str(i): {} for i in range(count)}}),
        encoding="utf-8",
    )


def make_model(root: Path, label: str, model_dir: str, scores: list[tuple[str, float, int]]) -> None:
    attempt = root / label / "attempt-001"
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text(
        json.dumps({"status": "completed", "exit_code": 0}), encoding="utf-8"
    )
    result_dir = attempt / "20260807_120000" / "results" / model_dir
    for name, score, count in scores:
        write_result(result_dir / f"{name}.json", score, count)


def test_collects_four_sample_weighted_benchmark_rows(tmp_path):
    module = load_module()
    make_model(
        tmp_path,
        "base",
        "base_hf",
        [("ceval-a", 50.0, 10), ("ceval-b", 100.0, 30), ("cmmlu-a", 25.0, 20)],
    )
    make_model(
        tmp_path,
        "best_sft",
        "best_hf",
        [("ceval-a", 60.0, 10), ("cmmlu-a", 40.0, 20)],
    )

    rows = module.collect_formal_results(tmp_path)

    assert {(row["model"], row["dataset"]) for row in rows} == {
        ("base", "ceval"),
        ("base", "cmmlu"),
        ("best_sft", "ceval"),
        ("best_sft", "cmmlu"),
    }
    base_ceval = next(
        row for row in rows if row["model"] == "base" and row["dataset"] == "ceval"
    )
    assert base_ceval["score"] == 87.5
    assert base_ceval["macro_subject_score"] == 75.0
    assert base_ceval["question_count"] == 40
    assert base_ceval["subject_count"] == 2

import hashlib
import json
import sys
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DAY14_ROOT))

from eval_harness import (  # noqa: E402
    DEFAULT_GENERATION_CONFIG,
    build_messages,
    build_candidates,
    make_result,
    run_evaluation,
    validate_candidate_paths,
)


QUESTIONS = [
    {
        "id": "MATH-01",
        "category": "math",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "reference": {"answer": "2"},
        "automatic_check": {"type": "numeric", "expected": 2, "tolerance": 0},
        "difficulty": "hard",
        "human_scoring_notes": "检查结果。",
    }
]


class FakeRunner:
    def __init__(self, base_model: str, adapter_path: str | None):
        self.base_model = base_model
        self.adapter_path = adapter_path

    def generate(self, messages: list[dict], generation_config: dict) -> dict:
        assert generation_config == DEFAULT_GENERATION_CONFIG
        return {
            "raw_response": "2",
            "input_tokens": 8,
            "output_tokens": 1,
            "peak_gpu_memory_bytes": 1024,
        }

    def close(self) -> None:
        return None


def fake_factory(base_model: str, adapter_path: str | None) -> FakeRunner:
    return FakeRunner(base_model, adapter_path)


def test_adapter_and_base_use_identical_generation_config() -> None:
    base = run_evaluation("base", None, QUESTIONS, model_factory=fake_factory)
    sft = run_evaluation("base", "adapter", QUESTIONS, model_factory=fake_factory)

    assert base[0]["generation_config"] == sft[0]["generation_config"]
    assert base[0]["status"] == "completed"
    assert sft[0]["status"] == "completed"


def test_system_message_is_applied_before_the_user_prompt() -> None:
    assert build_messages("system rule", QUESTIONS[0]["messages"]) == [
        {"role": "system", "content": "system rule"},
        {"role": "user", "content": "1+1=?"},
    ]


def test_raw_response_is_never_rewritten() -> None:
    result = make_result(
        candidate_id="candidate",
        question=QUESTIONS[0],
        raw_response="wrong but original",
        generation_config=DEFAULT_GENERATION_CONFIG,
        input_tokens=4,
        output_tokens=3,
        elapsed_seconds=0.2,
    )

    assert result["raw_response"] == "wrong but original"
    assert result["messages"] == QUESTIONS[0]["messages"]


def test_generation_failure_is_a_terminal_record() -> None:
    class BrokenRunner(FakeRunner):
        def generate(self, messages: list[dict], generation_config: dict) -> dict:
            raise RuntimeError("synthetic failure")

    records = run_evaluation(
        "base",
        None,
        QUESTIONS,
        model_factory=lambda base, adapter: BrokenRunner(base, adapter),
    )

    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["raw_response"] is None
    assert "synthetic failure" in records[0]["error"]


def test_model_load_failure_records_every_question_and_continues_contract() -> None:
    emitted = []

    records = run_evaluation(
        "base",
        "broken-adapter",
        QUESTIONS,
        candidate_id="broken",
        model_factory=lambda _base, _adapter: (_ for _ in ()).throw(
            RuntimeError("adapter load failed")
        ),
        on_record=emitted.append,
    )

    assert records == emitted
    assert len(records) == len(QUESTIONS)
    assert all(row["status"] == "failed_model_load" for row in records)
    assert all("adapter load failed" in row["error"] for row in records)


def test_each_terminal_record_is_emitted_immediately() -> None:
    emitted = []

    records = run_evaluation(
        "base",
        None,
        QUESTIONS,
        model_factory=fake_factory,
        on_record=emitted.append,
    )

    assert emitted == records


def test_candidates_come_only_from_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "completed_adapters.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "rank-r8",
                    "status": "completed",
                    "adapter_path": "runs/rank-r8/attempt-001/adapter",
                }
            ]
        ),
        encoding="utf-8",
    )

    candidates = build_candidates(
        manifest_path,
        week3_root=Path("/root/autodl-tmp/qwen25-week3"),
        expected_adapter_count=1,
    )

    assert candidates == [
        {"candidate_id": "base", "adapter_path": None, "kind": "base"},
        {
            "candidate_id": "rank-r8",
            "adapter_path": "/root/autodl-tmp/qwen25-week3/runs/rank-r8/attempt-001/adapter",
            "kind": "sft",
        },
    ]


def test_formal_path_gate_requires_base_and_every_adapter(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()

    errors = validate_candidate_paths(
        str(base),
        [
            {"candidate_id": "base", "adapter_path": None, "kind": "base"},
            {"candidate_id": "rank-r8", "adapter_path": str(adapter), "kind": "sft"},
        ],
    )

    assert errors == []
    adapter.rmdir()
    assert validate_candidate_paths(
        str(base),
        [
            {"candidate_id": "base", "adapter_path": None, "kind": "base"},
            {"candidate_id": "rank-r8", "adapter_path": str(adapter), "kind": "sft"},
        ],
    ) == [f"missing adapter directory: rank-r8: {adapter}"]


def test_result_records_question_content_hash() -> None:
    expected = hashlib.sha256(
        json.dumps(
            QUESTIONS[0],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    result = make_result(
        candidate_id="base",
        question=QUESTIONS[0],
        raw_response="2",
        generation_config=DEFAULT_GENERATION_CONFIG,
        input_tokens=4,
        output_tokens=1,
        elapsed_seconds=0.2,
    )

    assert result["question_sha256"] == expected

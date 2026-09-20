from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
DAY32 = REPO_ROOT / "deliverables/week6/day32/source"
DAY31 = REPO_ROOT / "deliverables/week6/day31/source"


def _load_evaluator():
    path = DAY32 / "scripts/evaluate_agent.py"
    spec = importlib.util.spec_from_file_location("day32_evaluator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_cli_binds_frozen_eval_baseline_and_prompt_manifest(tmp_path: Path) -> None:
    """Catches Day32 starting from copied files whose hashes do not match Day31 evidence."""
    script = DAY32 / "scripts/prepare_day32.py"
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--frozen",
            str(DAY31 / "data/tool_sft_eval_frozen.json"),
            "--day31-frozen",
            str(DAY31 / "data/tool_sft_eval_frozen.json"),
            "--baseline",
            str(DAY31 / "results/post_eval.json"),
            "--prompt-v1",
            str(DAY32 / "configs/system_prompt_main.txt"),
            "--prompt-v2",
            str(DAY32 / "configs/system_prompt_ablation_input_fidelity.txt"),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(
        (DAY31 / "data/tool_sft_eval_frozen.json").read_bytes()
    ).hexdigest()
    assert report["status"] == "ready"
    assert report["frozen_eval_sha256"] == expected_sha
    assert report["baseline_sha256"] == hashlib.sha256(
        (DAY31 / "results/post_eval.json").read_bytes()
    ).hexdigest()
    assert report["case_count"] == 30
    assert report["adapter_sha256"] == (
        "7d7672cbbe7679d5badb20b040870b616e17c7093813baeb6605ee498925623c"
    )
    assert report["prompt_manifest"]["change_scope"] == "single_rule_prompt_ablation"


def test_evaluation_cli_exposes_prompt_bound_formal_interface() -> None:
    """Catches the remote runner lacking the prompt manifest required for the ablation."""
    script = DAY32 / "scripts/evaluate_agent.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--prompt-manifest" in completed.stdout
    assert "--compare-to" in completed.stdout
    assert "--prompt-variant" in completed.stdout
    assert "--eval-file" in completed.stdout
    assert "--dataset-manifest" in completed.stdout
    assert "--policy-mode" in completed.stdout


def test_day32_inference_gate_reuses_signed_parser_receipt_without_git_replay(
    tmp_path: Path,
) -> None:
    """Day32 inference must not depend on LLaMA-Factory checkout metadata still existing."""
    module = _load_evaluator()
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "merged_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "ready",
                "lineage": {
                    "scheme": "A",
                    "load_mode": "verified_week4_merged",
                    "model_dir": str(model),
                    "merged_manifest_path": str(manifest),
                    "merged_manifest_sha256": module.WEEK4_MERGED_MANIFEST_SHA256,
                },
                "official_parser_smoke": {"exit_code": 0, "receipt_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def resolver(**kwargs):
        calls.append(kwargs)
        return {
            "scheme": "A",
            "load_mode": "verified_week4_merged",
            "model_dir": str(model.resolve()),
            "merged_manifest_path": str(manifest.resolve()),
            "merged_manifest_sha256": module.WEEK4_MERGED_MANIFEST_SHA256,
        }

    result = module.verify_day32_inference_lineage(
        preflight, model, tmp_path / "agent_config.json", resolver=resolver
    )
    assert result["lineage"]["model_dir"] == str(model)
    assert len(calls) == 1

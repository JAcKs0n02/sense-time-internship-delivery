from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION = REPO_ROOT / "Submission" / "Week6"
DAY28 = SUBMISSION / "Day28_LangChain_and_Basic_Tools"
DAY29 = SUBMISSION / "Day29_ReAct_Agent"
DAY30 = SUBMISSION / "Day30_Complex_Tools_and_Multistep"
DAY31 = SUBMISSION / "Day31_Tool_Call_SFT"
DAY32 = SUBMISSION / "Day32_Error_Analysis"
DAY33 = SUBMISSION / "Day33_Weekly_Report"


def test_teacher_submission_has_shallow_day_folders() -> None:
    assert {path.name for path in DAY28.iterdir()} == {
        "Code",
        "Data",
        "Results",
        "README.md",
        "requirements.txt",
        "Submission_Map.json",
    }
    assert {path.name for path in DAY29.iterdir()} == {
        "Code",
        "Config",
        "Results",
        "Scripts",
        "Provenance",
        "README.md",
        "requirements.txt",
        "Submission_Map.json",
    }
    assert {path.name for path in DAY30.iterdir()} == {
        "Code",
        "Data",
        "Results",
        "Scripts",
        "README.md",
        "requirements.txt",
        "Submission_Map.json",
    }
    assert {path.name for path in DAY31.iterdir()} == {
        "Config",
        "Data",
        "Model_Archive",
        "Results",
        "Scripts",
        "README.md",
        "Submission_Map.json",
    }
    assert {path.name for path in DAY32.iterdir()} == {
        "Code",
        "Config",
        "Data",
        "Results",
        "Scripts",
        "README.md",
        "Submission_Map.json",
    }
    assert {path.name for path in DAY33.iterdir()} == {
        "Code",
        "Model_Archive",
        "Report",
        "Results",
        "Scripts",
        "README.md",
        "Submission_Map.json",
    }
    assert not list(SUBMISSION.rglob("repository"))
    assert not list(SUBMISSION.rglob("deliverables"))
    assert not list(SUBMISSION.rglob("__pycache__"))
    assert not list(SUBMISSION.rglob(".pytest_cache"))
    assert not list(SUBMISSION.rglob("*.pyc"))


def test_submission_maps_preserve_formal_file_bytes() -> None:
    mapped_roots = {
        DAY28: {"Code", "Data", "Results"},
        DAY29: {"Code", "Config", "Results", "Scripts", "Provenance"},
        DAY30: {"Code", "Data", "Results", "Scripts"},
        DAY31: {"Config", "Data", "Model_Archive", "Results", "Scripts"},
        DAY32: {"Code", "Config", "Data", "Results", "Scripts"},
        DAY33: {
            "Code",
            "Model_Archive",
            "Report",
            "Results",
            "Scripts",
        },
    }
    for day, roots in mapped_roots.items():
        mapping = json.loads((day / "Submission_Map.json").read_text(encoding="utf-8"))
        assert mapping["schema_version"] == "1.0"
        assert mapping["files"]
        submission_paths = [record["submission_path"] for record in mapping["files"]]
        formal_sources = [record["formal_source"] for record in mapping["files"]]
        assert len(submission_paths) == len(set(submission_paths))
        assert len(formal_sources) == len(set(formal_sources))
        assert all(
            not Path(path).is_absolute() and ".." not in Path(path).parts
            for path in submission_paths + formal_sources
        )
        eligible = {
            path.relative_to(day).as_posix()
            for root_name in roots
            for path in (day / root_name).rglob("*")
            if path.is_file() and path.name != "README.md"
        }
        assert set(submission_paths) == eligible
        for record in mapping["files"]:
            source = REPO_ROOT / record["formal_source"]
            submitted = day / record["submission_path"]
            assert source.is_file()
            assert submitted.is_file()
            source_bytes = source.read_bytes()
            assert submitted.read_bytes() == source_bytes
            assert hashlib.sha256(source_bytes).hexdigest() == record["sha256"]


def test_results_and_checksum_manifest_remain_valid() -> None:
    day28 = json.loads((DAY28 / "Results" / "day28_validation.json").read_text())
    day29 = json.loads((DAY29 / "Results" / "day29_validation.json").read_text())
    trace = json.loads((DAY29 / "Results" / "single_turn_trace.json").read_text())
    day30_acceptance = json.loads(
        (DAY30 / "Results" / "teacher_acceptance.json").read_text()
    )
    day30_trace = json.loads(
        (DAY30 / "Results" / "teacher_multistep_trace.json").read_text()
    )
    assert day28["status"] == "PASS"
    assert day29["status"] == "PASS" and day29["failed_checks"] == []
    assert trace["trace"][0]["action"] == "calculator"
    assert trace["trace"][0]["observation"]["data"]["value"] == 56088
    assert day30_acceptance["status"] == "PASS"
    assert [row["action"] for row in day30_trace["trace"]] == [
        "knowledge_retrieval",
        "calculator",
    ]
    assert day30_trace["trace"][1]["observation"]["data"]["value"] == 719
    assert day30_trace["side_effect_marker_exists_after"] is False

    day31_data = json.loads(
        (DAY31 / "Results" / "data_validation_v2.json").read_text()
    )
    day31_parser = json.loads(
        (DAY31 / "Results" / "official_parser_receipt_v2.json").read_text()
    )
    day31_preflight = json.loads(
        (DAY31 / "Results" / "formal_preflight_v2.json").read_text()
    )
    day31_training = json.loads(
        (DAY31 / "Results" / "training_summary_v2.json").read_text()
    )
    day31_adapter = json.loads(
        (DAY31 / "Model_Archive" / "adapter_manifest.json").read_text()
    )
    checkpoint_selection = json.loads(
        (DAY31 / "Results" / "checkpoint_selection_v2.json").read_text()
    )
    final_test = json.loads(
        (DAY31 / "Results" / "final_test_summary_v2.json").read_text()
    )
    assert day31_data["valid"] is True
    assert day31_data["counts"]["train_core_100"] == 100
    assert day31_data["counts"]["train_v2"] == 300
    assert day31_data["counts"]["dev_v2"] == 40
    assert day31_data["counts"]["test_v2"] == 100
    assert day31_parser["exit_code"] == 0
    assert day31_parser["train_aligned_rows"] == 300
    assert day31_parser["eval_aligned_rows"] == 40
    assert day31_preflight["status"] == "ready"
    assert day31_training["status"] == "collected"
    assert day31_adapter["status"] == "collected"
    assert checkpoint_selection["selected_checkpoint"].endswith("checkpoint-38")
    assert checkpoint_selection["selection_metrics"]["strict_success"] == 0.45
    assert final_test["results"]["raw"]["case_count"] == 100
    assert final_test["results"]["raw"]["metrics"]["first_tool_choice"] >= 0.9
    assert final_test["results"]["raw"]["metrics"]["full_tool_sequence"] >= 0.9

    day32_comparison = json.loads(
        (DAY32 / "Results" / "prompt_comparison.json").read_text()
    )
    day33_validation = json.loads(
        (DAY33 / "Results" / "day33_validation.json").read_text()
    )
    final_archive = json.loads(
        (DAY33 / "Results" / "final_agent_archive.json").read_text()
    )
    assert day32_comparison["baseline_unique_failed_cases"] == 22
    assert day32_comparison["optimized_unique_failed_cases"] == 26
    assert len(day32_comparison["regression_cases"]) == 5
    assert day33_validation["status"] == "PASS"
    assert final_archive["deployment_recommendation"] == "checkpoint_38_plus_main_prompt"
    assert final_archive["prompt"]["variant"] == "main"

    checksum_path = SUBMISSION / "SHA256SUMS.txt"
    lines = checksum_path.read_text().splitlines()
    assert lines
    listed_paths = [line.split("  ", 1)[1].removeprefix("./") for line in lines]
    assert len(listed_paths) == len(set(listed_paths))
    submitted_paths = {
        path.relative_to(SUBMISSION).as_posix()
        for path in SUBMISSION.rglob("*")
        if path.is_file()
        and path != checksum_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert set(listed_paths) == submitted_paths
    for line in lines:
        expected, relative = line.split("  ", 1)
        assert "repository/" not in relative and "deliverables/" not in relative
        target = SUBMISSION / relative.removeprefix("./")
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected


def test_day30_transitive_integrity_audit_matches_current_sources() -> None:
    audit = json.loads(
        (DAY30 / "Results" / "transitive_integrity_audit.json").read_text()
    )
    teacher_trace = json.loads(
        (DAY30 / "Results" / "teacher_multistep_trace.json").read_text()
    )
    run_spec = json.loads((DAY30 / "Results" / "run_spec.json").read_text())

    assert audit["status"] == "PASS_WITH_SCOPE"
    assert audit["audit_timing"] == "post_run"
    assert audit["run_spec_binding"]["run_spec_sha256"] == run_spec[
        "run_spec_sha256"
    ]
    assert audit["run_spec_binding"]["teacher_trace_run_spec_sha256"] == teacher_trace[
        "run_spec_sha256"
    ]
    for record in audit["files"]:
        source = REPO_ROOT / record["path"]
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["sha256"]

    retrieval = teacher_trace["trace"][0]["observation"]["data"]
    assert retrieval["knowledge_base_sha256"] == audit["knowledge_base"][
        "products_sha256"
    ]
    assert audit["knowledge_base"]["observation_matches_current_products"] is True

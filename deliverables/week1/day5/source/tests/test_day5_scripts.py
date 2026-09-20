import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


DAY5 = Path(__file__).resolve().parents[2]
SCRIPTS = DAY5 / "source" / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluateIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "evaluate_identity", SCRIPTS / "evaluate_identity.py"
        )

    def test_validate_prompts_requires_unique_ids_and_expected_fields(self):
        prompts = [
            {
                "id": "zh_name",
                "messages": [{"role": "user", "content": "你叫什么？"}],
                "expected_name": "Qwen",
                "expected_author": "Alibaba Cloud",
            }
        ]
        self.assertEqual(self.module.validate_prompts(prompts), prompts)

        duplicated = prompts + [dict(prompts[0])]
        with self.assertRaises(ValueError):
            self.module.validate_prompts(duplicated)

    def test_score_response_checks_name_author_and_conflict(self):
        prompt = {
            "id": "identity",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "expected_name": "Qwen",
            "expected_author": "Alibaba Cloud",
            "conflict_terms": ["OpenAI", "ChatGPT"],
        }
        good = self.module.score_response(
            prompt, "I am Qwen, created by Alibaba Cloud."
        )
        self.assertTrue(good["name_match"])
        self.assertTrue(good["author_match"])
        self.assertFalse(good["has_conflict"])
        self.assertTrue(good["passed"])

        bad = self.module.score_response(prompt, "I am ChatGPT from OpenAI.")
        self.assertFalse(bad["name_match"])
        self.assertFalse(bad["author_match"])
        self.assertTrue(bad["has_conflict"])
        self.assertFalse(bad["passed"])

    def test_summarize_records_counts_passes(self):
        records = [
            {"score": {"passed": True}},
            {"score": {"passed": False}},
            {"score": {"passed": True}},
        ]
        summary = self.module.summarize_records(records)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertAlmostEqual(summary["pass_rate"], 2 / 3)

    def test_bf16_load_plan_avoids_incompatible_auto_dispatch(self):
        marker = object()
        plan = self.module.build_model_load_kwargs(marker)
        self.assertIs(plan["torch_dtype"], marker)
        self.assertTrue(plan["low_cpu_mem_usage"])
        self.assertTrue(plan["trust_remote_code"])
        self.assertNotIn("device_map", plan)
        self.assertNotIn("quantization_config", plan)


class ValidateDay5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("validate_day5", SCRIPTS / "validate_day5.py")

    def test_read_jsonl_rejects_empty_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(json.dumps({"id": "one", "response": ""}) + "\n")
            with self.assertRaises(AssertionError):
                self.module.read_jsonl(path)

    def test_compare_runs_requires_identical_prompts_and_generation(self):
        common = {
            "id": "one",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "generation": {"do_sample": False, "max_new_tokens": 64},
            "response": "answer",
        }
        self.module.compare_runs([common], [dict(common, response="new answer")])

        changed = dict(common)
        changed["generation"] = {"do_sample": True, "max_new_tokens": 64}
        with self.assertRaises(AssertionError):
            self.module.compare_runs([common], [changed])

    def test_run2_changes_only_epochs_and_output_identity(self):
        def parse_simple_yaml(path: Path):
            values = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                values[key.strip()] = value.strip()
            return values

        run1 = parse_simple_yaml(DAY5 / "configs/qwen25_7b_identity_qlora.yaml")
        run2 = parse_simple_yaml(DAY5 / "configs/qwen25_7b_identity_qlora_run2.yaml")
        differences = {key for key in run1 if run1[key] != run2[key]}
        self.assertEqual(
            differences, {"output_dir", "run_name", "num_train_epochs"}
        )
        self.assertEqual(run1["num_train_epochs"], "3.0")
        self.assertEqual(run2["num_train_epochs"], "5.0")

    def test_complete_day5_delivery_passes_read_only_validation(self):
        result = self.module.validate(DAY5)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["before_passed"], 0)
        self.assertEqual(result["run1_passed"], 6)
        self.assertEqual(result["run2_passed"], 7)
        self.assertEqual(result["selected_run"], "run2")


if __name__ == "__main__":
    unittest.main()

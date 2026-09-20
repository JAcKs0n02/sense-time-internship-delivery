import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


DAY3_SOURCE = Path(__file__).resolve().parents[1]
SCRIPT_PATH = DAY3_SOURCE / "scripts" / "analyze_params.py"
CONFIG_PATH = DAY3_SOURCE / "config" / "config.json"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_params", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeParamsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_qwen25_7b_expected_counts(self):
        result = self.module.analyze_config(self.config)
        groups = {row["group"]: row["parameters"] for row in result["groups"]}

        self.assertEqual(result["head_dim"], 128)
        self.assertEqual(result["queries_per_kv_head"], 7)
        self.assertEqual(groups["token_embeddings"], 544_997_376)
        self.assertEqual(groups["decoder_layer_00"], 233_057_792)
        self.assertEqual(groups["decoder_layer_27"], 233_057_792)
        self.assertEqual(groups["final_norm"], 3_584)
        self.assertEqual(groups["lm_head"], 544_997_376)
        self.assertEqual(result["total_parameters"], 7_615_616_512)
        self.assertEqual(result["transformer_body_parameters"], 6_525_621_760)
        self.assertEqual(len(result["groups"]), 31)
        self.assertAlmostEqual(
            sum(row["percent"] for row in result["groups"]), 100.0, places=9
        )

    def test_rejects_incompatible_attention_dimensions(self):
        invalid = dict(self.config)
        invalid["hidden_size"] = 3585
        with self.assertRaisesRegex(ValueError, "hidden_size.*num_attention_heads"):
            self.module.analyze_config(invalid)

        invalid = dict(self.config)
        invalid["tie_word_embeddings"] = "false"
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings.*布尔值"):
            self.module.analyze_config(invalid)

    def test_csv_and_cli_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "counts.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--config",
                    str(CONFIG_PATH),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("total_parameters=7615616512", completed.stdout)
            self.assertIn("queries_per_kv_head=7", completed.stdout)
            self.assertTrue(output_path.is_file())

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["group"], "token_embeddings")
            self.assertEqual(rows[-1]["group"], "lm_head")
            self.assertEqual(rows[1]["parameters"], "233057792")


if __name__ == "__main__":
    unittest.main()

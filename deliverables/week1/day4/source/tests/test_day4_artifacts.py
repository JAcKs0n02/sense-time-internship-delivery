import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


DAY4_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = DAY4_ROOT / "source" / "scripts"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Day4ArtifactTests(unittest.TestCase):
    def test_fixed_cases_are_complete_and_stable(self):
        builder = load_module("day4_builder", "build_notebook.py")
        cases = builder.build_cases()
        expected_ids = [
            "pure_chinese",
            "pure_english",
            "mixed_language_version",
            "emoji",
            "rare_cjk",
            "unicode_composition",
            "fullwidth_halfwidth",
            "whitespace_controls",
            "python_code",
            "json_url_escape",
            "latex_math",
            "literal_special_tokens",
            "repeated_hanzi",
            "zero_width",
            "long_chinese",
        ]
        self.assertEqual([case["id"] for case in cases], expected_ids)
        self.assertEqual(len(cases), 15)
        self.assertTrue(all(case["text"] for case in cases))

    def test_notebook_contains_all_required_experiments(self):
        builder = load_module("day4_builder_for_notebook", "build_notebook.py")
        with tempfile.TemporaryDirectory() as tmp:
            notebook_path = Path(tmp) / "tokenizer_experiments.ipynb"
            builder.write_notebook(notebook_path)
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertEqual(notebook["nbformat"], 4)
        for required in (
            "15 个固定极端用例",
            "特殊令牌实验",
            "add_generation_prompt",
            "max_length=64",
            "ByteLevel BPE",
            "SentencePiece Unigram",
            "tokenizer_extreme_cases.csv",
            "special_token_results.json",
            "truncation_results.json",
            "tokenizer_comparison.csv",
        ):
            self.assertIn(required, source)

    def test_every_generated_code_cell_compiles(self):
        builder = load_module("day4_builder_for_syntax", "build_notebook.py")
        notebook = builder.build_notebook()
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "llm_exp")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook_cell_{index}", "exec")

    def test_validator_accepts_complete_synthetic_artifacts(self):
        validator = load_module("day4_validator", "validate_day4.py")
        builder = load_module("day4_builder_for_validation", "build_notebook.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook_path = root / "tokenizer_experiments.executed.ipynb"
            builder.write_notebook(notebook_path)
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            for index, cell in enumerate(code_cells, start=1):
                cell["execution_count"] = index
            notebook_path.write_text(
                json.dumps(notebook, ensure_ascii=False), encoding="utf-8"
            )

            results = root / "results"
            results.mkdir()
            (results / "tokenizer_extreme_cases.csv").write_text(
                "case_id,response\n"
                + "\n".join(f'{case["id"]},ok' for case in builder.build_cases()),
                encoding="utf-8",
            )
            (results / "special_token_results.json").write_text(
                json.dumps(
                    {
                        "tokens": {
                            "<|endoftext|>": {},
                            "<|im_start|>": {},
                            "<|im_end|>": {},
                        },
                        "chat_template": {
                            "add_generation_prompt_true": {},
                            "add_generation_prompt_false": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (results / "truncation_results.json").write_text(
                json.dumps({"max_length": 64, "before_tokens": 100, "after_tokens": 64}),
                encoding="utf-8",
            )
            (results / "tokenizer_comparison.csv").write_text(
                "tokenizer,sample_id,target_vocab_size,actual_vocab_size\n"
                "ByteLevel BPE,zh,800,800\nSentencePiece Unigram,zh,800,800\n",
                encoding="utf-8",
            )
            (results / "day4_environment.txt").write_text(
                "Python=3.10\ntransformers=1\ntokenizers=1\nsentencepiece=1\npandas=1\n",
                encoding="utf-8",
            )
            report = validator.validate(root, notebook_path)
        self.assertTrue(report["ok"], report)
        self.assertIn(
            "both trained tokenizers have actual vocab size 800", report["checks"]
        )


if __name__ == "__main__":
    unittest.main()

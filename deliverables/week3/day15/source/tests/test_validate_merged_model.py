import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_merged_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_merged_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_success_payload_records_independent_model_and_tokenizer_classes():
    module = load_module()

    payload = module.success_payload(
        model_path=Path("/models/merged"),
        model_class="Qwen2ForCausalLM",
        tokenizer_class="Qwen2TokenizerFast",
    )

    assert payload == {
        "status": "PASS",
        "merged_model_load": "PASS",
        "model_path": "/models/merged",
        "model_class": "Qwen2ForCausalLM",
        "tokenizer_class": "Qwen2TokenizerFast",
        "local_files_only": True,
    }

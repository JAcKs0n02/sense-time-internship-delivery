import importlib.util
from pathlib import Path


DAY8_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = DAY8_ROOT / "source" / "scripts" / "preview_training_sample.py"


def load_module():
    spec = importlib.util.spec_from_file_location("preview_training_sample", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_messages_matches_llamafactory_alpaca_joining() -> None:
    module = load_module()
    messages = module.build_messages(
        {
            "instruction": "Explain LoRA.",
            "input": "Use two sentences.",
            "output": "LoRA trains low-rank adapters.",
        }
    )

    assert messages == [
        {
            "role": "user",
            "content": "Explain LoRA.\nUse two sentences.",
        },
        {
            "role": "assistant",
            "content": "LoRA trains low-rank adapters.",
        },
    ]


def test_build_sft_labels_masks_prompt_only() -> None:
    module = load_module()
    labels = module.build_sft_labels([10, 11, 12], [20, 21])

    assert labels == [-100, -100, -100, 20, 21]

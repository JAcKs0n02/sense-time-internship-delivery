from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_vlm_smoke.py"


def load_module():
    assert SCRIPT.is_file(), f"missing VLM smoke script: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("run_vlm_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCuda:
    def __init__(self, bf16_supported: bool):
        self._bf16_supported = bf16_supported

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"

    def __init__(self, bf16_supported: bool):
        self.cuda = FakeCuda(bf16_supported)


def test_smoke_message_uses_an_absolute_local_file_uri(tmp_path: Path):
    module = load_module()
    image = tmp_path / "scene image.jpg"
    image.write_bytes(b"fixture")

    messages = module.build_messages(image, "Describe only what is visible.")

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.resolve().as_uri()},
                {"type": "text", "text": "Describe only what is visible."},
            ],
        }
    ]


def test_dtype_selection_prefers_bf16_and_falls_back_to_fp16():
    module = load_module()
    assert module.choose_dtype(FakeTorch(True)) == ("bf16", "bfloat16")
    assert module.choose_dtype(FakeTorch(False)) == ("fp16", "float16")


def test_default_visual_budget_fits_a_24gb_smoke_test(monkeypatch):
    """The 7B BF16 smoke test must leave headroom for visual activations."""
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--model-dir",
            "/tmp/model",
            "--image",
            "/tmp/image.jpg",
            "--repository",
            "Qwen/Qwen2-VL-7B-Instruct",
            "--revision",
            "e" * 40,
            "--output",
            "/tmp/result.json",
        ],
    )

    args = module.parse_args()

    assert args.min_pixels == 256 * 28 * 28
    assert args.max_pixels == 384 * 28 * 28
    assert args.max_new_tokens == 32


def test_processor_size_maps_pixel_budget_to_transformers_450_contract():
    module = load_module()

    assert module.build_processor_size(200704, 301056) == {
        "shortest_edge": 200704,
        "longest_edge": 301056,
    }

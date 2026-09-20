from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


DAY24_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = DAY24_ROOT / "source" / "scripts" / f"{name}.py"
    if not path.is_file():
        raise AssertionError(f"implementation missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_layer_20_maps_to_zero_based_layer_19() -> None:
    attention = load_script("extract_cross_modal_attention")

    assert attention.resolve_layer_index(20, 28) == 19
    with pytest.raises(ValueError, match="1-based"):
        attention.resolve_layer_index(0, 28)
    with pytest.raises(ValueError, match="exceeds"):
        attention.resolve_layer_index(29, 28)


def test_target_character_occurrence_maps_to_overlapping_token_span() -> None:
    attention = load_script("extract_cross_modal_attention")
    text = "first gamma and second gamma"
    offsets = [(0, 5), (5, 11), (11, 15), (15, 22), (22, 28)]

    assert attention.locate_token_span_from_offsets(
        text=text,
        target_text="gamma",
        occurrence=2,
        offsets=offsets,
    ) == (4, 5)
    with pytest.raises(ValueError, match="occurrence"):
        attention.locate_token_span_from_offsets(
            text=text,
            target_text="gamma",
            occurrence=3,
            offsets=offsets,
        )


def test_prediction_queries_are_one_position_before_each_target_token() -> None:
    attention = load_script("extract_cross_modal_attention")

    assert attention.prediction_query_positions(target_start=12, target_end=15) == [11, 12, 13]
    with pytest.raises(ValueError, match="preceding"):
        attention.prediction_query_positions(target_start=0, target_end=1)


def test_single_image_visual_tokens_must_be_contiguous() -> None:
    attention = load_script("extract_cross_modal_attention")

    assert attention.locate_visual_token_span(
        input_ids=[10, 20, 99, 99, 99, 21, 30], image_token_id=99
    ) == (2, 5)
    with pytest.raises(ValueError, match="contiguous"):
        attention.locate_visual_token_span(
            input_ids=[10, 99, 20, 99, 30], image_token_id=99
        )


@pytest.mark.parametrize(
    ("grid_thw", "merge_size", "expected"),
    [
        ([1, 32, 48], 2, (16, 24, 384)),
        ([1, 14, 84], 2, (7, 42, 294)),
        ([1, 22, 50], 2, (11, 25, 275)),
        ([1, 38, 38], 2, (19, 19, 361)),
    ],
)
def test_merged_grid_matches_frozen_day23_cases(
    grid_thw: list[int], merge_size: int, expected: tuple[int, int, int]
) -> None:
    attention = load_script("extract_cross_modal_attention")

    assert attention.merged_image_grid(grid_thw, merge_size) == expected


def test_attention_reduction_uses_previous_queries_and_visual_keys_only() -> None:
    attention = load_script("extract_cross_modal_attention")
    weights = np.zeros((2, 6, 8), dtype=np.float32)
    # Target tokens occupy positions 4 and 5, so prediction queries are rows 3 and 4.
    weights[0, 3, 1:4] = [0.1, 0.2, 0.3]
    weights[0, 4, 1:4] = [0.3, 0.4, 0.5]
    weights[1, 3, 1:4] = [0.2, 0.4, 0.6]
    weights[1, 4, 1:4] = [0.4, 0.6, 0.8]

    reduced = attention.reduce_cross_modal_attention(
        weights=weights,
        query_positions=[3, 4],
        visual_start=1,
        visual_end=4,
    )

    np.testing.assert_allclose(
        reduced["token_heads"],
        np.array(
            [
                [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
                [[0.3, 0.4, 0.5], [0.4, 0.6, 0.8]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        reduced["aggregated_heads"],
        np.array([[0.2, 0.3, 0.4], [0.3, 0.5, 0.7]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        reduced["head_mean"], np.array([0.25, 0.4, 0.55], dtype=np.float32)
    )


def test_attention_rows_must_be_finite_nonnegative_and_sum_to_one() -> None:
    attention = load_script("extract_cross_modal_attention")
    weights = np.array(
        [
            [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]],
            [[0.4, 0.4, 0.2], [0.25, 0.25, 0.5]],
        ],
        dtype=np.float32,
    )

    summary = attention.validate_attention_rows(weights=weights, query_positions=[0, 1])
    assert summary["row_sum_min"] == pytest.approx(1.0)
    assert summary["row_sum_max"] == pytest.approx(1.0)

    broken = weights.copy()
    broken[0, 0] = [0.2, 0.2, 0.2]
    with pytest.raises(ValueError, match="sum to 1"):
        attention.validate_attention_rows(weights=broken, query_positions=[0])


def test_selective_hook_forces_and_captures_only_the_selected_attention_output() -> None:
    attention = load_script("extract_cross_modal_attention")

    class FakeHandle:
        def __init__(self) -> None:
            self.removed = False

        def remove(self) -> None:
            self.removed = True

    class FakeModule:
        def register_forward_pre_hook(self, hook, *, with_kwargs):
            assert with_kwargs is True
            self.pre_hook = hook
            self.pre_handle = FakeHandle()
            return self.pre_handle

        def register_forward_hook(self, hook):
            self.post_hook = hook
            self.post_handle = FakeHandle()
            return self.post_handle

    module = FakeModule()
    capture = attention.install_attention_capture(module)
    _, changed_kwargs = module.pre_hook(module, (), {"output_attentions": False})
    module.post_hook(
        module,
        (),
        ("attention_output", np.ones((1, 2, 3, 3), dtype=np.float32), None),
    )

    assert changed_kwargs["output_attentions"] is True
    assert capture["weights"].shape == (1, 2, 3, 3)
    attention.remove_hook_handles(capture["handles"])
    assert module.pre_handle.removed is True
    assert module.post_handle.removed is True


def test_teacher_forced_sequence_includes_target_but_uses_preceding_prediction_rows() -> None:
    attention = load_script("extract_cross_modal_attention")

    result = attention.build_teacher_forced_sequence(
        prompt_ids=[10, 11, 12],
        output_ids=[20, 21, 22, 23],
        target_output_start=1,
        target_output_end=3,
    )

    assert result == {
        "full_input_ids": [10, 11, 12, 20, 21, 22],
        "target_absolute_start": 4,
        "target_absolute_end": 6,
        "prediction_query_positions": [3, 4],
    }


def test_frozen_day23_response_is_hash_verified_and_used_for_attention() -> None:
    attention = load_script("extract_cross_modal_attention")

    class FakeTokenizer:
        all_special_ids: list[int] = []

        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            assert text == "alpha gamma"
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return {
                "input_ids": [101, 202],
                "offset_mapping": [(0, 5), (6, 11)],
            }

        def decode(self, token_ids, *, clean_up_tokenization_spaces):
            assert clean_up_tokenization_spaces is False
            return {101: "alpha", 202: "gamma"}[token_ids[0]]

    response = "alpha gamma"
    case = {
        "case_id": "formula",
        "expected_day23_response": response,
        "expected_day23_response_sha256": attention.sha256_text(response),
        "target_text": "gamma",
        "target_occurrence": 1,
    }

    prepared = attention.prepare_frozen_day23_response(FakeTokenizer(), case)

    assert prepared["response"] == response
    assert prepared["response_source"] == "day23_frozen_generation"
    assert prepared["token_ids"] == [101, 202]
    assert prepared["target_start"] == 1
    assert prepared["target_end"] == 2
    assert prepared["target_token_ids"] == [202]

    case["expected_day23_response_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        attention.prepare_frozen_day23_response(FakeTokenizer(), case)

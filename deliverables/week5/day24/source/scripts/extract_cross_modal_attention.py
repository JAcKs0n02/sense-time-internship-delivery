#!/usr/bin/env python3
"""Extract target-token-to-image attention from the frozen Qwen2-VL model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_processor_size(min_pixels: int, max_pixels: int) -> dict[str, int]:
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("pixel budget must satisfy 0 < min_pixels <= max_pixels")
    return {"shortest_edge": min_pixels, "longest_edge": max_pixels}


def resolve_layer_index(layer_number_1_based: int, num_layers: int) -> int:
    """Convert the teacher-facing layer number to a Python list index."""
    if layer_number_1_based < 1:
        raise ValueError("layer number must be 1-based and positive")
    if layer_number_1_based > num_layers:
        raise ValueError(
            f"layer number {layer_number_1_based} exceeds model layer count {num_layers}"
        )
    return layer_number_1_based - 1


def locate_token_span_from_offsets(
    *,
    text: str,
    target_text: str,
    occurrence: int,
    offsets: Sequence[Sequence[int]],
) -> tuple[int, int]:
    """Return the token interval overlapping a configured target occurrence."""
    if not target_text:
        raise ValueError("target_text must be non-empty")
    if occurrence < 1:
        raise ValueError("occurrence must be 1-based and positive")

    character_starts: list[int] = []
    cursor = 0
    while True:
        cursor = text.find(target_text, cursor)
        if cursor < 0:
            break
        character_starts.append(cursor)
        cursor += max(1, len(target_text))
    if len(character_starts) < occurrence:
        raise ValueError(
            f"target occurrence {occurrence} not found; available={len(character_starts)}"
        )

    char_start = character_starts[occurrence - 1]
    char_end = char_start + len(target_text)
    overlapping = [
        index
        for index, pair in enumerate(offsets)
        if len(pair) == 2 and int(pair[1]) > char_start and int(pair[0]) < char_end
    ]
    if not overlapping:
        raise ValueError("target occurrence did not overlap any tokenizer offsets")
    if overlapping != list(range(overlapping[0], overlapping[-1] + 1)):
        raise ValueError("target token offsets are not contiguous")
    return overlapping[0], overlapping[-1] + 1


def prediction_query_positions(*, target_start: int, target_end: int) -> list[int]:
    """Map target token positions to the preceding causal prediction rows."""
    if target_start < 1:
        raise ValueError("the first target token must have a preceding query position")
    if target_end <= target_start:
        raise ValueError("target_end must be greater than target_start")
    return list(range(target_start - 1, target_end - 1))


def build_teacher_forced_sequence(
    *,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
    target_output_start: int,
    target_output_end: int,
) -> dict[str, Any]:
    """Build a causal prefix through the target and identify prediction rows."""
    if target_output_start < 0 or target_output_end <= target_output_start:
        raise ValueError("invalid target output token interval")
    if target_output_end > len(output_ids):
        raise ValueError("target token interval exceeds generated output")
    prompt = [int(value) for value in prompt_ids]
    prefix = [int(value) for value in output_ids[:target_output_end]]
    absolute_start = len(prompt) + target_output_start
    absolute_end = len(prompt) + target_output_end
    return {
        "full_input_ids": prompt + prefix,
        "target_absolute_start": absolute_start,
        "target_absolute_end": absolute_end,
        "prediction_query_positions": prediction_query_positions(
            target_start=absolute_start,
            target_end=absolute_end,
        ),
    }


def locate_visual_token_span(
    *, input_ids: Sequence[int], image_token_id: int
) -> tuple[int, int]:
    """Locate one contiguous image-token interval in a single-image prompt."""
    positions = [index for index, token_id in enumerate(input_ids) if int(token_id) == image_token_id]
    if not positions:
        raise ValueError("no image tokens found in input_ids")
    expected = list(range(positions[0], positions[-1] + 1))
    if positions != expected:
        raise ValueError("image tokens are not contiguous; multi-image input is unsupported")
    return positions[0], positions[-1] + 1


def merged_image_grid(
    grid_thw: Sequence[int], spatial_merge_size: int
) -> tuple[int, int, int]:
    """Return merged height, width, and visual-token count for one image."""
    if len(grid_thw) != 3:
        raise ValueError("image_grid_thw must contain exactly t, h, w")
    t, h, w = (int(value) for value in grid_thw)
    if t != 1:
        raise ValueError("Day24 formal cases support single images with temporal grid 1")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")
    if h % spatial_merge_size or w % spatial_merge_size:
        raise ValueError("image grid is not divisible by spatial_merge_size")
    merged_h = h // spatial_merge_size
    merged_w = w // spatial_merge_size
    return merged_h, merged_w, t * merged_h * merged_w


def reduce_cross_modal_attention(
    *,
    weights: Any,
    query_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
) -> dict[str, np.ndarray]:
    """Slice target prediction rows to image keys and aggregate tokens/heads."""
    array = np.asarray(weights, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("attention weights must have shape [heads, query, key]")
    if not query_positions:
        raise ValueError("at least one query position is required")
    if visual_start < 0 or visual_end <= visual_start or visual_end > array.shape[2]:
        raise ValueError("visual token range is outside the attention key dimension")
    if min(query_positions) < 0 or max(query_positions) >= array.shape[1]:
        raise ValueError("query position is outside the attention query dimension")

    token_heads = np.stack(
        [array[:, int(position), visual_start:visual_end] for position in query_positions],
        axis=0,
    ).astype(np.float32, copy=False)
    if not np.isfinite(token_heads).all():
        raise ValueError("attention slice contains non-finite values")
    if (token_heads < 0).any():
        raise ValueError("attention slice contains negative values")
    aggregated_heads = token_heads.mean(axis=0, dtype=np.float32)
    head_mean = aggregated_heads.mean(axis=0, dtype=np.float32)
    return {
        "token_heads": token_heads,
        "aggregated_heads": aggregated_heads,
        "head_mean": head_mean,
    }


def validate_attention_rows(
    *, weights: Any, query_positions: Sequence[int], atol: float = 5e-3
) -> dict[str, float]:
    """Validate complete post-softmax attention rows before visual slicing."""
    array = np.asarray(weights, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("attention weights must have shape [heads, query, key]")
    if not query_positions:
        raise ValueError("at least one query position is required")
    selected = array[:, [int(value) for value in query_positions], :]
    if not np.isfinite(selected).all():
        raise ValueError("attention rows contain non-finite values")
    if (selected < 0).any():
        raise ValueError("attention rows contain negative values")
    row_sums = selected.sum(axis=-1, dtype=np.float32)
    if not np.allclose(row_sums, 1.0, atol=atol, rtol=0.0):
        raise ValueError(
            f"attention rows do not sum to 1 within atol={atol}: "
            f"min={float(row_sums.min())}, max={float(row_sums.max())}"
        )
    return {
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
    }


def _attention_to_numpy(value: Any) -> np.ndarray:
    tensor = value
    for method_name in ("detach", "float", "cpu"):
        method = getattr(tensor, method_name, None)
        if callable(method):
            tensor = method()
    numpy_method = getattr(tensor, "numpy", None)
    if callable(numpy_method):
        tensor = numpy_method()
    return np.asarray(tensor, dtype=np.float32)


def install_attention_capture(attention_module: Any) -> dict[str, Any]:
    """Force attention output only on one module and capture its returned weights."""
    capture: dict[str, Any] = {"weights": None, "handles": []}

    def force_output_attentions(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        updated = dict(kwargs)
        updated["output_attentions"] = True
        return args, updated

    def capture_output(module: Any, args: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError("selected attention module returned no attention weights")
        capture["weights"] = _attention_to_numpy(output[1])

    capture["handles"] = [
        attention_module.register_forward_pre_hook(force_output_attentions, with_kwargs=True),
        attention_module.register_forward_hook(capture_output),
    ]
    return capture


def remove_hook_handles(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _validate_static_inputs(
    *,
    model_dir: Path,
    data_root: Path,
    cases_path: Path,
    generation_config_path: Path,
    cases_payload: dict[str, Any],
    generation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise ValueError("model directory is incomplete")
    if cases_payload.get("repository") != generation_config.get("repository"):
        raise ValueError("case/model repository mismatch")
    if cases_payload.get("revision") != generation_config.get("revision"):
        raise ValueError("case/model revision mismatch")
    if cases_payload.get("generation_config_sha256") != sha256_file(generation_config_path):
        raise ValueError("generation config hash changed after case preregistration")
    if generation_config.get("do_sample") is not False or generation_config.get("seed") != 42:
        raise ValueError("Day24 requires the frozen deterministic Day23 generation settings")
    expected_size = build_processor_size(
        int(generation_config["min_pixels"]), int(generation_config["max_pixels"])
    )
    if generation_config.get("processor_size") != expected_size:
        raise ValueError("generation processor_size does not match the pixel budget")
    cases = cases_payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 3:
        raise ValueError("Day24 requires at least three preregistered cases")
    if len({case.get("case_id") for case in cases}) != len(cases):
        raise ValueError("Day24 case_id values must be unique")
    for case in cases:
        image_path = data_root / str(case["image_relative_path"])
        if not image_path.is_file() or sha256_file(image_path) != case.get("image_sha256"):
            raise ValueError(f"image integrity check failed: {case.get('image_id')}")
        response = str(case.get("expected_day23_response", ""))
        if sha256_text(response) != case.get("expected_day23_response_sha256"):
            raise ValueError(f"Day23 response hash mismatch: {case.get('case_id')}")
    if not cases_path.is_file():
        raise ValueError("attention case configuration is missing")
    return cases


def _target_encoding(tokenizer: Any, response: str, case: dict[str, Any]) -> dict[str, Any]:
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [[int(value) for value in pair] for pair in encoded["offset_mapping"]]
    target_start, target_end = locate_token_span_from_offsets(
        text=response,
        target_text=str(case["target_text"]),
        occurrence=int(case["target_occurrence"]),
        offsets=offsets,
    )
    return {
        "token_ids": token_ids,
        "offsets": offsets,
        "target_start": target_start,
        "target_end": target_end,
        "target_token_ids": token_ids[target_start:target_end],
        "target_token_texts": [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in token_ids[target_start:target_end]
        ],
    }


def prepare_frozen_day23_response(tokenizer: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Tokenize the exact Day23 generation selected before attention extraction.

    Qwen2-VL must use eager attention to expose attention weights. Re-running
    ``generate`` under that implementation can produce a different response from
    the already archived Day23 output, so Day24 teacher-forces the hash-verified
    Day23 text instead of silently analysing a newly generated answer.
    """
    response = str(case.get("expected_day23_response", ""))
    expected_hash = str(case.get("expected_day23_response_sha256", ""))
    if not response:
        raise ValueError(f"frozen Day23 response is empty: {case.get('case_id')}")
    if sha256_text(response) != expected_hash:
        raise ValueError(f"frozen Day23 response hash mismatch: {case.get('case_id')}")
    return {
        "response": response,
        "response_source": "day23_frozen_generation",
        **_target_encoding(tokenizer, response, case),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases_payload = read_json(args.cases)
    generation_config = read_json(args.generation_config)
    cases = _validate_static_inputs(
        model_dir=args.model_dir,
        data_root=args.data_root,
        cases_path=args.cases,
        generation_config_path=args.generation_config,
        cases_payload=cases_payload,
        generation_config=generation_config,
    )
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
        if len(cases) != 1:
            raise ValueError(f"case_id not found: {args.case_id}")

    metadata_path = args.results_root / "attention_metadata.jsonl"
    summary_path = args.results_root / "run_summary.json"
    if metadata_path.exists() or summary_path.exists():
        if not args.overwrite:
            raise ValueError(f"results already exist: {args.results_root}")
        metadata_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the frozen bfloat16 configuration is unsupported by this GPU")
    torch.manual_seed(int(generation_config["seed"]))
    torch.cuda.manual_seed_all(int(generation_config["seed"]))
    torch.cuda.empty_cache()

    processor_size = build_processor_size(
        int(generation_config["min_pixels"]), int(generation_config["max_pixels"])
    )
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        size=processor_size,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = round(time.perf_counter() - load_started, 3)

    layer_count = len(model.model.layers)
    layer_number = int(cases_payload["layer_number_1_based"])
    layer_index = resolve_layer_index(layer_number, layer_count)
    attention_module = model.model.layers[layer_index].self_attn
    module_path = f"model.model.layers.{layer_index}.self_attn"
    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
    effective_processor_size = dict(processor.image_processor.size)
    if effective_processor_size != processor_size:
        raise RuntimeError(
            f"effective processor size changed: {effective_processor_size} != {processor_size}"
        )

    pass_count = 0
    fail_count = 0
    for case in cases:
        case_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        base_metadata = {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "case_id": case["case_id"],
            "record_id": case["record_id"],
            "image_id": case["image_id"],
            "image_relative_path": case["image_relative_path"],
            "image_sha256": case["image_sha256"],
            "prompt": case["prompt"],
            "target_text": case["target_text"],
            "target_occurrence": case["target_occurrence"],
            "target_display_label": case["target_display_label"],
            "target_category": case["target_category"],
            "repository": generation_config["repository"],
            "model_revision": generation_config["revision"],
            "dtype": generation_config["dtype"],
            "attn_implementation": "eager",
            "layer_number_1_based": layer_number,
            "layer_index_0_based": layer_index,
            "attention_module_path": module_path,
            "attention_module_class": type(attention_module).__name__,
            "layer_count": layer_count,
            "spatial_merge_size": spatial_merge_size,
            "processor_size": processor_size,
            "generation_config_sha256": sha256_file(args.generation_config),
            "attention_cases_sha256": sha256_file(args.cases),
        }
        try:
            image_path = args.data_root / str(case["image_relative_path"])
            messages = build_messages(image_path, str(case["prompt"]))
            rendered_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[rendered_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            prompt_ids = [int(value) for value in inputs.input_ids[0].tolist()]
            visual_start, visual_end = locate_visual_token_span(
                input_ids=prompt_ids,
                image_token_id=int(model.config.image_token_id),
            )
            grid_thw = [int(value) for value in inputs.image_grid_thw[0].tolist()]
            merged_h, merged_w, visual_count = merged_image_grid(
                grid_thw, spatial_merge_size
            )
            if visual_end - visual_start != visual_count:
                raise RuntimeError(
                    "visual token interval does not match the merger grid: "
                    f"interval={visual_end - visual_start}, grid={visual_count}"
                )

            target = prepare_frozen_day23_response(processor.tokenizer, case)
            response = str(target["response"])
            semantic_output_ids = [int(value) for value in target["token_ids"]]
            forced = build_teacher_forced_sequence(
                prompt_ids=prompt_ids,
                output_ids=semantic_output_ids,
                target_output_start=int(target["target_start"]),
                target_output_end=int(target["target_end"]),
            )
            full_input_ids = torch.tensor(
                [forced["full_input_ids"]], dtype=inputs.input_ids.dtype, device="cuda"
            )
            full_attention_mask = torch.ones_like(full_input_ids)

            capture = install_attention_capture(attention_module)
            try:
                capture["weights"] = None
                model.rope_deltas = None
                with torch.inference_mode():
                    model(
                        input_ids=full_input_ids,
                        attention_mask=full_attention_mask,
                        pixel_values=inputs.pixel_values,
                        image_grid_thw=inputs.image_grid_thw,
                        use_cache=False,
                        output_attentions=False,
                        return_dict=True,
                    )
            finally:
                remove_hook_handles(capture["handles"])
            if capture["weights"] is None:
                raise RuntimeError("attention Hook captured no weights")
            captured = np.asarray(capture["weights"], dtype=np.float32)
            if captured.ndim != 4 or captured.shape[0] != 1:
                raise RuntimeError(
                    f"unexpected attention shape; expected [1, heads, query, key], got {captured.shape}"
                )
            weights = captured[0]
            row_summary = validate_attention_rows(
                weights=weights,
                query_positions=forced["prediction_query_positions"],
            )
            reduced = reduce_cross_modal_attention(
                weights=weights,
                query_positions=forced["prediction_query_positions"],
                visual_start=visual_start,
                visual_end=visual_end,
            )
            token_heads = reduced["token_heads"].reshape(
                len(forced["prediction_query_positions"]), weights.shape[0], merged_h, merged_w
            )
            aggregated_heads = reduced["aggregated_heads"].reshape(
                weights.shape[0], merged_h, merged_w
            )
            head_mean = reduced["head_mean"].reshape(merged_h, merged_w)
            visual_masses = reduced["token_heads"].sum(axis=-1, dtype=np.float32)

            raw_dir = args.results_root / "raw_attention"
            raw_dir.mkdir(parents=True, exist_ok=True)
            token_heads_path = raw_dir / f"{case['case_id']}-token-heads.npy"
            aggregated_heads_path = raw_dir / f"{case['case_id']}-heads.npy"
            head_mean_path = raw_dir / f"{case['case_id']}-mean.npy"
            np.save(token_heads_path, token_heads)
            np.save(aggregated_heads_path, aggregated_heads)
            np.save(head_mean_path, head_mean)

            payload = {
                **base_metadata,
                "status": "PASS",
                "expected_day23_response_sha256": case[
                    "expected_day23_response_sha256"
                ],
                "response_source": target["response_source"],
                "source_generation_replayed": False,
                "actual_response": response,
                "actual_response_sha256": sha256_text(response),
                "response_matches_day23": True,
                "input_tokens": len(prompt_ids),
                "recorded_response_tokens": len(semantic_output_ids),
                "target_output_token_start": target["target_start"],
                "target_output_token_end": target["target_end"],
                "target_token_ids": target["target_token_ids"],
                "target_token_texts": target["target_token_texts"],
                "target_absolute_start": forced["target_absolute_start"],
                "target_absolute_end": forced["target_absolute_end"],
                "prediction_query_positions": forced["prediction_query_positions"],
                "visual_token_start": visual_start,
                "visual_token_end": visual_end,
                "visual_token_count": visual_count,
                "image_grid_thw": grid_thw,
                "merged_grid_hw": [merged_h, merged_w],
                "captured_attention_shape": list(captured.shape),
                "token_head_shape": list(token_heads.shape),
                "aggregated_head_shape": list(aggregated_heads.shape),
                "head_mean_shape": list(head_mean.shape),
                "head_aggregation": "arithmetic_mean",
                "target_token_aggregation": "arithmetic_mean",
                "attention_row_sum_min": row_summary["row_sum_min"],
                "attention_row_sum_max": row_summary["row_sum_max"],
                "visual_attention_mass_min": float(visual_masses.min()),
                "visual_attention_mass_max": float(visual_masses.max()),
                "visual_attention_mass_mean": float(visual_masses.mean()),
                "array_paths": {
                    "token_heads": str(token_heads_path.relative_to(args.results_root)),
                    "aggregated_heads": str(aggregated_heads_path.relative_to(args.results_root)),
                    "head_mean": str(head_mean_path.relative_to(args.results_root)),
                },
                "array_sha256": {
                    "token_heads": sha256_file(token_heads_path),
                    "aggregated_heads": sha256_file(aggregated_heads_path),
                    "head_mean": sha256_file(head_mean_path),
                },
                "latency_seconds": round(time.perf_counter() - case_started, 3),
                "peak_gpu_memory_mib": round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024, 3
                ),
                "completed_at": utc_now(),
                "error_type": "",
                "error": "",
            }
            pass_count += 1
        except Exception as exc:
            payload = {
                **base_metadata,
                "status": "FAIL",
                "latency_seconds": round(time.perf_counter() - case_started, 3),
                "peak_gpu_memory_mib": round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024, 3
                ),
                "completed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            fail_count += 1
        append_jsonl(metadata_path, payload)
        print(
            f"[{case['case_id']}] {payload['status']} "
            f"target={case['target_display_label']} latency={payload['latency_seconds']}s",
            flush=True,
        )
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "1.0",
        "status": "PASS" if fail_count == 0 and pass_count == len(cases) else "FAIL",
        "run_id": args.run_id,
        "completed_at": utc_now(),
        "repository": generation_config["repository"],
        "model_revision": generation_config["revision"],
        "local_files_only": True,
        "dtype": generation_config["dtype"],
        "attn_implementation": "eager",
        "layer_number_1_based": layer_number,
        "layer_index_0_based": layer_index,
        "attention_module_path": module_path,
        "attention_module_class": type(attention_module).__name__,
        "layer_count": layer_count,
        "spatial_merge_size": spatial_merge_size,
        "attention_extraction_mode": "teacher_forced_frozen_day23_response",
        "source_generation_replayed": False,
        "case_count": len(cases),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "load_seconds": load_seconds,
        "cuda_memory_after_load_mib": round(torch.cuda.memory_allocated() / 1024 / 1024, 3),
        "metadata_sha256": sha256_file(metadata_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

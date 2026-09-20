#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable


DEFAULT_GENERATION_CONFIG: dict[str, Any] = {
    "do_sample": False,
    "max_new_tokens": 512,
    "repetition_penalty": 1.0,
    "seed": 42,
}

REQUIRED_QUESTION_FIELDS = {
    "id",
    "category",
    "difficulty",
    "messages",
    "reference",
    "automatic_check",
    "human_scoring_notes",
}


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions file must contain a non-empty questions list")
    identifiers: list[str] = []
    categories: list[str] = []
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"question {index} must be an object")
        missing = REQUIRED_QUESTION_FIELDS - set(question)
        if missing:
            raise ValueError(
                f"question {question.get('id', index)} missing fields: {sorted(missing)}"
            )
        question_id = question["id"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"question {index} has invalid id")
        messages = question["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"question {question_id} has invalid messages")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError(f"question {question_id} contains an invalid message")
        identifiers.append(question_id)
        categories.append(str(question["category"]))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("question ids must be unique")
    if len(questions) == 20:
        expected = {"math": 7, "reasoning": 7, "code": 6}
        actual = {category: categories.count(category) for category in set(categories)}
        if actual != expected:
            raise ValueError(f"invalid Day 14 category distribution: {actual}")
    return questions


def load_evaluation_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("formal evaluation specification must be an object")
    questions = load_questions(path)
    generation = payload.get("generation_config")
    if generation != DEFAULT_GENERATION_CONFIG:
        raise ValueError("question file generation_config differs from the frozen config")
    system_message = payload.get("system_message")
    if not isinstance(system_message, str) or not system_message.strip():
        raise ValueError("formal evaluation specification requires a system_message")
    return {**payload, "questions": questions}


def build_messages(
    system_message: str | None,
    question_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.extend(dict(message) for message in question_messages)
    return messages


def load_rubric(path: Path) -> dict[str, Any]:
    rubric = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "accuracy": 0.30,
        "completeness": 0.25,
        "logic": 0.20,
        "safety": 0.15,
        "format": 0.10,
    }
    if rubric.get("weights") != expected:
        raise ValueError("rubric weights must match the frozen teacher requirements")
    if not math.isclose(sum(expected.values()), 1.0):
        raise ValueError("rubric weights must sum to 1.0")
    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("rubric dimensions must be an object")
    for dimension in expected:
        anchors = dimensions.get(dimension, {}).get("anchors")
        if not isinstance(anchors, dict) or set(anchors) != {str(i) for i in range(6)}:
            raise ValueError(f"rubric dimension {dimension} must define anchors 0 through 5")
    return rubric


def build_candidates(
    manifest_path: Path,
    *,
    week3_root: Path,
    expected_adapter_count: int = 9,
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("adapter manifest must be a list")
    candidates: list[dict[str, Any]] = [
        {"candidate_id": "base", "adapter_path": None, "kind": "base"}
    ]
    seen: set[str] = set()
    for item in payload:
        if item.get("status") != "completed":
            raise ValueError(f"adapter is not completed: {item.get('run_id')}")
        run_id = item.get("run_id")
        adapter_path = item.get("adapter_path")
        if not isinstance(run_id, str) or not isinstance(adapter_path, str):
            raise ValueError("adapter manifest contains an invalid row")
        if run_id in seen:
            raise ValueError(f"duplicate adapter run id: {run_id}")
        seen.add(run_id)
        resolved = Path(adapter_path)
        if not resolved.is_absolute():
            resolved = week3_root / resolved
        candidates.append(
            {
                "candidate_id": run_id,
                "adapter_path": str(resolved),
                "kind": "sft",
            }
        )
    if len(candidates) != expected_adapter_count + 1:
        raise ValueError(
            f"expected base plus {expected_adapter_count} adapters, "
            f"found {len(candidates)} candidates"
        )
    return candidates


def validate_candidate_paths(
    base_model: str,
    candidates: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not Path(base_model).is_dir():
        errors.append(f"missing base model directory: {base_model}")
    for candidate in candidates:
        adapter_path = candidate.get("adapter_path")
        if adapter_path is not None and not Path(adapter_path).is_dir():
            errors.append(
                f"missing adapter directory: {candidate['candidate_id']}: {adapter_path}"
            )
    return errors


def make_result(
    *,
    candidate_id: str,
    question: dict[str, Any],
    raw_response: str | None,
    generation_config: dict[str, Any],
    input_tokens: int | None,
    output_tokens: int | None,
    elapsed_seconds: float,
    status: str = "completed",
    error: str | None = None,
    peak_gpu_memory_bytes: int | None = None,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "question_id": question["id"],
        "question_category": question["category"],
        "question_sha256": canonical_json_sha256(question),
        "messages": [dict(message) for message in (messages or question["messages"])],
        "generation_config": dict(generation_config),
        "raw_response": raw_response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "status": status,
        "error": error,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


class HuggingFaceRunner:
    def __init__(self, base_model: str, adapter_path: str | None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("torch, transformers and bitsandbytes are required") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for formal Day 14 inference")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=True,
            trust_remote_code=True,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise RuntimeError("peft is required for adapter inference") from error
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        self.model = model

    def generate(self, messages: list[dict[str, str]], generation_config: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        seed = int(generation_config["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=int(generation_config["max_new_tokens"]),
                repetition_penalty=float(generation_config["repetition_penalty"]),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = generated[0, input_ids.shape[-1] :]
        return {
            "raw_response": self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip(),
            "input_tokens": int(input_ids.shape[-1]),
            "output_tokens": int(generated_ids.shape[-1]),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def run_evaluation(
    base_model: str,
    adapter_path: str | None,
    questions: list[dict[str, Any]],
    *,
    candidate_id: str | None = None,
    model_factory: Callable[[str, str | None], Any] = HuggingFaceRunner,
    generation_config: dict[str, Any] | None = None,
    system_message: str | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    config = dict(generation_config or DEFAULT_GENERATION_CONFIG)
    if config.get("do_sample") is not False:
        raise ValueError("formal Day 14 evaluation requires do_sample=false")
    resolved_candidate = candidate_id or (Path(adapter_path).name if adapter_path else "base")
    records: list[dict[str, Any]] = []
    try:
        runner = model_factory(base_model, adapter_path)
    except Exception as error:
        for question in questions:
            messages = build_messages(system_message, question["messages"])
            record = make_result(
                candidate_id=resolved_candidate,
                question=question,
                raw_response=None,
                generation_config=config,
                input_tokens=None,
                output_tokens=None,
                elapsed_seconds=0.0,
                status="failed_model_load",
                error=f"{type(error).__name__}: {error}",
                messages=messages,
            )
            record["base_model"] = base_model
            record["adapter_path"] = adapter_path
            records.append(record)
            if on_record is not None:
                on_record(record)
        return records
    try:
        for question in questions:
            started = time.perf_counter()
            messages = build_messages(system_message, question["messages"])
            try:
                generated = runner.generate(messages, config)
                record = make_result(
                    candidate_id=resolved_candidate,
                    question=question,
                    raw_response=generated["raw_response"],
                    generation_config=config,
                    input_tokens=generated["input_tokens"],
                    output_tokens=generated["output_tokens"],
                    elapsed_seconds=time.perf_counter() - started,
                    peak_gpu_memory_bytes=generated.get("peak_gpu_memory_bytes"),
                    messages=messages,
                )
            except Exception as error:  # preserve one terminal record per prompt
                record = make_result(
                    candidate_id=resolved_candidate,
                    question=question,
                    raw_response=None,
                    generation_config=config,
                    input_tokens=None,
                    output_tokens=None,
                    elapsed_seconds=time.perf_counter() - started,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    messages=messages,
                )
            record["base_model"] = base_model
            record["adapter_path"] = adapter_path
            records.append(record)
            if on_record is not None:
                on_record(record)
    finally:
        runner.close()
    return records


def extract_last_number(text: str) -> float | None:
    matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])


def automatic_check(question: dict[str, Any], response: str | None) -> dict[str, Any]:
    check = question["automatic_check"]
    check_type = check["type"]
    if response is None:
        return {"status": "not_evaluable", "passed": False, "detail": "generation failed"}
    if check_type == "numeric":
        observed = extract_last_number(response)
        expected = float(check["expected"])
        tolerance = float(check.get("tolerance", 0))
        passed = observed is not None and abs(observed - expected) <= tolerance
        return {"status": "checked", "passed": passed, "observed": observed, "expected": expected, "tolerance": tolerance}
    if check_type == "required_terms":
        groups = check["term_groups"]
        normalized = response.casefold()
        group_results = [any(term.casefold() in normalized for term in group) for group in groups]
        return {"status": "checked", "passed": all(group_results), "group_results": group_results}
    if check_type == "format":
        required = check.get("required_patterns", [])
        matches = [re.search(pattern, response, re.MULTILINE) is not None for pattern in required]
        return {"status": "checked", "passed": all(matches), "pattern_results": matches}
    if check_type == "python_tests":
        scripts_dir = Path(__file__).resolve().parent / "source" / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from run_code_checks import check_code_response

        return check_code_response(response, check["assertions"])
    return {"status": "unsupported", "passed": None, "detail": f"unknown check type: {check_type}"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Week 3 Day 14 evaluation.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--week3-root", type=Path, default=Path("/root/autodl-tmp/qwen25-week3"))
    parser.add_argument("--candidate", help="Run only one candidate id for smoke testing.")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec = load_evaluation_spec(args.questions)
        questions = spec["questions"]
        candidates = build_candidates(args.adapter_manifest, week3_root=args.week3_root)
        if args.candidate:
            candidates = [item for item in candidates if item["candidate_id"] == args.candidate]
            if not candidates:
                raise ValueError(f"unknown candidate: {args.candidate}")
        path_errors = validate_candidate_paths(args.base_model, candidates)
        if path_errors:
            raise ValueError("; ".join(path_errors))
        manifest = {
            "status": "validated" if args.validate_only else "running",
            "question_file": str(args.questions.resolve()),
            "question_file_sha256": file_sha256(args.questions),
            "question_count": len(questions),
            "generation_config": DEFAULT_GENERATION_CONFIG,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "evaluation_manifest.json", manifest)
        if args.validate_only:
            return 0
        combined: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for candidate in candidates:
            response_path = args.output_dir / candidate["candidate_id"] / "responses.jsonl"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            with response_path.open("x", encoding="utf-8") as response_handle:
                def persist(record: dict[str, Any]) -> None:
                    response_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    response_handle.flush()

                records = run_evaluation(
                    args.base_model,
                    candidate["adapter_path"],
                    questions,
                    candidate_id=candidate["candidate_id"],
                    system_message=spec["system_message"],
                    on_record=persist,
                )
            combined.extend(records)
            for question, record in zip(questions, records, strict=True):
                evidence.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "question_id": question["id"],
                        "automatic_evidence": automatic_check(question, record["raw_response"]),
                    }
                )
        write_jsonl(args.output_dir / "all_responses.jsonl", combined)
        write_json(args.output_dir / "automatic_evidence.json", {"ranking_input": False, "records": evidence})
        manifest.update(
            {
                "status": "completed",
                "terminal_record_count": len(combined),
                "failed_record_count": sum(item["status"] != "completed" for item in combined),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(args.output_dir / "evaluation_manifest.json", manifest)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pure aggregation logic for the BF16/AWQ/GPTQ benchmark."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .acceptance import QuantizationReceipt, memory_reduction, quantization_passes


class BenchmarkContractError(ValueError):
    """Raised when raw benchmark records cannot be compared fairly."""


@dataclass(frozen=True)
class TimingSummary:
    tokens_per_second: float
    latency_median_seconds: float
    latency_p10_seconds: float
    latency_p90_seconds: float
    ttft_median_seconds: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_timings(
    output_tokens: Sequence[int],
    durations: Sequence[float],
    ttfts: Sequence[float],
) -> TimingSummary:
    """Summarize measured generations using total-token throughput."""
    if not output_tokens or not (
        len(output_tokens) == len(durations) == len(ttfts)
    ):
        raise BenchmarkContractError("timing series must have the same non-zero length")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in output_tokens
    ):
        raise BenchmarkContractError("output token counts must be positive integers")
    if any(
        not math.isfinite(value) or value <= 0
        for value in tuple(durations) + tuple(ttfts)
    ):
        raise BenchmarkContractError("timings must be positive finite values")
    return TimingSummary(
        tokens_per_second=sum(output_tokens) / sum(durations),
        latency_median_seconds=_percentile(durations, 0.5),
        latency_p10_seconds=_percentile(durations, 0.1),
        latency_p90_seconds=_percentile(durations, 0.9),
        ttft_median_seconds=_percentile(ttfts, 0.5),
    )


_NUMERIC_FIELDS = (
    "weight_bytes",
    "static_load_mib",
    "peak_mib",
    "tokens_per_second",
    "ttft_median_seconds",
    "nll",
    "perplexity",
)


def _validated_record(record: Mapping[str, object]) -> dict[str, object]:
    model = record.get("model")
    if model not in {"bf16", "awq", "gptq"}:
        raise BenchmarkContractError(f"unsupported model label: {model!r}")
    protocol = record.get("protocol_sha256")
    if not isinstance(protocol, str) or len(protocol) != 64:
        raise BenchmarkContractError("protocol_sha256 must be 64 characters")
    normalized = dict(record)
    for field in _NUMERIC_FIELDS:
        value = record.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise BenchmarkContractError(f"{model}.{field} must be positive finite")
        normalized[field] = value
    for field in ("load_smoke_passed", "generation_smoke_passed"):
        if not isinstance(record.get(field), bool):
            raise BenchmarkContractError(f"{model}.{field} must be boolean")
    return normalized


def aggregate_model_results(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate and compare exactly one BF16, AWQ and GPTQ result."""
    normalized = [_validated_record(record) for record in records]
    labels = [str(record["model"]) for record in normalized]
    if len(normalized) != 3 or set(labels) != {"bf16", "awq", "gptq"}:
        raise BenchmarkContractError("comparison requires exactly bf16, awq and gptq")
    if len(labels) != len(set(labels)):
        raise BenchmarkContractError("comparison requires exactly bf16, awq and gptq")
    protocols = {str(record["protocol_sha256"]) for record in normalized}
    if len(protocols) != 1:
        raise BenchmarkContractError("all records must use one protocol")

    ordered = {str(record["model"]): record for record in normalized}
    baseline = ordered["bf16"]
    output: list[dict[str, object]] = []
    for model in ("bf16", "awq", "gptq"):
        record = dict(ordered[model])
        if model == "bf16":
            record["memory_reduction"] = 0.0
            record["acceptance_passed"] = None
        else:
            reduction = memory_reduction(
                float(baseline["peak_mib"]), float(record["peak_mib"])
            )
            receipt = QuantizationReceipt(
                baseline_peak_mib=float(baseline["peak_mib"]),
                quantized_peak_mib=float(record["peak_mib"]),
                load_smoke_passed=bool(record["load_smoke_passed"]),
                generation_smoke_passed=bool(record["generation_smoke_passed"]),
                perplexity=float(record["perplexity"]),
            )
            record["memory_reduction"] = reduction
            record["acceptance_passed"] = quantization_passes(receipt)
        record["ppl_delta_vs_bf16"] = float(record["perplexity"]) - float(
            baseline["perplexity"]
        )
        record["throughput_ratio_vs_bf16"] = float(
            record["tokens_per_second"]
        ) / float(baseline["tokens_per_second"])
        output.append(record)

    return {
        "schema_version": "1.0",
        "protocol_sha256": protocols.pop(),
        "models": output,
        "at_least_one_quantization_passed": any(
            row["acceptance_passed"] is True for row in output
        ),
    }

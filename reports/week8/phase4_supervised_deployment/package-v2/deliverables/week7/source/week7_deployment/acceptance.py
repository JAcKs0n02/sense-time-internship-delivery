"""Fail-closed acceptance calculations for quantized model receipts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationReceipt:
    baseline_peak_mib: float
    quantized_peak_mib: float
    load_smoke_passed: bool
    generation_smoke_passed: bool
    perplexity: float


def memory_reduction(bf16_peak_mib: float, quantized_peak_mib: float) -> float:
    """Calculate BF16-relative peak-memory reduction under one protocol."""
    values = (bf16_peak_mib, quantized_peak_mib)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("finite positive memory values required")
    return 1.0 - quantized_peak_mib / bf16_peak_mib


def quantization_passes(receipt: QuantizationReceipt) -> bool:
    """Require valid quality, both smokes and at least 30% memory reduction."""
    if not receipt.load_smoke_passed or not receipt.generation_smoke_passed:
        return False
    if not math.isfinite(receipt.perplexity) or receipt.perplexity <= 0:
        return False
    try:
        reduction = memory_reduction(
            receipt.baseline_peak_mib, receipt.quantized_peak_mib
        )
    except ValueError:
        return False
    return reduction >= 0.30

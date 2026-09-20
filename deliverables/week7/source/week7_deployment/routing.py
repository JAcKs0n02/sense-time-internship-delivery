"""Deterministic backend selection for text and single-image requests."""

from __future__ import annotations

from typing import Literal

from .validation import ValidatedImage


def choose_backend(
    image: ValidatedImage | None,
) -> Literal["text", "vision"]:
    return "vision" if image is not None else "text"

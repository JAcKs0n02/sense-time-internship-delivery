"""Convert validated chat turns to OpenAI-compatible message payloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .validation import ValidatedImage


class MessageContractError(ValueError):
    """Raised when history cannot be represented without ambiguity."""


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


def _validated_history(history: Sequence[Turn]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    expected = "user"
    for turn in history:
        if turn.role != expected or not turn.content.strip():
            raise MessageContractError(
                "history roles must alternate user and assistant with non-empty content"
            )
        output.append({"role": turn.role, "content": turn.content})
        expected = "assistant" if expected == "user" else "user"
    if expected != "user":
        raise MessageContractError(
            "history roles must alternate complete user/assistant pairs"
        )
    return output


def _prompt(prompt: str) -> str:
    value = prompt.strip()
    if not value:
        raise MessageContractError("prompt must be non-empty")
    return value


def to_text_messages(
    history: Sequence[Turn], prompt: str
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = list(_validated_history(history))
    messages.append({"role": "user", "content": _prompt(prompt)})
    return messages


def to_vision_messages(
    history: Sequence[Turn], prompt: str, image: ValidatedImage
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = list(_validated_history(history))
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _prompt(prompt)},
                {"type": "image_url", "image_url": {"url": image.data_url}},
            ],
        }
    )
    return messages

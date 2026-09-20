"""OpenAI-compatible streaming adapter with deferred SDK imports."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Literal

from .config import Settings


def _delta_content(chunk: object) -> str | None:
    if isinstance(chunk, Mapping):
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return None
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            return None
        content = delta.get("content")
        return content if isinstance(content, str) else None
    choices = getattr(chunk, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else None


def iter_text_deltas(chunks: Iterable[object]) -> Iterator[str]:
    """Yield cumulative non-empty text from OpenAI stream chunks."""
    accumulated = ""
    for chunk in chunks:
        content = _delta_content(chunk)
        if not content:
            continue
        accumulated += content
        yield accumulated


def stream_chat_completion(
    settings: Settings,
    backend: Literal["text", "vision"],
    messages: Sequence[Mapping[str, object]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Iterator[str]:
    """Call one selected local service and expose cumulative text."""
    from openai import OpenAI

    base_url = (
        settings.text_base_url if backend == "text" else settings.vision_base_url
    )
    model = settings.text_model if backend == "text" else settings.vision_model
    client = OpenAI(
        base_url=base_url,
        api_key=settings.api_key,
        timeout=settings.request_timeout_seconds,
    )
    chunks = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
    )
    yield from iter_text_deltas(chunks)


def user_facing_error(error: Exception) -> str:
    if isinstance(error, PermissionError):
        return "认证失败：请检查本地 WEEK7_API_KEY 与服务配置。"
    if isinstance(error, TimeoutError):
        return "请求超时：模型可能仍在加载，请稍后重试。"
    if isinstance(error, ConnectionError):
        return "无法连接目标模型服务：请按部署指南启动对应后端。"
    return f"模型服务请求失败：{type(error).__name__}。请查看本地日志。"

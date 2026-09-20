"""Environment-only runtime settings for the Week 7 local services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse


class SettingsError(ValueError):
    """Raised when required service settings are missing or unsafe."""


def _local_base_url(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip().rstrip("/")
    if not value:
        raise SettingsError(f"{name} is required")
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise SettingsError(f"{name} must be an http loopback URL")
    return value


@dataclass(frozen=True)
class Settings:
    text_base_url: str
    vision_base_url: str
    api_key: str
    request_timeout_seconds: float
    text_model: str = "week4-dpo-quantized"
    vision_model: str = "week5-qwen2-vl-base"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        text_base_url = _local_base_url(env, "WEEK7_TEXT_BASE_URL")
        vision_base_url = _local_base_url(env, "WEEK7_VISION_BASE_URL")
        raw_timeout = env.get("WEEK7_REQUEST_TIMEOUT_SECONDS", "120")
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise SettingsError(
                "WEEK7_REQUEST_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if timeout <= 0:
            raise SettingsError(
                "WEEK7_REQUEST_TIMEOUT_SECONDS must be positive"
            )
        return cls(
            text_base_url=text_base_url,
            vision_base_url=vision_base_url,
            api_key=env.get("WEEK7_API_KEY", "local-demo-key"),
            request_timeout_seconds=timeout,
        )

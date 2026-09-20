"""Input and generation-parameter validation for the Week 7 UI."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path


MAX_IMAGE_BYTES = 10 * 1024 * 1024


class InputValidationError(ValueError):
    """Raised when a user input violates the public UI contract."""


@dataclass(frozen=True)
class ValidatedImage:
    path: Path
    mime_type: str
    byte_count: int
    data_url: str


_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _magic_matches(mime_type: str, payload: bytes) -> bool:
    if mime_type == "image/png":
        return payload.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return payload.startswith(b"\xff\xd8\xff")
    if mime_type == "image/webp":
        return len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP"
    return False


def validate_image(path: Path) -> ValidatedImage:
    """Validate one image and encode it only after all bounds pass."""
    suffix = path.suffix.lower()
    mime_type = _MIME_BY_SUFFIX.get(suffix)
    if mime_type is None:
        raise InputValidationError("only JPEG, PNG or WEBP images are accepted")
    if not path.is_file():
        raise InputValidationError("image file does not exist")
    byte_count = path.stat().st_size
    if byte_count <= 0:
        raise InputValidationError("image file is empty")
    if byte_count > MAX_IMAGE_BYTES:
        raise InputValidationError("image exceeds the 10 MiB limit")
    payload = path.read_bytes()
    if not _magic_matches(mime_type, payload):
        raise InputValidationError("image content does not match its extension")
    encoded = base64.b64encode(payload).decode("ascii")
    return ValidatedImage(
        path=path,
        mime_type=mime_type,
        byte_count=byte_count,
        data_url=f"data:{mime_type};base64,{encoded}",
    )


def validate_generation_parameters(
    temperature: float, top_p: float, max_tokens: int
) -> tuple[float, float, int]:
    if not 0.0 <= temperature <= 2.0:
        raise InputValidationError("temperature must be between 0.0 and 2.0")
    if not 0.05 <= top_p <= 1.0:
        raise InputValidationError("top_p must be between 0.05 and 1.0")
    if isinstance(max_tokens, bool) or not 32 <= max_tokens <= 1024:
        raise InputValidationError("max_tokens must be between 32 and 1024")
    return float(temperature), float(top_p), int(max_tokens)

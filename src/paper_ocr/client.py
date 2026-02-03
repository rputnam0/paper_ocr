from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any, Optional

from openai import AsyncOpenAI


@dataclass
class OCRResponse:
    content: str
    usage: dict[str, Any] | None
    raw: dict[str, Any]


def _data_url(mime_type: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _is_retryable_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    return False


async def call_olmocr(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    max_tokens: int,
    max_attempts: int = 5,
) -> OCRResponse:
    attempt = 0
    last_exc: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url(mime_type, image_bytes)},
                            },
                        ],
                    }
                ],
            )
            content = response.choices[0].message.content or ""
            usage_obj = getattr(response, "usage", None)
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif hasattr(usage_obj, "dict"):
                usage = usage_obj.dict()
            else:
                usage = usage_obj
            raw = response.model_dump() if hasattr(response, "model_dump") else response.dict()
            return OCRResponse(content=content, usage=usage, raw=raw)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable_error(exc) or attempt >= max_attempts:
                raise
            backoff = min(2 ** (attempt - 1), 30)
            jitter = 0.2 * backoff
            await asyncio.sleep(backoff + jitter)

    raise last_exc or RuntimeError("Unknown failure")

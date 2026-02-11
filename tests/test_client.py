from __future__ import annotations

import asyncio

from paper_ocr.client import call_olmocr, call_text_model


class _FakeUsage:
    def model_dump(self):  # noqa: D401
        return {"prompt_tokens": 1}


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return _FakeResponse("ok")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_call_olmocr_sets_temperature_zero():
    client = _FakeClient()
    asyncio.run(
        call_olmocr(
            client=client,  # type: ignore[arg-type]
            model="m",
            prompt="p",
            image_bytes=b"x",
            mime_type="image/png",
            max_tokens=10,
            max_attempts=1,
        )
    )
    assert client.chat.completions.calls[0]["temperature"] == 0


def test_call_text_model_sets_temperature_zero():
    client = _FakeClient()
    asyncio.run(
        call_text_model(
            client=client,  # type: ignore[arg-type]
            model="m",
            prompt="p",
            max_tokens=10,
            max_attempts=1,
        )
    )
    assert client.chat.completions.calls[0]["temperature"] == 0

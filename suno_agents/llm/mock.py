"""Scripted runner for tests and the offline demo mode.

A :class:`MockRunner` answers from a list of responses. Each response is either
a raw string or a callable receiving ``(system, messages)`` and returning a
string. Responses are consumed in order; the last one repeats forever so a
pipeline that asks for an extra round never dies mid-test.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from .wire import ChatMessage, ChatResult, ChatUsage, LLMParseError, extract_json

Responder = str | Callable[[str, list[dict[str, str]]], str]


class MockRunner:
    """Deterministic Runner. Records every call for assertions."""

    def __init__(
        self,
        responses: Sequence[Responder],
        *,
        model: str = "mock-model",
        trace=None,
    ) -> None:
        self.model = model
        #: Optional recorder. When present the mock logs the same shape a real
        #: client would, which is what makes the agent-log tab useful in offline
        #: demo mode instead of showing decisions with no traffic above them.
        self.trace = trace
        self.label = "demo"
        self._responses = list(responses) or ["{}"]
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    # ------------------------------------------------------------------ helpers
    def push(self, response: Responder) -> None:
        """Append a response (useful for mid-test scripting)."""
        self._responses.append(response)

    @property
    def prompt_of(self) -> list[str]:
        """All user/system text sent so far, newest first."""
        return [
            "\n".join(m["content"] for m in call["messages"]) for call in reversed(self.calls)
        ]

    def saw(self, needle: str) -> bool:
        """True when any prompt so far contained ``needle``."""
        return any(needle in text for text in self.prompt_of)

    # ------------------------------------------------------------------ Runner
    def complete(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: str | None = None,
        schema: Mapping[str, Any] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ChatResult:
        wire = [{"role": "system", "content": system or ""}] + [
            m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in messages
        ]
        self.calls.append(
            {
                "messages": wire,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        started = self._now()
        text = self._next(system or "", wire)
        if on_delta is not None:
            on_delta(text)
        result = ChatResult(
            text=text,
            finish_reason="stop",
            model=self.model,
            usage=ChatUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
        )
        self._trace(wire, result, started, json_mode)
        return result

    @staticmethod
    def _now() -> float:
        import time

        return time.perf_counter()

    def _trace(self, wire, result: ChatResult, started: float, json_mode) -> None:
        """Record a simulated exchange, labelled like a real one."""
        if self.trace is None:
            return
        try:
            import time

            self.trace.record_exchange(
                label=self.label,
                stage=self.label.split("#", 1)[0],
                model=self.model,
                provider="demo (runner tiruan)",
                url="in-process://mock",
                stream=False,
                duration_s=round(time.perf_counter() - started, 3),
                messages=[
                    {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                    for m in wire
                ],
                temperature=None,
                max_tokens=None,
                json_mode=json_mode or "",
                response=result.text,
                finish_reason=result.finish_reason,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                status="ok",
            )
        except Exception:  # pragma: no cover - tracing must never break a call
            pass

    def complete_json(self, messages, *, retries: int = 2, on_delta=None, **kwargs):
        result = self.complete(messages, on_delta=on_delta, **kwargs)
        try:
            return extract_json(result.text), result
        except LLMParseError:
            raise

    def _next(self, system: str, wire: list[dict[str, str]]) -> str:
        responder = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        if callable(responder):
            return str(responder(system, wire))
        return str(responder)


def json_responder(payload: Any) -> Responder:
    """Turn a Python object into a responder that emits its JSON form."""

    def responder(_system: str, _wire: list[dict[str, str]]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    return responder


def sequence(*payloads: Any) -> list[Responder]:
    """Convenience: JSON-encode a list of payloads into responders."""
    return [json_responder(p) if not isinstance(p, str) else p for p in payloads]


__all__ = ["MockRunner", "json_responder", "sequence", "Responder"]

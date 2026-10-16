"""Pure helpers that translate between OpenAI wire format and our types.

Kept free of I/O so it can be unit-tested without a server.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "bot": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "developer": "system",
}

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """Any failure while talking to a provider endpoint."""


class LLMHTTPError(LLMError):
    """Transport or HTTP-status failure; ``status`` is 0 for network errors."""

    def __init__(self, message: str, *, status: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def is_retryable(self) -> bool:
        return self.status in (0, 408, 409, 425, 429, 500, 502, 503, 504, 522, 524)


class LLMParseError(LLMError):
    """The model replied, but not with the JSON shape we asked for."""

    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class LLMEmptyResponse(LLMHTTPError):
    """The model produced no visible content.

    Its own type because it is *repairable*, unlike a transport or auth failure:
    the usual cause is an output budget spent on hidden reasoning, and the fix is a
    bigger budget plus a demand for an immediate answer. ``chat_json`` retries this
    and never retries a 401.
    """

    def __init__(self, message: str, *, reasoning_chars: int = 0, finish_reason: str = "") -> None:
        super().__init__(message, status=0, body=message)
        self.reasoning_chars = reasoning_chars
        self.finish_reason = finish_reason

    @property
    def looks_like_reasoning_burn(self) -> bool:
        """True when hidden reasoning, not a broken server, ate the answer."""
        return self.reasoning_chars > 0 or self.finish_reason == "length"


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": ROLE_ALIASES.get(self.role, self.role), "content": self.content}


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_wire(cls, data: Mapping[str, Any] | None) -> "ChatUsage":
        if not data:
            return cls()
        prompt = int(data.get("prompt_tokens") or 0)
        completion = int(data.get("completion_tokens") or 0)
        total = int(data.get("total_tokens") or (prompt + completion))
        return cls(prompt, completion, total)


@dataclass
class ChatResult:
    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    model: str = ""
    usage: ChatUsage = field(default_factory=ChatUsage)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason in {"length", "max_tokens", "max_output_tokens"}


def normalise_messages(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    system: str | None = None,
) -> list[dict[str, str]]:
    """Return a wire-ready message list, optionally prefixed with ``system``."""
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        if isinstance(message, ChatMessage):
            out.append(message.to_dict())
            continue
        role = ROLE_ALIASES.get(str(message.get("role", "user")), "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        out.append({"role": role, "content": content})
    return out


def strip_reasoning(text: str) -> tuple[str, str]:
    """Split inline reasoning blocks from the visible answer.

    Reasoning models behind OpenAI-compatible servers often leak
    ``<think…>`` blocks or a ``Reasoning:`` preamble into ``content``.
    """
    reasoning_parts: list[str] = []
    cleaned = text
    for match in _THINK_RE.finditer(text):
        reasoning_parts.append(match.group(2).strip())
    cleaned = _THINK_RE.sub("", cleaned).strip()
    return cleaned, "\n\n".join(part for part in reasoning_parts if part)


def extract_json(text: str) -> Any:
    """Best-effort JSON recovery from a model reply.

    Handles fenced blocks, leading prose, trailing commentary and the
    ``[{...}]``-wrapped replies some local servers emit.
    """
    if not text or not text.strip():
        raise LLMParseError("Model returned an empty response.", raw=text)

    cleaned = _THINK_RE.sub("", text).strip()
    candidates: list[str] = []

    fenced = _FENCE_RE.findall(cleaned)
    if fenced:
        candidates.extend(block.strip() for block in fenced)
    candidates.append(cleaned)

    for candidate in candidates:
        parsed = _try_json(candidate)
        if parsed is not _MISSING:
            return _unwrap_singleton(parsed)
        for chunk in _balanced_objects(candidate):
            parsed = _try_json(chunk)
            if parsed is not _MISSING:
                return _unwrap_singleton(parsed)

    raise LLMParseError(
        "Could not find a JSON object in the model response. "
        f"First 200 chars: {cleaned[:200]!r}",
        raw=text,
    )


_MISSING = object()


def _try_json(candidate: str) -> Any:
    candidate = candidate.strip()
    if not candidate:
        return _MISSING
    # Tolerate trailing commas, a very common local-model defect.
    trimmed = re.sub(r",(\s*[}\]])", r"\1", candidate)
    for attempt in (candidate, trimmed):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return _MISSING


def _unwrap_singleton(value: Any) -> Any:
    """Unwrap the ``[{"result": ...}]`` reply shape used by chat-templated servers."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        # A list of section objects is a legitimate answer; a list of one answer
        # has already been unwrapped above.
        return value
    return value


def _balanced_objects(text: str) -> Iterable[str]:
    """Yield balanced ``{...}`` spans, ignoring braces inside strings."""
    start = -1
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : index + 1]


def parse_chat_response(payload: Mapping[str, Any]) -> ChatResult:
    """Convert a non-streaming chat-completions payload into :class:`ChatResult`."""
    if "error" in payload and "choices" not in payload:
        error = payload["error"]
        message = error.get("message") if isinstance(error, Mapping) else str(error)
        raise LLMHTTPError(f"Provider returned an error object: {message}")

    choices = payload.get("choices") or []
    if not choices:
        raise LLMError("Provider returned no choices.")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Some servers return content parts.
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, Mapping)
        )
    content = content or ""
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or (choice.get("reasoning_content") if isinstance(choice, Mapping) else "")
        or ""
    )
    visible, inline_reasoning = strip_reasoning(str(content))
    if inline_reasoning:
        reasoning = f"{reasoning}\n\n{inline_reasoning}".strip()

    return ChatResult(
        text=visible,
        reasoning=str(reasoning),
        finish_reason=str(choice.get("finish_reason") or ""),
        model=str(payload.get("model") or ""),
        usage=ChatUsage.from_wire(payload.get("usage")),
        tool_calls=list(message.get("tool_calls") or []),
        raw=dict(payload),
    )


def parse_stream_chunk(data: str) -> tuple[str, str, bool, Mapping[str, Any] | None]:
    """Parse one SSE payload; returns ``(content_delta, reasoning_delta, done, usage)``."""
    if data.strip() == "[DONE]":
        return "", "", True, None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return "", "", False, None
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    choices = payload.get("choices") or []
    if not choices:
        return "", "", False, usage
    delta = choices[0].get("delta") or {}
    content = delta.get("content") or ""
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, Mapping))
    return str(content), str(reasoning), False, usage


#: OpenAI (and the many gateways that clone it) reject ``response_format:
#: json_object`` unless the word "json" actually appears in the prompt, and with
#: streaming enabled that rule is enforced strictly. A silent 400 from an
#: otherwise correct request is a miserable thing to debug, so the client
#: guarantees the word is present instead of hoping each stage prompt says it.
JSON_OBJECT_HINT = (
    "Respond with a single valid JSON object only, and nothing else. "
    "The JSON must contain every requested key; no markdown fences, no commentary."
)


def mentions_json(messages: Sequence[Mapping[str, Any]]) -> bool:
    """True when the conversation already mentions JSON somewhere."""
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str) and "json" in content.lower():
            return True
        if isinstance(content, list):  # content parts
            for part in content:
                if isinstance(part, Mapping) and "json" in str(part.get("text", "")).lower():
                    return True
    return False


def ensure_json_keyword(wire: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    """Append the JSON instruction when ``json_object`` mode needs it.

    Only ``json_object`` is affected: ``json_schema`` mode carries the requirement
    through the schema itself, and ``off`` asks for no JSON at all.
    """
    if mode != "object" or not wire or mentions_json(wire):
        return wire
    patched = list(wire)
    last = patched[-1]
    if last.get("role") == "user":
        patched[-1] = {
            "role": "user",
            "content": f"{last.get('content', '')}\n\n{JSON_OBJECT_HINT}",
        }
    else:
        patched.append({"role": "user", "content": JSON_OBJECT_HINT})
    return patched


def json_response_format(mode: str, schema: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build ``response_format`` for the requested strictness."""
    if mode == "off":
        return None
    if mode == "schema" and schema:
        return {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": False},
        }
    return {"type": "json_object"}


def is_unsupported_parameter(body: str, parameter: str) -> bool:
    """Detect the many ways servers say 'I do not know that field'."""
    lowered = body.lower()
    if parameter.lower() not in lowered:
        return False
    markers = (
        "unsupported",
        "unrecognized",
        "unrecognised",
        "unknown",
        "not supported",
        "invalid",
        "extra_forbidden",
        "unexpected",
    )
    return any(marker in lowered for marker in markers)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for context budgeting."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def fit_messages(
    system: str,
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    *,
    max_prompt_tokens: int,
) -> list[dict[str, str]]:
    """Drop the oldest turns until the prompt fits the budget.

    The system prompt and the last user turn are always kept.
    """
    wire = normalise_messages(messages, system)
    while len(wire) > 2 and sum(estimate_tokens(m["content"]) for m in wire) > max_prompt_tokens:
        wire.pop(1)
    return wire

"""Blocking OpenAI-compatible chat-completions client.

Deliberately dependency-light (stdlib ``urllib``) so the app installs with
nothing but the standard library plus numpy. Works against OpenAI, OpenRouter,
Groq, DeepSeek, vLLM, llama.cpp server, LM Studio, Ollama's ``/v1`` shim, and
anything else that speaks the same dialect.

Compatibility notes, learned the hard way against real servers:

* Some endpoints reject ``response_format``; we retry without it and remember.
* Some reject ``temperature``; same treatment.
* Reasoning models leak ``<think…>`` blocks into ``content``; those are split
  out by :mod:`suno_agents.llm.wire` before the caller ever sees the text.
* Streaming is optional per provider; the caller falls back to one-shot calls
  automatically, and only stream failures on the *first* chunk are retried.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, Mapping, Sequence

from .trace import TraceRecorder
from .wire import (
    ChatMessage,
    ensure_json_keyword,
    ChatResult,
    ChatUsage,
    LLMError,
    LLMHTTPError,
    LLMEmptyResponse,
    LLMParseError,
    estimate_tokens,
    extract_json,
    fit_messages,
    is_unsupported_parameter,
    json_response_format,
    normalise_messages,
    parse_chat_response,
    parse_stream_chunk,
)

#: Parameters known to be rejected by some OpenAI-compatible servers. When a
#: request fails with an "unsupported parameter" error we drop the offender and
#: remember it for the lifetime of the client, so later calls stay cheap.
OPTIONAL_PARAMETERS = ("response_format", "temperature", "top_p", "max_tokens")

EventHook = Callable[[str, str], None]
DeltaHook = Callable[[str], None]


class LLMClient:
    """One provider endpoint, with retries, streaming and JSON mode support."""

    def __init__(
        self,
        provider,
        *,
        on_event: EventHook | None = None,
        trace: TraceRecorder | None = None,
        label: str = "",
    ) -> None:
        self.provider = provider
        self.on_event = on_event
        #: Optional log of every request and reply. A client without one behaves
        #: exactly as before, so the recorder is never load-bearing.
        self.trace = trace
        self.label = label
        self._unsupported: set[str] = set()
        self._json_hint_forced = False
        self._lock = threading.Lock()
        self.last_usage = ChatUsage()

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, message: str) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, message)
        except Exception:  # pragma: no cover - UI callbacks must never break a run
            pass

    # ------------------------------------------------------------------ public
    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: str | None = None,
        schema: Mapping[str, Any] | None = None,
        stream: bool | None = None,
        on_delta: DeltaHook | None = None,
        context_budget: int = 120_000,
        model: str | None = None,
    ) -> ChatResult:
        """Send a completion request and return the parsed result.

        Raises :class:`LLMHTTPError` on transport/status failure (with
        ``is_retryable`` telling the caller whether a retry makes sense) and
        :class:`LLMParseError` when JSON was demanded but not delivered.
        """
        mode = json_mode or self.provider.json_mode
        want_stream = self.provider.supports_streaming if stream is None else stream
        wire = fit_messages(system or "", messages, max_prompt_tokens=context_budget)

        use_stream = bool(want_stream and on_delta is not None)
        result: ChatResult | None = None
        last_error: Exception | None = None

        for attempt in range(max(1, int(self.provider.max_retries) + 1)):
            try:
                if use_stream:
                    result = self._stream_once(
                        wire, mode, schema, temperature, max_tokens, on_delta, model
                    )
                else:
                    result = self._post_once(
                        wire, mode, schema, temperature, max_tokens, model
                    )
                break
            except LLMHTTPError as exc:
                last_error = exc
                if self._absorb_unsupported(exc):
                    continue
                if not exc.is_retryable or attempt == max(1, int(self.provider.max_retries)):
                    raise
                self._sleep_backoff(attempt, exc)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = LLMHTTPError(f"Network error: {exc}", status=0, body=str(exc))
                if attempt == max(1, int(self.provider.max_retries)):
                    raise last_error
                self._sleep_backoff(attempt, last_error)

        if result is None:
            raise last_error or LLMError("Request failed without a result.")
        if result.usage.total_tokens:
            with self._lock:
                self.last_usage = result.usage
        return result

    def chat_json(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        *,
        system: str | None = None,
        schema: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool | None = None,
        on_delta: DeltaHook | None = None,
        retries: int = 2,
        model: str | None = None,
    ) -> tuple[Any, ChatResult]:
        """Demand JSON, repairing the reply across a bounded number of retries.

        Two failure modes are handled separately, because they need opposite fixes:

        * **Truncated** (``finish_reason == "length"``): the output budget ran out.
          Retry with a bigger budget and a demand for a compact answer. The budget
          is raised even when the caller left ``max_tokens`` unset, because the
          effective cap then comes from the provider config - raising only the
          call-level value would silently never apply.
        * **Reasoning-only** (empty ``content``): reasoning models spend the output
          budget on hidden reasoning tokens, so a generous-looking cap can come
          back with nothing usable at all. Measured against one such model:
          ``max_tokens=300`` returned 1032 characters of reasoning, zero characters
          of content, and ``reasoning_tokens=300``. The fix is more budget plus an
          explicit instruction to answer without deliberating.
        """
        conversation = list(messages)
        last_raw = ""
        #: Effective output cap for this call. ``0``/``None`` means "no cap": the
        #: request carries no max_tokens and the provider applies its own maximum.
        #: Tracked locally so a retry can raise it *only* when it exists - retrying
        #: an uncapped call with a cap would be a regression.
        budget = max_tokens if max_tokens is not None else getattr(self.provider, "max_tokens", 0)
        budget = int(budget or 0)
        for attempt in range(retries + 1):
            try:
                result = self.chat(
                    conversation,
                    system=system,
                    temperature=temperature,
                    max_tokens=budget,
                    json_mode=self.provider.json_mode,
                    schema=schema,
                    stream=stream,
                    on_delta=on_delta,
                    model=model,
                )
            except LLMEmptyResponse as exc:
                # Repairable: the model answered, it just answered with reasoning.
                # A transport error or a 401 is not, and is raised untouched.
                if attempt >= retries or not exc.looks_like_reasoning_burn:
                    raise
                self._emit(
                    "warn",
                    f"Model produced no visible content (attempt {attempt + 1}); "
                    f"raising the output budget to {self._raise_budget(budget)} and "
                    "asking for an immediate answer.",
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Answer immediately without deliberating: emit the JSON "
                            "object as your first characters, and keep it compact."
                        ),
                    }
                )
                budget = self._raise_budget(budget)
                continue
            last_raw = result.text
            try:
                return extract_json(result.text), result
            except LLMParseError as exc:
                if attempt >= retries:
                    raise
                self._emit(
                    "warn",
                    f"Reply was not valid JSON (attempt {attempt + 1}) - asking the model to fix it.",
                )
                if result.truncated or not result.text.strip():
                    reason = (
                        "cut off by the output token limit (max_tokens="
                        + (str(budget) if budget else "provider maximum")
                        + ")"
                        if result.truncated
                        else "returned nothing but hidden reasoning"
                    )
                    conversation.append({"role": "assistant", "content": result.text[:2000]})
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous answer was {reason}. Answer immediately, "
                                "without deliberation: one COMPLETE, COMPACT JSON object, "
                                "no commentary, no markdown fences, no trailing text."
                            ),
                        }
                    )
                    budget = self._raise_budget(budget)
                else:
                    conversation.append({"role": "assistant", "content": result.text[:4000]})
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "That reply could not be parsed as JSON "
                                f"({exc}). Reply with ONLY one valid JSON object, "
                                "matching the required keys exactly. No markdown, "
                                "no explanation, no trailing text."
                            ),
                        }
                    )
        raise LLMParseError("Model never produced valid JSON.", raw=last_raw)

    def list_models(self, timeout: float = 20.0) -> list[str]:
        """Return model ids from ``GET /models``; empty list when unsupported."""
        self._require_usable_url()
        request = urllib.request.Request(
            self.provider.models_url, headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", "replace") or "{}")
        except urllib.error.HTTPError as exc:
            raise LLMHTTPError(
                f"GET {self.provider.models_url} failed: HTTP {exc.code}",
                status=exc.code,
                body=_safe_read(exc),
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMHTTPError(f"Cannot reach {self.provider.models_url}: {exc}") from exc
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if isinstance(data, list):
            return sorted(
                str(item.get("id"))
                for item in data
                if isinstance(item, Mapping) and item.get("id")
            )
        if isinstance(payload, Mapping) and isinstance(payload.get("models"), list):
            return sorted(str(item) for item in payload["models"])
        return []

    def test_connection(self, model: str | None = None) -> tuple[bool, str, float]:
        """Round-trip a one-token request. Returns ``(ok, message, latency_s)``."""
        started = time.perf_counter()
        body: dict[str, Any] = {
            "model": model or self.provider.model,
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 8,
            "stream": False,
        }
        try:
            status, text = self._post_raw(body, timeout=min(45.0, float(self.provider.timeout)))
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the settings dialog
            return False, f"{type(exc).__name__}: {exc}", time.perf_counter() - started
        latency = time.perf_counter() - started
        if status != 200:
            return False, f"HTTP {status}: {text[:300]}", latency
        try:
            parsed = parse_chat_response(json.loads(text))
        except (json.JSONDecodeError, LLMError) as exc:
            return False, f"Unexpected response shape: {exc}", latency
        model_name = parsed.model or body["model"]
        return True, f"OK - {model_name} replied in {latency * 1000:.0f} ms", latency

    # ------------------------------------------------------------------ internals
    def _require_usable_url(self) -> None:
        """Fail with a diagnosable error before urllib can raise a bare ValueError.

        urllib raises ``ValueError: unknown url type`` for a malformed base URL, and
        the UI only catches :class:`LLMError`. Turning that into our own error type
        is the difference between "settings dialog shows the problem" and "the app
        crashes with a stack trace".
        """
        url = str(self.provider.chat_url or "")
        if not url.startswith(("http://", "https://")):
            raise LLMHTTPError(
                f"{self.provider.name}: base URL must start with http:// or https:// "
                f"(got {url!r}).",
                status=0,
                body=url,
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LirikStudio/1.0 (+openai-compatible)",
        }
        key = self.provider.resolved_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        for name, value in (self.provider.extra_headers or {}).items():
            headers[str(name)] = str(value)
        return headers

    def _build_body(
        self,
        wire: list[dict[str, str]],
        mode: str,
        schema: Mapping[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        *,
        stream: bool,
        model: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.provider.model,
            "messages": ensure_json_keyword(wire, mode),
            "stream": stream,
        }
        if "temperature" not in self._unsupported:
            body["temperature"] = float(
                self.provider.temperature if temperature is None else temperature
            )
        if "top_p" not in self._unsupported and self.provider.top_p not in (None, 1.0):
            body["top_p"] = float(self.provider.top_p)
        # Omit the field entirely when there is no cap, rather than sending a huge
        # number: some gateways validate against their own maximum and answer 400,
        # and "no limit" should mean the provider decides, not this client.
        cap = max_tokens if max_tokens is not None else self.provider.max_tokens
        if "max_tokens" not in self._unsupported and cap:
            body["max_tokens"] = int(cap)
        if "response_format" not in self._unsupported:
            response_format = json_response_format(mode, schema)
            if response_format:
                body["response_format"] = response_format
        body.update(self.provider.extra_body or {})
        return body

    def _post_once(
        self,
        wire: list[dict[str, str]],
        mode: str,
        schema: Mapping[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        model: str | None = None,
    ) -> ChatResult:
        body = self._build_body(
            wire, mode, schema, temperature, max_tokens, stream=False, model=model
        )
        self._emit(
            "debug",
            f"POST {self.provider.chat_url} model={body['model']} "
            f"prompt_tokens~{sum(estimate_tokens(m['content']) for m in wire)}",
        )
        started = time.perf_counter()
        status, text = self._post_raw(body)
        elapsed = time.perf_counter() - started
        if status != 200:
            self._trace(
                wire=wire, model=str(body.get("model", "")), mode=mode, stream=False,
                temperature=temperature, max_tokens=max_tokens, duration=elapsed,
                status="error", error=_error_detail(text).lstrip(" -"),
                response=text[:4000],
            )
            raise LLMHTTPError(
                f"{self.provider.name}: HTTP {status} from {self.provider.chat_url}"
                + _error_detail(text),
                status=status,
                body=text,
            )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            self._trace(
                wire=wire, model=str(body.get("model", "")), mode=mode, stream=False,
                temperature=temperature, max_tokens=max_tokens, duration=elapsed,
                status="error", error=f"response was not JSON ({exc})",
                response=text[:4000],
            )
            raise LLMHTTPError(
                f"{self.provider.name}: response was not JSON ({exc}).",
                status=status,
                body=text,
            ) from exc
        result = parse_chat_response(payload)
        # An empty or truncated reply is a problem, so the log must say *which*
        # problem - a bare "[empty]" status with no explanation is exactly the kind
        # of entry that sends someone back to reading source code.
        if not result.text.strip():
            status = "empty"
            trace_error = (
                f"balasan tanpa teks (finish_reason={result.finish_reason or 'unknown'}"
                + (f", reasoning {len(result.reasoning)} karakter)" if result.reasoning else ")")
            )
        elif result.truncated:
            status = "truncated"
            trace_error = (
                f"balasan terpotong oleh batas token "
                f"(finish_reason={result.finish_reason}, max_tokens={max_tokens})"
            )
        else:
            status, trace_error = "ok", ""
        self._trace(
            wire=wire, model=result.model or str(body.get("model", "")), mode=mode,
            stream=False, temperature=temperature, max_tokens=max_tokens,
            duration=elapsed, status=status, error=trace_error,
            response=result.text, reasoning=result.reasoning,
            finish_reason=result.finish_reason, usage=result.usage,
        )
        if not result.text.strip():
            # An empty completion is never a usable answer, and the two causes need
            # different user action, so say which one it is. Reasoning models can
            # spend the entire completion budget before emitting a first character.
            note = (
                f" Only hidden reasoning was produced ({len(result.reasoning)} chars) - "
                "raise max_tokens for this provider."
                if result.reasoning.strip()
                else f" finish_reason={result.finish_reason or 'unknown'}."
            )
            raise LLMEmptyResponse(
                f"{self.provider.name}: model returned no visible content.{note}",
                reasoning_chars=len(result.reasoning),
                finish_reason=result.finish_reason,
            )
        return result

    def _stream_once(
        self,
        wire: list[dict[str, str]],
        mode: str,
        schema: Mapping[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        on_delta: DeltaHook | None,
        model: str | None = None,
    ) -> ChatResult:
        body = self._build_body(
            wire, mode, schema, temperature, max_tokens, stream=True, model=model
        )
        self._require_usable_url()
        request = urllib.request.Request(
            self.provider.chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = ChatUsage()
        finish_reason = ""
        resolved_model = model or self.provider.model
        status = 200
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=float(self.provider.timeout)) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    chunk_content, chunk_reasoning, done, chunk_usage = parse_stream_chunk(
                        line[len("data:") :].strip()
                    )
                    if chunk_usage:
                        usage = ChatUsage.from_wire(chunk_usage)
                    if chunk_content:
                        content_parts.append(chunk_content)
                        if on_delta is not None:
                            on_delta(chunk_content)
                    if chunk_reasoning:
                        reasoning_parts.append(chunk_reasoning)
                    if done:
                        break
        except urllib.error.HTTPError as exc:
            detail_body = _safe_read(exc)
            self._trace(
                wire=wire, model=resolved_model, mode=mode, stream=True,
                temperature=temperature, max_tokens=max_tokens,
                duration=time.perf_counter() - started, status="error",
                error=_error_detail(detail_body).lstrip(" -"),
                response=detail_body[:4000],
            )
            raise LLMHTTPError(
                f"{self.provider.name}: HTTP {exc.code} from {self.provider.chat_url}"
                + _error_detail(detail_body),
                status=exc.code,
                body=detail_body,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._trace(
                wire=wire, model=resolved_model, mode=mode, stream=True,
                temperature=temperature, max_tokens=max_tokens,
                duration=time.perf_counter() - started, status="error",
                error=f"stream failed: {exc}",
            )
            raise LLMHTTPError(f"{self.provider.name}: stream failed: {exc}", status=0) from exc

        text = "".join(content_parts)
        self._trace(
            wire=wire, model=resolved_model, mode=mode, stream=True,
            temperature=temperature, max_tokens=max_tokens,
            duration=time.perf_counter() - started,
            status="empty" if not text.strip() else "ok",
            error="" if text.strip() else "stream tanpa teks terlihat",
            response=text, reasoning="\n\n".join(reasoning_parts),
            finish_reason=finish_reason, usage=usage,
        )
        if not text.strip():
            reasoning_note = (
                f" Only hidden reasoning was produced ({len(''.join(reasoning_parts))} chars) - "
                "this model spends its output budget on reasoning, so raise max_tokens."
                if reasoning_parts
                else " The server may not support stream=true for this model."
            )
            raise LLMEmptyResponse(
                f"{self.provider.name}: stream produced no visible content.{reasoning_note}",
                reasoning_chars=len("".join(reasoning_parts)),
                # finish_reason is what makes the retry meaningful: a stream that ends
                # on "length" burned its budget, so the repair loop must raise it. A
                # live run retried twice at the same cap and failed identically
                # (41906 then 44967 characters of reasoning) precisely because this
                # was reported as "no content" rather than "truncated".
                finish_reason=finish_reason,
            )
        return ChatResult(
            text=text.strip(),
            reasoning="\n\n".join(p for p in reasoning_parts if p).strip(),
            finish_reason=finish_reason,
            model=resolved_model,
            usage=usage,
        )

    def _post_raw(
        self, body: Mapping[str, Any], timeout: float | None = None
    ) -> tuple[int, str]:
        self._require_usable_url()
        request = urllib.request.Request(
            self.provider.chat_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(timeout or self.provider.timeout)
            ) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, _safe_read(exc)

    #: First retry budget for a truncated or reasoning-only reply. Deliberately
    #: generous: the observed failure is a 31.5k-character artifact that needs
    #: ~15.7k completion tokens, so a smaller retry merely fails again and burns
    #: another several minutes.
    RETRY_BUDGET_FLOOR = 16384
    RETRY_BUDGET_CEILING = 49152

    @classmethod
    def _raise_budget(
        cls, budget: int | None, ceiling: int | None = None
    ) -> int:
        """Grow an output budget for a retry, never shrinking it.

        ``0`` means "no cap" and is returned unchanged: there is nothing to raise,
        and turning an uncapped call into a capped one during a repair would undo
        the setting the user asked for. Otherwise the floor matters as much as the
        growth, because a reply cut off at 8k tokens needs real headroom - 1.5x a
        small budget would simply be cut off again after another long wait.
        """
        ceiling = ceiling or cls.RETRY_BUDGET_CEILING
        if budget is None or int(budget) == 0:
            return 0
        grown = max(cls.RETRY_BUDGET_FLOOR, int(int(budget) * 1.5))
        return min(grown, ceiling)

    def _trace(
        self,
        *,
        wire: Sequence[Mapping[str, str]],
        model: str,
        mode: str,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        duration: float,
        status: str,
        response: str = "",
        reasoning: str = "",
        finish_reason: str = "",
        usage: ChatUsage | None = None,
        error: str = "",
    ) -> None:
        """Log one completed request. Never raises: tracing is not load-bearing."""
        if self.trace is None:
            return
        try:
            usage = usage or ChatUsage()
            self.trace.record_exchange(
                label=self.label or self.provider.name,
                stage=_stage_from_label(self.label),
                model=model,
                provider=self.provider.name,
                url=self.provider.chat_url,
                stream=stream,
                duration_s=round(duration, 2),
                messages=[{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in wire],
                temperature=temperature if temperature is not None else self.provider.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.provider.max_tokens,
                json_mode=mode,
                response=response,
                reasoning=reasoning,
                finish_reason=finish_reason,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                status=status,
                error=error,
            )
        except Exception:  # pragma: no cover - tracing must never break a call
            pass

    def _absorb_unsupported(self, exc: LLMHTTPError) -> bool:
        """Drop a rejected parameter, or satisfy a rejected JSON-mode precondition.

        Returns True when the request is worth retrying. Two distinct failures land
        here: a server that does not know a parameter at all (we drop it), and a
        server that refuses ``json_object`` because the prompt never said "json"
        while streaming (we force the hint in and retry, which is cheaper than
        dropping structured output entirely).
        """
        if exc.status not in (400, 404, 415, 422):
            return False
        if self._is_json_keyword_rejection(exc):
            if not self._json_hint_forced:
                self._json_hint_forced = True
                self._emit(
                    "warn",
                    f"{self.provider.name}: endpoint requires the word 'json' in the "
                    "prompt for json_object mode - adding it and retrying.",
                )
                return True
            # The hint is already there and it still refuses: fall back to plain
            # text and let the JSON repair loop in chat_json sort it out.
            if "response_format" not in self._unsupported:
                self._unsupported.add("response_format")
                self._emit(
                    "warn",
                    f"{self.provider.name}: json_object refused even with an explicit "
                    "JSON instruction - continuing without response_format.",
                )
                return True
            return False
        for parameter in OPTIONAL_PARAMETERS:
            if parameter in self._unsupported:
                continue
            if is_unsupported_parameter(exc.body, parameter):
                self._unsupported.add(parameter)
                self._emit(
                    "warn",
                    f"{self.provider.name}: endpoint rejects '{parameter}' - "
                    "retrying without it for this session.",
                )
                return True
        return False

    @staticmethod
    def _is_json_keyword_rejection(exc: LLMHTTPError) -> bool:
        """Detect the documented OpenAI precondition failure.

        The server answers 400 with variants like "Prompt must contain the word
        'json' in some form to use 'response_format' of type 'json_object'".
        Matching the phrase rather than a keyword avoids false positives from
        servers that merely echo the request body back in their error.
        """
        body = (exc.body or "").lower()
        if "json" not in body:
            return False
        if "response_format" not in body and "json_object" not in body:
            return False
        markers = (
            "must contain",
            "should contain",
            "must include",
            "word 'json'",
            'word "json"',
            "the word json",
            "at least one message",
        )
        return any(marker in body for marker in markers)

    def _sleep_backoff(self, attempt: int, exc: LLMHTTPError) -> None:
        delay = min(30.0, 1.5 * (2**attempt))
        if exc.status == 429:
            delay = max(delay, 5.0)
        self._emit("warn", f"Retrying in {delay:.1f}s after: {exc}")
        time.sleep(delay)


def _stage_from_label(label: str) -> str:
    """Derive the stage id from a client label like ``composer#2``."""
    if not label:
        return ""
    return label.split("#", 1)[0].strip()


def _error_detail(body: str, limit: int = 300) -> str:
    """One-line provider error text for exception messages.

    Servers routinely wrap the real reason in an ``{"error": {"message": ...}}``
    envelope, and burying it means the user sees "HTTP 400" and nothing else -
    which is exactly the situation that made this client's JSON-mode bug take
    twenty minutes to find instead of ten seconds.
    """
    if not body:
        return ""
    detail = body.strip()
    try:
        payload = json.loads(detail)
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                detail = str(error.get("message") or error.get("code") or detail)
            elif isinstance(error, str):
                detail = error
            elif payload.get("message"):
                detail = str(payload["message"])
    except (json.JSONDecodeError, ValueError):
        pass
    detail = " ".join(detail.split())[:limit]
    return f" - {detail}" if detail else ""


def _safe_read(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:4000]
    except Exception:  # pragma: no cover - body already consumed
        return ""


class StageRoutedClient:
    """Pins one stage's model and output cap onto a shared provider endpoint.

    Why this exists: per-stage routing needs a different *model* per stage, but
    building a fresh :class:`LLMClient` for each stage would throw away the learned
    compatibility state (which optional parameters this endpoint rejects) and issue
    a separate connection pool per stage. Wrapping is cheaper and keeps that state.

    The previous implementation had the orchestrator call ``runner.for_stage()``,
    which only the test and demo runners implemented - so in the real app every
    stage quietly used the active provider's model and the routing UI had no effect
    at all.
    """

    def __init__(self, client: LLMClient, *, model: str, max_tokens: int | None = None) -> None:
        self._client = client
        self._model = model or client.provider.model
        self._max_tokens = max_tokens

    # Delegate the surface the agents actually use.
    @property
    def provider(self):
        return self._client.provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def on_event(self):
        return self._client.on_event

    def attach_event_hook(self, hook: EventHook | None) -> None:
        """Forward warnings from the underlying client to the caller's UI."""
        self._client.on_event = hook
        self._client._emit = (
            (lambda kind, message: hook(kind, message)) if hook else (lambda *_: None)
        )

    @property
    def trace(self):
        return self._client.trace

    @property
    def label(self) -> str:
        return self._client.label

    @label.setter
    def label(self, value: str) -> None:
        self._client.label = value

    def chat(self, messages, **kwargs) -> ChatResult:
        kwargs["model"] = self._model
        if self._max_tokens:
            kwargs.setdefault("max_tokens", self._max_tokens)
        return self._client.chat(messages, **kwargs)

    def chat_json(self, messages, **kwargs):
        kwargs["model"] = self._model
        if self._max_tokens:
            kwargs.setdefault("max_tokens", self._max_tokens)
        return self._client.chat_json(messages, **kwargs)

    def test_connection(self, model: str | None = None):
        return self._client.test_connection(self._model)

    def list_models(self, timeout: float = 20.0):
        return self._client.list_models(timeout)


def client_for(provider, *, on_event: EventHook | None = None) -> LLMClient:
    """Factory kept separate so tests can monkeypatch a single seam."""
    return LLMClient(provider, on_event=on_event)

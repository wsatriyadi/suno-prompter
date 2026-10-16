"""Agent runtime: one LLM call, one artifact, one prompt built from shared law."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..domain.graph import STAGE_INFO
from ..llm.wire import ChatMessage, ChatResult, ChatUsage, LLMError
from .prompts import JSON_DISCIPLINE, build_system


class AgentError(RuntimeError):
    """The agent could not produce a usable artifact."""


@dataclass
class AgentRun:
    """Telemetry for one agent invocation, recorded into the run history."""

    stage: str = ""
    model: str = ""
    duration_s: float = 0.0
    usage: ChatUsage = field(default_factory=ChatUsage)
    validation_problems: list[str] = field(default_factory=list)
    attempts: int = 0
    raw: str = ""


class Agent(ABC):
    """Base class for the nine specialists.

    Subclasses declare three things: the stage id, the JSON schema, and the
    system prompt. Everything else - calling, retrying, parsing, validating,
    coercing - happens here so the specialists stay focused on craft.
    """

    stage: str = ""
    schema: dict[str, object] = {}
    #: How many times to re-ask the model when the reply is unusable.
    parse_retries: int = 2

    def __init__(
        self,
        runner,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: Callable[[str], None] | None = None,
        trace=None,
    ) -> None:
        self.runner = runner
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.on_delta = on_delta
        #: Optional decision log. The orchestrator passes the session's recorder so
        #: the "why" of every parse and repair sits next to the raw traffic.
        self.trace = trace
        self.last_run = AgentRun(stage=self.stage)
        self.last_result: ChatResult | None = None

    # ------------------------------------------------------------------ identity
    @property
    def info(self):
        return STAGE_INFO[self.stage]

    @property
    def title(self) -> str:
        return self.info.title

    # ------------------------------------------------------------------ prompt
    def needs(self, state) -> Sequence[str]:
        """Which artifacts this stage is allowed to read. Override as needed."""
        return ()

    @abstractmethod
    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        """The full system prompt for this stage."""

    @abstractmethod
    def parse(self, payload: Any, state) -> Any:
        """Coerce the model's JSON into the stage's artifact dataclass."""

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        """Task framing; extra context arrives through the system prompt."""
        return (
            f"Produce the {self.info.produces} now.\n"
            f"Stage: {self.info.title} - {self.info.role}.\n"
            f"Reply with one JSON object only."
        )

    # ------------------------------------------------------------------ execution
    def run(self, state, revision_notes: Sequence[str] = (), *, model: str = "") -> Any:
        """Execute the agent and return its artifact.

        On an unparseable or invalid reply the model is re-asked with the exact
        problems appended, up to ``parse_retries`` times. That repair loop is the
        difference between a demo and something that survives a 30B local model.
        """
        started = time.perf_counter()
        attempts = 0
        usage = ChatUsage()
        problems: list[str] = []
        raw = ""
        messages: list[ChatMessage | Mapping[str, Any]] = [
            {"role": "user", "content": self.user_message(state, revision_notes)}
        ]
        model_name = model or getattr(self.runner, "model", "")

        while attempts <= self.parse_retries:
            attempts += 1
            self._label_client(attempts)
            system = self.system_prompt(state, revision_notes)
            if problems:
                system = build_system(
                    system,
                    "YOUR PREVIOUS REPLY WAS REJECTED. Fix exactly these problems and "
                    "reply with the complete corrected JSON object:\n"
                    + "\n".join(f"- {p}" for p in problems),
                )
            try:
                payload, result = self._call(messages, system)
            except LLMError as exc:
                # A failure raised by the client itself - unparseable JSON, an HTTP
                # error, an empty reply - must still leave a decision behind.
                # Without this the log stopped mid-stage on exactly the runs someone
                # opens the log to understand, which is the same blind spot as the
                # original `exc` NameError in the UI: the error path was the one
                # nobody instrumented.
                self._decide(
                    attempts,
                    "repair" if attempts <= self.parse_retries else "failed",
                    f"permintaan gagal: {exc}",
                    problems=[str(exc)],
                    model=model_name,
                )
                if attempts > self.parse_retries:
                    raise AgentError(
                        f"{self.title} could not produce a usable artifact after "
                        f"{attempts} attempts: {exc}"
                    ) from exc
                problems = [f"Provider returned nothing usable: {exc}"]
                messages = [
                    {"role": "user", "content": self.user_message(state, revision_notes)}
                ]
                continue
            raw = result.text
            usage = ChatUsage(
                usage.prompt_tokens + result.usage.prompt_tokens,
                usage.completion_tokens + result.usage.completion_tokens,
                usage.total_tokens + result.usage.total_tokens,
            )
            try:
                artifact = self.parse(payload, state)
                issues = self.validate(artifact)
                self._decide(
                    attempts,
                    "reply",
                    f"balasan diterima dan di-parse ({len(raw)} karakter)",
                    model=model_name,
                )
                # One re-ask, not two. Live runs show this model class often omits
                # an *optional* field, and paying a second 7-minute generation to
                # recover it is a bad trade: a residual problem is recorded on the
                # run and the deterministic gates still judge the artifact.
                if issues and attempts == 1 and self.parse_retries > 0:
                    problems = issues
                    messages = [
                        {
                            "role": "user",
                            "content": self.user_message(state, revision_notes),
                        }
                    ]
                    self._decide(
                        attempts,
                        "repair",
                        f"artifact diterima tapi {len(issues)} masalah validasi; "
                        "minta satu putaran perbaikan",
                        problems=issues,
                        model=model_name,
                    )
                    continue
                artifact_problems = issues if issues else []
                if artifact_problems:
                    self._decide(
                        attempts,
                        "accepted_with_problems",
                        f"diterima setelah {attempts} percobaan dengan "
                        f"{len(artifact_problems)} catatan validasi yang tidak diperbaiki",
                        problems=artifact_problems,
                        model=model_name,
                    )
                else:
                    self._decide(
                        attempts,
                        "parsed",
                        f"artifact lolos validasi pada percobaan {attempts}",
                        model=model_name,
                    )
                self.last_run = AgentRun(
                    stage=self.stage,
                    model=model_name or getattr(result, "model", ""),
                    duration_s=time.perf_counter() - started,
                    usage=usage,
                    validation_problems=artifact_problems,
                    attempts=attempts,
                    raw=raw,
                )
                self.last_result = result
                return artifact
            except (ValueError, TypeError, AgentError) as exc:
                problems = [str(exc)]
                messages = [
                    {"role": "user", "content": self.user_message(state, revision_notes)}
                ]
                if attempts > self.parse_retries:
                    raise AgentError(
                        f"{self.title} could not produce a usable artifact after "
                        f"{attempts} attempts: {exc}"
                    ) from exc
        raise AgentError(f"{self.title} exhausted its retries.")

    def validate(self, artifact: Any) -> list[str]:
        """Structural problems worth re-asking about. Empty means acceptable."""
        if hasattr(artifact, "validate"):
            return list(artifact.validate())
        return []

    # ------------------------------------------------------------------ plumbing
    #: Output cap. **0 means no cap** - the request carries no ``max_tokens`` and the
    #: provider applies its own maximum.
    #:
    #: There is deliberately no floor any more. A floor is still a guess, and every
    #: guess here was wrong in the expensive direction: a 300-token cap returned 1032
    #: characters of reasoning with zero answer, 8192 truncated a blueprint mid-JSON,
    #: and 16384 was still not enough for a real blueprint (~15.7k tokens of output)
    #: once reasoning was added on top. A reasoning model can burn an arbitrary share
    #: of the budget before its first visible character, so the only setting that
    #: cannot silently truncate is the model's own maximum.
    DEFAULT_OUTPUT_TOKENS = 0

    def _label_client(self, attempt: int) -> None:
        """Tag the underlying client so the trace can attribute the exchange."""
        client = getattr(self.runner, "_client", None)
        if client is not None and hasattr(client, "label"):
            client.label = f"{self.stage}#{attempt}"

    def _decide(
        self,
        attempt: int,
        outcome: str,
        detail: str,
        *,
        problems: Sequence[str] = (),
        model: str = "",
    ) -> None:
        """Record what this agent concluded. Silent when no recorder is attached."""
        if self.trace is None:
            return
        try:
            self.trace.decide(
                stage=self.stage,
                attempt=attempt,
                outcome=outcome,
                detail=detail,
                problems=[str(p) for p in problems],
                model=model or getattr(self.runner, "model", ""),
            )
        except Exception:  # pragma: no cover - never break a run over logging
            pass

    def _call(self, messages, system: str) -> tuple[Any, ChatResult]:
        try:
            return self.runner.complete_json(
                messages,
                system=system,
                schema=self.schema,
                temperature=self.temperature,
                # 0 (or unset) means no cap: the field is omitted and the provider
                # uses its maximum. An explicit positive budget is honoured exactly,
                # because that is a deliberate decision - usually a cost one - and the
                # repair loop will raise it and say so if it turns out too small.
                max_tokens=int(self.max_tokens or 0),
                on_delta=self.on_delta,
                retries=1,
            )
        except LLMError:
            raise
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise AgentError(f"{self.title}: malformed JSON: {exc}") from exc

    def describe(self) -> str:
        return f"{self.info.title} [{self.stage}] -> {self.info.produces}"


def render_state_digest(state) -> str:
    """Compact one-block digest of everything available, for debug logging."""
    keys = [
        ("brief", state.brief),
        ("blueprint", state.blueprint),
        ("lyrics", state.lyrics),
        ("arrangement", state.arrangement),
        ("prompt", state.prompt),
    ]
    lines: list[str] = []
    for name, artifact in keys:
        if artifact is None:
            lines.append(f"{name}: (none)")
        elif hasattr(artifact, "summary"):
            lines.append(f"{name}: {artifact.summary()[:200]}...")
        else:
            lines.append(f"{name}: {type(artifact).__name__}")
    return "\n".join(lines)


__all__ = ["Agent", "AgentError", "AgentRun", "render_state_digest"]

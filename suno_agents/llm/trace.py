"""Structured record of everything the agents asked for and got back.

The app already emits coarse string events for the UI and a per-stage
:class:`~suno_agents.agents.base.AgentRun` summary. Neither is enough to answer
"what exactly did we send, what came back, and why did the agent decide that?" -
which is the question you actually have when a run produces something odd.

So this module keeps the raw traffic. One entry per HTTP attempt (including each
retry, because retries are where the interesting failures live) plus one entry per
agent decision. It lives beside the client that performs the calls, records only
what it is handed, and never raises: a logging bug must not be able to fail a run.

Two shapes, discriminated by ``kind``:

* :class:`LLMExchange` - one request/response pair, with the full prompt, the full
  reply, token usage, and the failure reason when there was one.
* :class:`AgentDecision` - what the agent did with that reply: parsed, needed a
  repair round, was accepted with recorded problems, or gave up.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

#: Cap on a single stored text field. Prompts for this app run to a few tens of
#: kilobytes; the cap exists so one pathological reply cannot balloon the run
#: outputs, and every truncated field says so.
MAX_FIELD_CHARS = 200_000

#: How many exchanges to keep in memory. A nine-stage run with retries produces
#: roughly 20-60, so this holds several runs before the oldest are dropped.
MAX_ENTRIES = 2_000


def _clip(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    """Bound a stored text field, marking it honestly when cut."""
    if text is None:
        return ""
    value = str(text)
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n[... {omitted} chars omitted ...]"


@dataclass
class LLMExchange:
    """One request to a provider and the reply that came back."""

    kind: str = "exchange"
    at: str = ""
    label: str = ""  # usually "<stage>#<attempt>"
    stage: str = ""
    model: str = ""
    provider: str = ""
    url: str = ""
    stream: bool = False
    duration_s: float = 0.0
    #: Full message list sent, already normalised (system first).
    messages: list[dict[str, str]] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: str = ""
    #: Reply text after reasoning blocks were stripped.
    response: str = ""
    #: Hidden reasoning, when the model produced any.
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    #: ok | truncated | empty | error | retried
    status: str = "ok"
    error: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            self.at = datetime.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------------ views
    @property
    def request_chars(self) -> int:
        return sum(len(m.get("content", "")) for m in self.messages)

    @property
    def is_problem(self) -> bool:
        return self.status != "ok"

    @property
    def system_prompt(self) -> str:
        for message in self.messages:
            if message.get("role") == "system":
                return message.get("content", "")
        return ""

    @property
    def user_message(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def headline(self) -> str:
        """One line for the list view."""
        name = self.stage or self.label or "-"
        tokens = f"{self.total_tokens} tok" if self.total_tokens else "no usage"
        return f"{self.at}  {name}  {self.model}  {self.duration_s:.1f}s  {tokens}  [{self.status}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecision:
    """What an agent concluded about a reply, and why."""

    kind: str = "decision"
    at: str = ""
    stage: str = ""
    attempt: int = 0
    #: reply | parsed | repair | accepted_with_problems | failed | gate | note
    outcome: str = ""
    detail: str = ""
    problems: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    model: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            self.at = datetime.now().strftime("%H:%M:%S")

    @property
    def is_problem(self) -> bool:
        """True when this decision is something a reader would want to look into.

        ``gate`` is included when it describes a rejection: a reviewer that failed
        the artifact and sent the pipeline backwards is *the* entry someone opens
        the log to find, so "problems" must not skip it.
        """
        if self.outcome == "gate":
            return not self.detail.startswith("pass ")
        return self.outcome in {"repair", "failed", "accepted_with_problems"}

    def headline(self) -> str:
        mark = {
            "reply": "balasan masuk",
            "parsed": "artifact diterima",
            "repair": "minta perbaikan",
            "accepted_with_problems": "diterima dengan catatan",
            "failed": "gagal",
            "gate": "keputusan gate",
            "note": "catatan",
        }.get(self.outcome, self.outcome)
        return f"{self.at}  {(self.stage or '-')}#{self.attempt}  {mark}  {self.detail[:60]}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TraceEntry = LLMExchange | AgentDecision


class TraceRecorder:
    """Thread-safe, bounded log of LLM traffic and agent decisions.

    Written from worker threads and read from the UI thread, so every mutation
    takes the lock. Recording is best-effort by design: it swallows its own errors
    rather than let a logging problem take down a run that is otherwise fine.
    """

    def __init__(self, *, max_entries: int = MAX_ENTRIES, enabled: bool = True) -> None:
        self._entries: list[TraceEntry] = []
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.enabled = enabled

    # ------------------------------------------------------------------ writing
    def record(self, entry: TraceEntry) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self._entries.append(entry)
                overflow = len(self._entries) - self.max_entries
                if overflow > 0:
                    del self._entries[:overflow]
        except Exception:  # pragma: no cover - logging must never break a run
            pass

    def record_exchange(self, **fields: Any) -> LLMExchange:
        exchange = LLMExchange(**fields)
        self.record(exchange)
        return exchange

    def decide(self, **fields: Any) -> AgentDecision:
        decision = AgentDecision(**fields)
        self.record(decision)
        return decision

    def note(self, message: str, *, stage: str = "", attempt: int = 0) -> AgentDecision:
        """A free-form marker, e.g. a run boundary or a provider warning."""
        return self.decide(stage=stage, attempt=attempt, outcome="note", detail=message)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    # ------------------------------------------------------------------ reading
    def entries(self) -> list[TraceEntry]:
        """A copy, so the UI can iterate without holding the lock."""
        with self._lock:
            return list(self._entries)

    def since(self, index: int) -> tuple[list[TraceEntry], int]:
        """Entries after ``index``, plus the new total. For incremental UI updates."""
        with self._lock:
            entries = self._entries[index:]
            return list(entries), len(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------ export
    def counts(self) -> dict[str, int]:
        tally = {"exchange": 0, "decision": 0, "problems": 0}
        for entry in self.entries():
            tally[entry.kind] = tally.get(entry.kind, 0) + 1
            if entry.is_problem:
                tally["problems"] += 1
        return tally

    def to_jsonl(self) -> str:
        lines = [json.dumps(entry.to_dict(), ensure_ascii=False) for entry in self.entries()]
        return "\n".join(lines) + ("\n" if lines else "")

    def to_markdown(self) -> str:
        """Human-readable transcript, for the run directory.

        Long fields are clipped here but kept in full in the JSONL: a reader wants
        the shape of the conversation, a machine wants every character.
        """
        out: list[str] = ["# Log agen", ""]
        for entry in self.entries():
            if entry.kind == "exchange":
                out.extend(_exchange_markdown(entry))  # type: ignore[arg-type]
            else:
                out.extend(_decision_markdown(entry))  # type: ignore[arg-type]
        return "\n".join(out) + "\n"


def _exchange_markdown(exchange: LLMExchange) -> list[str]:
    lines = [
        f"## {exchange.at} - {exchange.stage or exchange.label or 'request'} "
        f"({exchange.model})",
        "",
        f"- status: **{exchange.status}**"
        + (f" - {exchange.error}" if exchange.error else ""),
        f"- durasi: {exchange.duration_s:.1f}s"
        + f" | temperature: {exchange.temperature}"
        + f" | max_tokens: {exchange.max_tokens}"
        + f" | json_mode: {exchange.json_mode}"
        + f" | stream: {exchange.stream}",
        f"- permintaan: {exchange.request_chars} karakter"
        f" | balasan: {len(exchange.response)} karakter"
        + (f" | reasoning: {len(exchange.reasoning)} karakter" if exchange.reasoning else ""),
        f"- token: prompt {exchange.prompt_tokens}, completion "
        f"{exchange.completion_tokens}, total {exchange.total_tokens}"
        + (f" | finish: {exchange.finish_reason}" if exchange.finish_reason else ""),
        "",
    ]
    system = exchange.system_prompt
    if system:
        lines += ["### System prompt", "", "```text", _clip(system, 12_000), "```", ""]
    user = exchange.user_message
    if user:
        lines += ["### Pesan user", "", "```text", _clip(user, 4_000), "```", ""]
    if exchange.response:
        lines += ["### Balasan", "", "```text", _clip(exchange.response, 20_000), "```", ""]
    if exchange.reasoning:
        lines += ["### Reasoning tersembunyi", "", "```text",
                  _clip(exchange.reasoning, 4_000), "```", ""]
    return lines


def _decision_markdown(decision: AgentDecision) -> list[str]:
    lines = [
        f"## {decision.at} - keputusan {decision.stage}#{decision.attempt}: "
        f"{decision.outcome}",
        "",
        decision.detail,
        "",
    ]
    if decision.problems:
        lines += ["Catatan:", ""] + [f"- {p}" for p in decision.problems] + [""]
    return lines


def summarise(entries: Sequence[TraceEntry]) -> str:
    """Short human summary, used by the UI status line and the report."""
    tally = {"exchange": 0, "decision": 0, "problems": 0, "tokens": 0}
    for entry in entries:
        tally[entry.kind] = tally.get(entry.kind, 0) + 1
        if entry.is_problem:
            tally["problems"] += 1
        if entry.kind == "exchange":
            tally["tokens"] += entry.total_tokens  # type: ignore[union-attr]
    return (
        f"{tally['exchange']} permintaan, {tally['decision']} keputusan, "
        f"{tally['problems']} bermasalah, ~{tally['tokens']} token"
    )


def filter_entries(
    entries: Iterable[TraceEntry], mode: str = "all", stage: str = ""
) -> list[TraceEntry]:
    """Filter for the panel. ``mode`` is all | problems | exchanges | decisions."""
    out: list[TraceEntry] = []
    for entry in entries:
        if stage and entry.stage != stage:
            continue
        if mode == "problems" and not entry.is_problem:
            continue
        if mode == "exchanges" and entry.kind != "exchange":
            continue
        if mode == "decisions" and entry.kind != "decision":
            continue
        out.append(entry)
    return out


__all__ = [
    "AgentDecision",
    "LLMExchange",
    "MAX_ENTRIES",
    "MAX_FIELD_CHARS",
    "TraceEntry",
    "TraceRecorder",
    "filter_entries",
    "summarise",
    "time",
]

"""Session: the UI-facing wrapper around providers, drafts and the pipeline.

The Tkinter layer must never touch HTTP, threads, or dataclasses directly. It
talks to :class:`StudioSession`, which owns the settings, builds runners, keeps
the intake conversation, and can run the pipeline on a worker thread while
pushing events into a queue the UI drains.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..agents import build_agent
from ..config.models import AppSettings, ProviderConfig
from ..config.store import RunStore, SettingsStore
from ..domain.graph import COMPOSER, INTAKE, STAGE_INFO, STAGE_ORDER
from ..domain.models import CreativeBrief, PipelineState
from ..llm.client import LLMClient
from ..llm.runner import ClientRunner, NullRunner
from ..llm.trace import TraceRecorder
from ..llm.wire import ChatMessage
from .orchestrator import Orchestrator, PipelineEvent, PipelineResult


@dataclass
class ChatTurn:
    """One line in the user/consultant transcript."""

    role: str  # user | assistant | system | note
    text: str
    at: str = ""


@dataclass
class ReferenceBundle:
    """Everything the user attached, plus its analysis."""

    paths: list[str] = field(default_factory=list)
    summary: str = ""
    lossless: str = "unknown"
    quality: int = 0
    error: str = ""
    profile: Any = None

    @property
    def has_audio(self) -> bool:
        return bool(self.summary)


class StudioSession:
    """Owns configuration, the conversation, the audio analysis and the run."""

    def __init__(
        self,
        *,
        settings_store: SettingsStore | None = None,
        run_store: RunStore | None = None,
        on_event: Callable[[PipelineEvent], None] | None = None,
        on_chat: Callable[[ChatTurn], None] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
    ) -> None:
        self.settings_store = settings_store or SettingsStore()
        self.run_store = run_store or RunStore()
        self.settings: AppSettings = self.settings_store.load()
        self.on_event = on_event
        self.on_chat = on_chat
        self.on_delta = on_delta

        self.transcript: list[ChatTurn] = []
        self.brief: CreativeBrief | None = None
        self.reference = ReferenceBundle()
        self.orchestrator: Orchestrator | None = None
        self.result: PipelineResult | None = None
        #: Every request, reply and agent decision of the current session. Shared
        #: with clients (which record the traffic) and agents (which record the
        #: decisions), so the log tab and the run outputs tell the same story.
        self.trace = TraceRecorder()
        self.events: "queue.Queue[PipelineEvent]" = queue.Queue()
        self._thread: threading.Thread | None = None

        self.turn_budget = 24
        self.turn_count = 0

    # ------------------------------------------------------------------ providers
    def reload_settings(self) -> AppSettings:
        self.settings = self.settings_store.load()
        return self.settings

    def save_settings(self) -> Path:
        return self.settings_store.save(self.settings)

    def provider(self) -> ProviderConfig | None:
        return self.settings.active_provider

    def build_client(self, provider: ProviderConfig | None = None) -> LLMClient | None:
        provider = provider or self.provider()
        if provider is None:
            return None
        return LLMClient(
            provider, on_event=self._client_event, trace=self.trace, label=provider.name
        )

    def build_runner(self, provider: ProviderConfig | None = None):
        """Build the live runner, or the offline demo runner when asked."""
        if getattr(self.settings, "offline_demo", False) and provider is None:
            from ..testing.demo import build_demo_runner

            return build_demo_runner(trace=self.trace)
        client = self.build_client(provider)
        if client is None:
            return NullRunner()
        # settings is what makes per-stage routing real: without it ClientRunner
        # cannot answer for_stage() and every agent falls back to one model.
        return ClientRunner(client, settings=self.settings)

    def _client_event(self, kind: str, message: str) -> None:
        if kind == "warn":
            self.events.put(PipelineEvent(kind="warn", message=message))
        elif kind == "debug":
            self.events.put(PipelineEvent(kind="debug", message=message))

    def provider_status(self) -> str:
        if getattr(self.settings, "offline_demo", False):
            return "demo offline (agent tiruan)"
        provider = self.provider()
        if provider is None:
            return "Belum ada provider. Buka Pengaturan > Provider LLM."
        problems = provider.validate()
        if problems:
            return "Provider bermasalah: " + "; ".join(problems)
        routed = self.settings.routing_for(COMPOSER)
        distinct = {
            target.model
            for target in (self.settings.routing_for(s) for s in STAGE_ORDER)
            if target.model
        }
        suffix = f" \u00b7 {len(distinct)} model" if len(distinct) > 1 else ""
        return f"{routed.provider_name} \u00b7 {routed.model}{suffix}"

    def routing_overview(self) -> dict[str, str]:
        """Per-stage "what will actually run", for the settings dialog and status.

        The dialog shows this instead of the raw override, so what the user reads is
        resolution of the same method the orchestrator calls.
        """
        return {stage: self.settings.routing_for(stage).describe() for stage in STAGE_ORDER}

    # ------------------------------------------------------------------ chat
    def add_turn(self, turn: ChatTurn) -> None:
        self.transcript.append(turn)
        if self.on_chat is not None:
            try:
                self.on_chat(turn)
            except Exception:  # pragma: no cover - UI callbacks must not break a run
                pass

    def send_user_message(self, text: str) -> Mapping[str, Any]:
        """One intake turn: returns the reply plus the updated brief."""
        text = text.strip()
        if not text:
            raise ValueError("Pesan kosong.")
        self.add_turn(ChatTurn(role="user", text=text))
        if self.turn_count >= self.turn_budget:
            # Keep the conversation finite: past the budget we ask for closure.
            text += (
                "\n\n[Catatan sistem: batas percakapan tercapai. Kunci brief sekarang "
                "dengan asumsi profesionalmu dan set ready_to_build=true.]"
            )
        runner = self.build_runner()
        agent = build_agent(
            INTAKE,
            runner,
            temperature=self.settings.effective_temperature(INTAKE),
            on_delta=(lambda chunk: self.on_delta(INTAKE, chunk)) if self.on_delta else None,
            trace=self.trace,
        )
        state = PipelineState(
            brief=self.brief,
            reference_summary=self.reference.summary,
            reference_lossless=self.reference.lossless,
            reference_quality=self.reference.quality,
        )
        self.turn_count += 1
        payload = agent.run(state, [f"User: {text}"])
        self.brief = payload["brief"]
        self.add_turn(ChatTurn(role="assistant", text=payload["reply"]))
        return payload

    def seed_conversation(self, opening: str) -> None:
        """Start a session with the consultant's opening line."""
        self.add_turn(ChatTurn(role="assistant", text=opening))

    def conversation_messages(self) -> list[ChatMessage]:
        """The transcript as chat messages, for a provider that needs history."""
        return [
            ChatMessage(role="user" if t.role == "user" else "assistant", content=t.text)
            for t in self.transcript
            if t.role in {"user", "assistant"}
        ]

    # ------------------------------------------------------------------ audio
    def analyse_references(self, paths: Sequence[str]) -> ReferenceBundle:
        """Measure the attached files and store the LLM-readable summary.

        The audio engine is imported here rather than at module scope: measuring
        needs numpy and ffmpeg, and a machine without either must still be able to
        open the studio and produce a prompt.
        """
        cleaned = [str(p) for p in paths if str(p).strip()]
        self.reference.paths = cleaned
        if not cleaned:
            self.reference = ReferenceBundle()
            return self.reference

        try:
            from ..audio import build_profile
        except Exception as exc:  # noqa: BLE001 - numpy/ffmpeg missing
            self.reference = ReferenceBundle(
                paths=cleaned,
                error=(
                    "Mesin analisa audio tidak tersedia "
                    f"({type(exc).__name__}: {exc}). Pastikan numpy dan ffmpeg terpasang "
                    "dan ada di PATH."
                ),
            )
            return self.reference

        try:
            profile = build_profile(cleaned)
        except Exception as exc:  # noqa: BLE001 - shown verbatim in the UI
            self.reference = ReferenceBundle(
                paths=cleaned, error=f"{type(exc).__name__}: {exc}"
            )
            return self.reference

        self.reference = ReferenceBundle(
            paths=cleaned,
            summary=profile.summary,
            lossless=profile.lossless_verdict,
            quality=profile.quality_score,
            profile=profile,
        )
        return self.reference

    # ------------------------------------------------------------------ pipeline
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_run(self, brief: CreativeBrief | None = None, *, run_dir: Path | str | None = None):
        """Start the pipeline on a worker thread; events arrive in ``self.events``."""
        if self.running:
            raise RuntimeError("Pipeline sedang berjalan.")
        # No brief yet? Pass None through and let the intake stage earn one,
        # rather than inventing an empty brief that merely skips the consultant.
        effective_brief = brief or self.brief
        title = effective_brief.working_title if effective_brief else "untitled"
        runner = self.build_runner()
        target = Path(run_dir) if run_dir else self.run_store.new_dir(title or "untitled")
        self.orchestrator = Orchestrator(
            runner,
            settings=self.settings,
            on_event=self._pipeline_event,
            on_delta=self.on_delta,
            run_dir=target,
            trace=self.trace,
        )
        self.trace.note(
            f"Pipeline dimulai (brief: {(effective_brief.working_title if effective_brief else '-') or '-'})"
        )

        def worker() -> None:
            try:
                self.result = self.orchestrator.run(
                    effective_brief,
                    reference_summary=self.reference.summary,
                    reference_lossless=self.reference.lossless,
                    reference_quality=self.reference.quality,
                )
                self.orchestrator.save_outputs(target)
            except Exception as exc:  # noqa: BLE001 - reported through the event queue
                self.events.put(
                    PipelineEvent(kind="error", message=f"{type(exc).__name__}: {exc}")
                )
            finally:
                self.events.put(PipelineEvent(kind="thread_done", message=str(target)))

        self._thread = threading.Thread(target=worker, name="lirik-pipeline", daemon=True)
        self._thread.start()
        return target

    def _pipeline_event(self, event: PipelineEvent) -> None:
        self.events.put(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # pragma: no cover - UI callbacks must not break a run
                pass

    def stop_run(self) -> None:
        if self.orchestrator is not None:
            self.orchestrator.stop()

    def drain_events(self, limit: int = 200) -> list[PipelineEvent]:
        out: list[PipelineEvent] = []
        for _ in range(limit):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    def wait_for_run(self, timeout: float | None = None) -> PipelineResult | None:
        """Block until the worker finishes (used by tests and the CLI)."""
        if self._thread is not None:
            self._thread.join(timeout)
        return self.result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def stage_title(stage: str) -> str:
        return STAGE_INFO[stage].title if stage in STAGE_INFO else stage

    @staticmethod
    def stage_list() -> tuple[str, ...]:
        return STAGE_ORDER

    def new_run_dir(self, title: str) -> Path:
        return self.run_store.new_dir(title)

    def history(self):
        return self.run_store.list_runs()


__all__ = ["StudioSession", "ChatTurn", "ReferenceBundle"]

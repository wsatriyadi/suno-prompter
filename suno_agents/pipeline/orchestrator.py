"""The pipeline orchestrator: nine stages, three reviewers, real rollbacks.

The control loop is deliberately boring so it stays auditable:

1. Run the stage.
2. If it is a reviewer, compute the deterministic gates and merge them with the
   model's opinion into one binding verdict (:mod:`suno_agents.quality.verdict`).
3. Pass -> advance. Fail -> jump back to the verdict's ``restart_from`` stage and
   re-run from there, feeding the exact fix list forward.
4. Stop when the final reviewer passes, when the revision budget is exhausted, or
   when the loop can no longer legally roll back.

Every decision is emitted as an event so the UI can show the loop live, and the
whole :class:`~suno_agents.domain.models.PipelineState` is checkpointed after each
stage, so a long run is inspectable and resumable instead of disposable.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..agents import build_agent
from ..agents.analyst import AudioAnalystAgent
from ..domain.graph import (
    ARRANGER,
    AUDIO_ANALYST,
    COMPOSER,
    INTAKE,
    LYRICIST,
    LYRIC_REVIEWER,
    MUSIC_REVIEWER,
    PROMPT_ENGINEER,
    PROMPT_REVIEWER,
    STAGE_INFO,
    STAGE_ORDER,
    is_reviewer,
    legal_restarts,
    resolve_restart,
    stage_index,
)
from ..domain.models import (
    PipelineState,
    ReviewVerdict,
    StageRecord,
    SunoPrompt,
)
from ..llm.wire import ChatUsage, LLMError
from ..quality.gates import (
    PASS_MARKS,
    gate_report,
    run_lyric_gates,
    run_music_gates,
    run_prompt_gates,
)
from ..quality.verdict import (
    failure_keys,
    finalise_review,
    gate_score,
    revision_notes_from,
)

EventHook = Callable[["PipelineEvent"], None]
DeltaHook = Callable[[str, str], None]


@dataclass
class PipelineEvent:
    """One thing that happened, for the UI, the transcript and the log."""

    kind: str  # run_start | stage_start | stage_done | gates | review | rollback | warn | error | finish
    stage: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            self.at = datetime.now().strftime("%H:%M:%S")


@dataclass
class PipelineResult:
    """What the caller gets back when the loop stops."""

    state: PipelineState
    prompt: SunoPrompt | None = None
    passed: bool = False
    stopped_reason: str = ""
    verdict: ReviewVerdict | None = None

    @property
    def report(self) -> str:
        return self.state.final_report


class Orchestrator:
    """Runs the nine-stage loop for one song."""

    def __init__(
        self,
        runner,
        *,
        settings=None,
        on_event: EventHook | None = None,
        run_dir: Path | str | None = None,
        on_delta: DeltaHook | None = None,
        trace=None,
    ) -> None:
        self.runner = runner
        self.settings = settings
        self.on_event = on_event
        self.on_delta = on_delta
        #: Optional agent log; forwarded to every agent built by this orchestrator.
        self.trace = trace
        self.run_dir = Path(run_dir) if run_dir else None
        self.state = PipelineState()
        self._stop = threading.Event()
        self._agents: dict[str, Any] = {}
        self._raw_verdicts: dict[str, ReviewVerdict] = {}

    # ------------------------------------------------------------------ config
    @property
    def max_revisions(self) -> int:
        return int(getattr(self.settings, "max_revisions", 24) or 24)

    @property
    def stage_attempts(self) -> int:
        return max(1, int(getattr(self.settings, "stage_attempts", 3) or 3))

    def stop(self) -> None:
        """Ask the loop to stop before the next stage starts."""
        self._stop.set()

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, stage: str = "", message: str = "", **data) -> None:
        event = PipelineEvent(kind=kind, stage=stage, message=message, data=data)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # pragma: no cover - the UI must never break a run
                pass

    # ------------------------------------------------------------------ agents
    def agent_for(self, stage: str):
        """Build (once) and configure the agent for ``stage``."""
        if stage in self._agents:
            return self._agents[stage]
        temperature = None
        max_tokens = None
        if self.settings is not None:
            target = self.settings.routing_for(stage)
            temperature = target.temperature
            max_tokens = target.max_tokens
        # Per-stage routing: the runner resolves this from AppSettings.routing_for,
        # the same method the settings dialog displays. A runner that cannot route
        # returns None and the stage inherits the active provider.
        runner = self.runner
        routed = getattr(self.runner, "for_stage", None)
        if callable(routed):
            replacement = routed(stage)
            if replacement is not None:
                runner = replacement
        delta = None
        if self.on_delta is not None:
            delta = lambda chunk, _stage=stage: self.on_delta(_stage, chunk)  # noqa: E731
        agent = build_agent(
            stage,
            runner,
            temperature=temperature,
            max_tokens=max_tokens,
            on_delta=delta,
            trace=self.trace,
        )
        self._agents[stage] = agent
        return agent

    # ------------------------------------------------------------------ main loop
    def run(
        self,
        brief=None,
        *,
        reference_summary: str = "",
        reference_lossless: str = "unknown",
        reference_quality: int = 0,
    ) -> PipelineResult:
        """Execute the full pipeline and return the final prompt plus the state."""
        self._stop.clear()
        self._agents.clear()
        self._raw_verdicts.clear()
        self.state = PipelineState(
            run_id=datetime.now().strftime("%Y%m%d-%H%M%S"),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            provider=str(getattr(self.runner, "model", "")),
            brief=brief,
            reference_summary=reference_summary,
            reference_lossless=reference_lossless,
            reference_quality=reference_quality,
        )
        self._emit("run_start", message="Pipeline dimulai", stages=list(STAGE_ORDER))

        index = 0
        revision_notes: list[str] = []
        last_verdict: ReviewVerdict | None = None
        stopped = ""

        while index < len(STAGE_ORDER):
            if self._stop.is_set():
                stopped = "Dihentikan oleh pengguna."
                break

            stage = STAGE_ORDER[index]
            if stage == INTAKE and self.state.brief is not None:
                # The caller already holds a brief: there is no conversation to have.
                index += 1
                continue

            try:
                self._run_stage(stage, revision_notes)
            except LLMError as exc:
                self._emit("error", stage=stage, message=f"Kesalahan provider: {exc}")
                stopped = f"Provider gagal di stage {stage}: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - surfaced in the report
                self._emit("error", stage=stage, message=f"{type(exc).__name__}: {exc}")
                stopped = f"Stage {stage} gagal: {exc}"
                break

            if not is_reviewer(stage):
                revision_notes = []
                index += 1
                self._checkpoint()
                continue

            verdict = self._merged_verdict(stage)
            last_verdict = verdict

            if verdict.passed:
                self._emit(
                    "review",
                    stage=stage,
                    message=f"LULUS ({verdict.overall_score:.1f}/100)",
                    verdict=verdict.to_dict(),
                )
                revision_notes = []
                index += 1
                self._checkpoint()
                if stage == PROMPT_REVIEWER:
                    stopped = "Selesai: prompt lulus review akhir."
                continue

            target = self._rollback_target(stage, verdict)
            if not target:
                self._emit(
                    "review",
                    stage=stage,
                    message=(
                        f"GAGAL ({verdict.overall_score:.1f}/100) dan tidak ada tahap "
                        "sebelumnya yang bisa diperbaiki."
                    ),
                    verdict=verdict.to_dict(),
                )
                stopped = (
                    f"{STAGE_INFO[stage].title} menolak hasil dan tidak ada target "
                    "mundur yang legal tersisa."
                )
                break

            if self.state.revisions >= self.max_revisions:
                stopped = (
                    f"Batas revisi ({self.max_revisions}) tercapai di "
                    f"{STAGE_INFO[stage].title}. Karya terbaik tetap dilampirkan "
                    "beserta temuan reviewer."
                )
                self._emit("warn", stage=stage, message=stopped)
                break

            # A defect that survives two rewrites of the same stage will not be
            # solved by a third: the reviewer keeps naming the same target, the
            # target keeps producing the same flaw, and every round costs minutes.
            # Stopping here with the best artifact beats burning the whole budget -
            # a live run spent three full regenerations on one gate that could never
            # be satisfied, and the honest outcome is to say so and deliver.
            if self._repeat_count(target, verdict) >= 2:
                stopped = (
                    f"{STAGE_INFO[target].title} sudah tiga kali menerima masalah yang "
                    f"sama tanpa menyelesaikannya. Loop dihentikan agar tidak membuang "
                    f"waktu; karya terbaik dilampirkan dengan temuan reviewer."
                )
                self._emit(
                    "warn",
                    stage=stage,
                    message=stopped,
                    repeated=[
                        key
                        for key in failure_keys(verdict)
                        if self._repeat_count(target, verdict, key=key) >= 2
                    ],
                )
                break

            self.state.revisions += 1
            revision_notes = revision_notes_from(
                verdict, for_stage=target, history=self.state.review_log
            )
            is_repeat = any(note.startswith("THIS IS A REPEAT") for note in revision_notes)
            attempts_at_target = sum(
                1 for record in self.state.history if record.stage == target
            )
            self._emit(
                "rollback",
                stage=stage,
                message=(
                    f"GAGAL ({verdict.overall_score:.1f}/100) \u2192 mundur ke "
                    f"{STAGE_INFO[target].title} (revisi {self.state.revisions}"
                    + (
                        f", percobaan ke-{attempts_at_target + 1} di tahap itu"
                        if attempts_at_target
                        else ""
                    )
                    + ")"
                    + (" \u00b7 MENGULANG masalah yang sama" if is_repeat else "")
                ),
                target=target,
                gates_failed=list(verdict.gates_failed),
                issues=[i.to_dict() for i in verdict.sorted_issues()[:8]],
                verdict=verdict.to_dict(),
                repeat=is_repeat,
                attempts_at_target=attempts_at_target,
            )
            index = stage_index(target)
            self._checkpoint()

        self.state.finished = True
        self.state.success = bool(
            self.state.prompt is not None
            and self.state.prompt_review is not None
            and self.state.prompt_review.passed
        )
        self.state.final_report = self.build_report(stopped)
        self._checkpoint()
        self._emit(
            "finish",
            message=stopped or "Selesai",
            success=self.state.success,
            revisions=self.state.revisions,
        )
        return PipelineResult(
            state=self.state,
            prompt=self.state.prompt,
            passed=self.state.success,
            stopped_reason=stopped,
            verdict=last_verdict,
        )

    # ------------------------------------------------------------------ stages
    def _run_stage(self, stage: str, revision_notes: Sequence[str]) -> None:
        """Run one stage, apply its artifact, and record the trace."""
        started = time.perf_counter()
        self._emit(
            "stage_start",
            stage=stage,
            message=STAGE_INFO[stage].title,
            pass_number=self.state.revisions + 1,
        )
        agent = self.agent_for(stage)
        attempt = 0
        notes = list(revision_notes)
        while True:
            attempt += 1
            try:
                artifact = agent.run(self.state, notes)
                break
            except Exception as exc:  # noqa: BLE001 - retried, then propagated
                if attempt >= self.stage_attempts:
                    raise
                self._emit(
                    "warn",
                    stage=stage,
                    message=(
                        f"{STAGE_INFO[stage].title}: percobaan {attempt} gagal "
                        f"({type(exc).__name__}: {exc}) - mengulang."
                    ),
                )
                notes = list(notes) + [f"Previous attempt failed: {exc}"]

        duration = time.perf_counter() - started
        run = getattr(agent, "last_run", None)
        usage = getattr(run, "usage", None) or ChatUsage()
        self._apply(stage, artifact)
        summary = self._summarise(stage, artifact)
        self.state.history.append(
            StageRecord(
                stage=stage,
                index=stage_index(stage),
                attempt=attempt,
                started_at=datetime.now().strftime("%H:%M:%S"),
                duration_s=round(duration, 2),
                model=getattr(run, "model", "") if run else "",
                status="done",
                summary=summary,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )
        self._emit(
            "stage_done",
            stage=stage,
            message=summary,
            duration=round(duration, 2),
            tokens=usage.total_tokens,
        )

    def _apply(self, stage: str, artifact: Any) -> None:
        """Store a stage's artifact, with the cross-stage post-processing it needs."""
        if stage == INTAKE:
            self.state.brief = artifact["brief"]
        elif stage == AUDIO_ANALYST:
            analyst = self._agents.get(stage)
            if isinstance(artifact, Mapping) and analyst is not None:
                addition = analyst.summary_block(self.state, artifact)
                self.state.reference_summary = (
                    f"{self.state.reference_summary}\n{addition}".strip()
                )
        elif stage == COMPOSER:
            self.state.blueprint = artifact
        elif stage == LYRICIST:
            self.state.lyrics = artifact
        elif stage == ARRANGER:
            self.state.arrangement = artifact
        elif stage == PROMPT_ENGINEER:
            self.state.prompt = artifact
            self._attach_reference_caveats(artifact)
            self._attach_brief_caveats(artifact)
        elif is_reviewer(stage):
            # Held until the deterministic gates are merged in.
            self._raw_verdicts[stage] = artifact

    def _attach_reference_caveats(self, prompt: SunoPrompt) -> None:
        """Append the honest lossless notes the user must see next to the prompt."""
        notes: list[str] = []
        verdict = self.state.reference_lossless
        if verdict in {"lossy", "lossy-transcoded", "mixed"}:
            notes.append(
                f"Referensi audio terdeteksi '{verdict}'. Prompt ini tidak menjanjikan "
                "high-end yang sudah rusak; mastering tetap dilakukan dari render WAV Suno."
            )
        if self.state.reference_quality and self.state.reference_quality < 60:
            notes.append(
                f"Skor kualitas referensi {self.state.reference_quality}/100 - jangan "
                "meniru loudness/kompresinya, ambil hanya ide aransemennya."
            )
        if notes:
            prompt.provider_notes = f"{prompt.provider_notes}\n" + "\n".join(notes)
            prompt.provider_notes = prompt.provider_notes.strip()
            prompt.lossless_checklist = list(prompt.lossless_checklist) + [
                "Mulai selalu dari render WAV Suno, bukan dari file referensi yang lossy."
            ]

    def _attach_brief_caveats(self, prompt: SunoPrompt) -> None:
        """Be explicit when the song was built on a brief the user never confirmed.

        A headless run has nobody to answer the consultant's questions, so the
        intake agent fills the gaps with labelled placeholders. Proceeding is the
        right call - half a song beats no song - but shipping it silently as if the
        user had chosen those details is not, because their own words are the one
        input this whole pipeline exists to serve.
        """
        brief = self.state.brief
        if brief is None:
            return
        uncertain = brief.confidence < 0.35
        if not (brief.open_questions or uncertain):
            return
        questions = "; ".join(brief.open_questions[:3])
        note = (
            "Brief ini belum dikonfirmasi user"
            + (f" (keyakinan {brief.confidence:.2f})" if uncertain else "")
            + (
                f". Pertanyaan yang masih terbuka: {questions}."
                if questions
                else ". Semua field kreatif masih berupa usulan, bukan preferensi user."
            )
            + " Anggap prompt ini draf yang kuat, bukan final: jawab pertanyaannya di "
            "panel konsultasi lalu jalankan ulang untuk mengunci arahnya."
        )
        # Idempotent: a rollback re-runs this stage, and stacking the same warning
        # twice would make the report look like two separate problems.
        if note not in self.state.warnings:
            self.state.warnings.append(note)
        prompt.provider_notes = f"{prompt.provider_notes}\n{note}".strip()
        prompt.meta_notes = f"{prompt.meta_notes}\n\n[PERLU KONFIRMASI] {note}".strip()

    def _merged_verdict(self, stage: str) -> ReviewVerdict:
        """Merge the model's review with the deterministic gates for ``stage``."""
        llm = self._raw_verdicts.get(stage)
        gates = self._run_gates(stage)
        # history already contains this stage's record, so subtract one.
        repeats = max(0, self._stage_runs(stage) - 1)
        verdict = finalise_review(
            stage, llm, gates, pass_mark=PASS_MARKS[stage], attempt=repeats
        )
        if stage == LYRIC_REVIEWER:
            self.state.lyric_review = verdict
        elif stage == MUSIC_REVIEWER:
            self.state.music_review = verdict
        else:
            self.state.prompt_review = verdict
        self._raw_verdicts.pop(stage, None)
        self.state.review_log.append(verdict)
        if self.trace is not None:
            try:
                hard = [g for g in gates if not g.ok and not g.soft]
                soft = [g for g in gates if not g.ok and g.soft]
                self.trace.decide(
                    stage=stage,
                    attempt=repeats + 1,
                    outcome="gate",
                    detail=(
                        # Leading token is machine-read by the log filter, so it
                        # must stay exactly "pass"/"revise"/"reject".
                        f"{verdict.verdict} {verdict.overall_score:.1f}/100 - "
                        f"{len(hard)} gate keras gagal, {len(soft)} gate lunak"
                        + (f"; mundur ke {verdict.restart_from}" if verdict.restart_from else "")
                    ),
                    problems=[g.message for g in hard] + [g.message for g in soft],
                )
            except Exception:  # pragma: no cover - never break a run over logging
                pass
        self._emit(
            "gates",
            stage=stage,
            message=gate_report(gates),
            failures=[g.code for g in gates if not g.ok],
            score=gate_score(gates),
        )
        return verdict

    def _stage_runs(self, stage: str) -> int:
        return sum(1 for h in self.state.history if h.stage == stage)

    def _run_gates(self, stage: str):
        if stage == LYRIC_REVIEWER:
            if self.state.lyrics is None:
                return []
            return run_lyric_gates(self.state.lyrics, self.state.brief, self.state.blueprint)
        if stage == MUSIC_REVIEWER:
            if self.state.blueprint is None or self.state.arrangement is None:
                return []
            return run_music_gates(
                self.state.blueprint,
                self.state.arrangement,
                self.state.brief,
                self.state.lyrics,
            )
        if self.state.prompt is None:
            return []
        return run_prompt_gates(
            self.state.prompt,
            self.state.brief,
            self.state.lyrics,
            self.state.reference_lossless,
        )

    def _repeat_count(
        self, target: str, verdict: ReviewVerdict, *, key: str = ""
    ) -> int:
        """How many times this target has already been sent back for the same defect.

        Counted from the review log, which survives rollbacks, so it measures actual
        repetition rather than attempts on the current artifact.
        """
        keys = {key} if key else failure_keys(verdict)
        if not keys:
            return 0
        seen = 0
        for earlier in self.state.review_log:
            if earlier is verdict:
                continue
            if earlier.restart_from and earlier.restart_from != target:
                continue
            if keys & failure_keys(earlier):
                seen += 1
        return seen

    def _rollback_target(self, stage: str, verdict: ReviewVerdict) -> str:
        """Validate a verdict's rollback request against the stage graph.

        When the reviewer's request is unusable, the loop escalates one legal
        target at a time as attempts accumulate, so a stuck run eventually walks
        back to the brief instead of spinning on the same stage forever.
        """
        legal = legal_restarts(stage)
        if not legal:
            return ""
        if verdict.restart_from:
            target = resolve_restart(stage, verdict.restart_from)
            if target:
                return target
        repeats = self._stage_runs(stage)
        return legal[min(max(0, repeats - 1), len(legal) - 1)]

    # ------------------------------------------------------------------ output
    def _summarise(self, stage: str, artifact: Any) -> str:
        if stage == INTAKE:
            return str(artifact["reply"])[:400]
        if is_reviewer(stage):
            return (
                f"{artifact.verdict} {artifact.overall_score:.1f}/100 - "
                f"{(artifact.reasoning or '')[:300]}"
            )
        if hasattr(artifact, "summary"):
            return str(artifact.summary())[:400]
        if isinstance(artifact, Mapping):
            return str(artifact.get("interpretation", ""))[:400] or "analisis referensi"
        return str(artifact)[:400]

    def build_report(self, stopped: str = "") -> str:
        """Human-readable run report, saved next to the prompt."""
        state = self.state
        lines = [
            "LIRIK STUDIO - LAPORAN RUN",
            f"Run ID: {state.run_id} | Dibuat: {state.created_at}",
            f"Model: {state.provider or '-'}",
            f"Status: {'LULUS' if state.success else 'BELUM LULUS LENGKAP'}",
            f"Jumlah revisi: {state.revisions}",
        ]
        if stopped:
            lines.append(f"Catatan berhenti: {stopped}")
        if state.warnings:
            lines += ["", "== PERINGATAN =="]
            lines += [f"- {warning}" for warning in state.warnings]
        if state.brief is not None:
            lines += ["", "== BRIEF ==", state.brief.summary()]
        if state.reference_summary:
            lines += ["", "== REFERENSI AUDIO ==", state.reference_summary]
        if state.blueprint is not None:
            lines += ["", "== MUSIK (nada) ==", state.blueprint.summary()]
        if state.lyrics is not None:
            lines += [
                "",
                "== LIRIK & VOKAL ==",
                f"Judul kerja: {state.lyrics.working_title or '-'}",
                f"Vokal: {state.lyrics.vocal.describe()}",
                state.lyrics.as_plain_lyrics(),
            ]
        if state.arrangement is not None:
            lines += ["", "== ARANSEMEN (lirik x nada) ==", state.arrangement.summary()]
        for label, verdict in (
            ("REVIEW LIRIK", state.lyric_review),
            ("REVIEW MUSIK", state.music_review),
            ("REVIEW PROMPT", state.prompt_review),
        ):
            if verdict is None:
                continue
            lines += [
                "",
                f"== {label} ==",
                f"Verdict: {verdict.verdict} | Skor: {verdict.overall_score:.1f}/100 | "
                f"Gate gagal: {', '.join(verdict.gates_failed) or '-'}",
                verdict.reasoning[:2000],
            ]
            if verdict.issues:
                lines.append("Isu:")
                lines += [
                    f"  [{i.severity}] {i.message}" + (f" -> {i.fix}" if i.fix else "")
                    for i in verdict.sorted_issues()[:12]
                ]
        repeats = _repeat_summary(state)
        if repeats:
            lines += ["", "== PENGULANGAN MASALAH =="]
            lines += repeats
        if state.history:
            lines += ["", "== JEJAK STAGE =="]
            for record in state.history:
                tokens = record.prompt_tokens + record.completion_tokens
                lines.append(
                    f"{record.started_at} {record.stage:17s} attempt {record.attempt} "
                    f"{record.duration_s:6.1f}s tokens~{tokens}"
                )
        if state.prompt is not None:
            lines += [
                "",
                "== PROMPT SUNO ==",
                f"Budget: {state.prompt.character_budget}",
                state.prompt.to_suno_block(),
            ]
            if state.prompt.lossless_checklist:
                lines += ["", "== CHECKLIST LOSSLESS =="]
                lines += [f"- {item}" for item in state.prompt.lossless_checklist]
        return "\n".join(lines)

    def output_files(self) -> dict[str, str]:
        """The files that make a run self-contained."""
        state = self.state
        files = {
            "report.md": self.build_report(state.final_report),
            "state.json": json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
        }
        if state.prompt is not None:
            files.update(
                {
                    "title.txt": state.prompt.title,
                    "style.txt": state.prompt.style,
                    "exclude.txt": state.prompt.exclude_style_text,
                    "lyrics.txt": state.prompt.lyrics,
                    "suno_prompt.txt": state.prompt.to_suno_block(),
                }
            )
            if state.prompt.lossless_checklist:
                files["lossless_checklist.md"] = "\n".join(
                    f"- {item}" for item in state.prompt.lossless_checklist
                )
        if state.lyrics is not None:
            files["lyrics_full.md"] = state.lyrics.as_plain_lyrics()
        if state.blueprint is not None:
            files["music_blueprint.md"] = state.blueprint.summary()
        if state.arrangement is not None:
            files["arrangement.md"] = state.arrangement.summary()
        if state.brief is not None:
            files["brief.md"] = state.brief.summary()
        if state.prompt is not None and state.prompt.provider_notes:
            files["provider_notes.md"] = state.prompt.provider_notes
        files["review_log.json"] = json.dumps(
            [v.to_dict() for v in state.review_log], indent=2, ensure_ascii=False
        )
        if self.trace is not None and len(self.trace):
            # Two forms on purpose: JSONL keeps every character for machines, the
            # markdown is for a human reading back what the agents actually did.
            files["agent_log.jsonl"] = self.trace.to_jsonl()
            files["agent_log.md"] = self.trace.to_markdown()
        return files

    def _checkpoint(self) -> None:
        """Persist the state, so a crash or a closed window does not lose a run."""
        if self.run_dir is None:
            return
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "state.json").write_text(
                json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:  # pragma: no cover - disk full / permissions
            pass

    def save_outputs(self, directory: Path | str | None = None) -> Path:
        """Write every artifact of the finished run into ``directory``."""
        target = Path(directory) if directory else self.run_dir
        if target is None:
            raise ValueError("No output directory configured for this run.")
        target.mkdir(parents=True, exist_ok=True)
        for name, content in self.output_files().items():
            (target / name).write_text(content, encoding="utf-8")
        return target


def _repeat_summary(state) -> list[str]:
    """Defects that reviewers reported more than once, for the run report."""
    seen: dict[str, list[str]] = {}
    for verdict in state.review_log:
        for key in failure_keys(verdict):
            seen.setdefault(key, []).append(verdict.stage)
    out: list[str] = []
    for key, stages in sorted(seen.items()):
        if len(stages) > 1:
            out.append(
                f"- {key} dilaporkan {len(stages)}x (oleh {', '.join(sorted(set(stages)))}) "
                "-> rewrite sebelumnya tidak menyelesaikannya"
            )
    return out


__all__ = ["Orchestrator", "PipelineEvent", "PipelineResult"]

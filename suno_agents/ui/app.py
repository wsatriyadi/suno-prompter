"""Main window: rail, conversation, inspector, prompt, status line.

Layout is a three-zone console:

    +--------------------------------------------------------------+
    | header: identity            provider state      settings      |
    +--------------+---------------------------+-------------------+
    | stage rail   | conversation              | inspector / prompt|
    | (signal path)| (konsultan berbahasa user)| (artifact + review)|
    +--------------+---------------------------+-------------------+
    | status: stage, timer, tokens, revision counter                |
    +--------------------------------------------------------------+

The right zone is a notebook of two views: "Prompt Suno" (the deliverable) and
"Inspeksi" (per-stage artifacts and reviewer scorecards).
"""

from __future__ import annotations

import os
import queue
import threading
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from ..config.models import AppSettings
from ..config.store import RunStore, SettingsStore, user_config_dir
from ..domain.graph import STAGE_INFO, STAGE_ORDER
from ..domain.models import PipelineState
from ..pipeline.orchestrator import PipelineEvent
from ..pipeline.session import ChatTurn, ReferenceBundle, StudioSession
from .agentlog import AgentLogPanel
from .chat import ChatPanel
from .settings import SettingsDialog
from .studio import PromptPanel, ReferencePanel, StageInspector
from .theme import Metrics, Palette, build_fonts, resolve_typography, truncate
from .widgets import (
    AutoWrapLabel,
    Badge,
    Divider,
    IconButton,
    PillButton,
    StageRail,
)

M = Metrics()

AUDIO_TYPES = [
    ("Audio", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.oga *.opus *.aiff *.aif *.wma"),
    ("MP3", "*.mp3"),
    ("WAV", "*.wav"),
    ("M4A/AAC", "*.m4a *.aac"),
    ("Lossless", "*.flac *.aiff *.aif"),
    ("Semua file", "*.*"),
]

WELCOME = (
    "Halo. Aku konsultan musik di studio ini - kamu tidak perlu tahu istilah teknis.\n\n"
    "Ceritakan saja lagu yang kamu bayangkan: untuk siapa, suasana seperti apa, "
    "dan bagian mana yang paling kamu tunggu. Kalau punya file musik sebagai "
    "referensi, kirim saja, aku akan mengukurnya dulu.\n\n"
    "Kalau sudah cukup jelas, tekan \"Bangun Prompt Suno\" dan sembilan agen akan "
    "mengerjakan lagunya sampai prompt siap dipakai."
)


class StudioApp(tk.Tk):
    """The application window."""

    def __init__(self, *, settings_store: SettingsStore | None = None, offline: bool = False) -> None:
        super().__init__()
        self.settings_store = settings_store or SettingsStore()
        self.session = StudioSession(
            settings_store=self.settings_store,
            run_store=RunStore(),
            on_event=None,
            on_chat=None,
            on_delta=self._on_delta,
        )
        if offline:
            self.session.settings.offline_demo = True

        self.palette = Palette()
        typo = resolve_typography(self)
        self.fonts = build_fonts(typo)
        self.typography = typo

        self.title("Lirik Studio")
        self.configure(bg=self.palette.bg)
        self.geometry(self._initial_geometry())
        self.minsize(1180, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._stage_status: dict[str, str] = {s: "pending" for s in STAGE_ORDER}
        self._run_dir: Path | None = None
        self._started_at = 0.0
        self._timer_job: str | None = None
        self._last_state: PipelineState = PipelineState()
        self._right_view = "prompt"
        #: Streamed text from worker threads, consumed by ``_drain``.
        self._deltas: "queue.Queue[tuple[str, str]]" = queue.Queue()
        #: Consultant replies from the intake worker: ("ok", payload) or ("error", text).
        self._intake_results: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        self._build()
        self._bind_shortcuts()
        self._sync_provider_badge()
        self.after(120, self._drain)

    # ------------------------------------------------------------------ geometry
    def _initial_geometry(self) -> str:
        window = self.session.settings.window or {}
        width = int(window.get("w") or 1420)
        height = int(window.get("h") or 900)
        x = int(window.get("x") or 80)
        y = int(window.get("y") or 40)
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width = max(1180, min(width, screen_w - 40))
        height = max(760, min(height, screen_h - 60))
        x = max(0, min(x, screen_w - width - 10))
        y = max(0, min(y, screen_h - height - 40))
        return f"{width}x{height}+{x}+{y}"

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = tk.Frame(self, bg=self.palette.bg)
        root.pack(fill="both", expand=True)
        self._build_header(root)
        # A three-column shell: fixed rail, then a draggable split between the
        # conversation and the information pane. The split exists because a fixed
        # right width cannot serve both "Prompt Suno" (four narrow fields) and
        # "Log Agen" (a prompt and a reply side by side): the log needs real
        # horizontal room, and the user is the one who knows which they are reading.
        body = tk.PanedWindow(
            root,
            orient="horizontal",
            bg=self.palette.bg,
            sashwidth=8,
            sashrelief="flat",
            bd=0,
            showhandle=False,
        )
        body.pack(fill="both", expand=True, padx=16)
        self._shell = body
        self._build_rail(body)
        self._build_centre(body)
        self._build_right(body)
        self._build_status(root)
        # Place the split after the first layout pass so both panes start at a
        # useful width: the conversation needs less than half of a 1560px window,
        # while the log and the prompt panel need the rest.
        self.after_idle(self._place_default_sash)

    def _rail_width(self) -> int:
        """Width of the stage rail, so the split can be computed relative to it."""
        try:
            return int(self._shell.panes()[0] and self._shell.nametowidget(self._shell.panes()[0]).winfo_width()) or M.rail_w
        except (tk.TclError, IndexError):  # pragma: no cover - geometry race
            return M.rail_w

    #: Fraction of the shell width given to the conversation column. The
    #: information pane needs more than half, because the agent log shows a prompt
    #: and a reply, and the prompt panel shows four long text fields.
    CHAT_FRACTION = 0.42

    def _place_default_sash(self, attempt: int = 0) -> None:
        """Set the conversation/information split, once geometry is real.

        Retried rather than placed once: at the first idle callback the shell has
        not been sized yet, so ``sash_place`` is silently overridden by Tk's own
        layout pass and the split reverts to whatever the child widths request.
        """
        try:
            if not self._shell.winfo_exists():
                return
            total = self._shell.winfo_width()
            if total < 600 and attempt < 12:
                self.after(60, lambda: self._place_default_sash(attempt + 1))
                return
            # Three columns means two sashes: 0 sits after the stage rail, 1 sits
            # between the conversation and the information pane. The interesting one
            # is 1; placing 0 would resize the rail instead of the split.
            rail = self._rail_width()
            target = rail + int((total - rail) * self.CHAT_FRACTION)
            saved = (self.session.settings.window or {}).get("panes")
            if saved and len(saved) >= 2:
                self._shell.sash_place(1, int(saved[1]), 0)
            elif total >= 600:
                self._shell.sash_place(1, max(rail + 360, target), 0)
        except tk.TclError:  # pragma: no cover - window closing
            pass

    def _build_header(self, parent) -> None:
        header = tk.Frame(parent, bg=self.palette.bg, height=M.header_h)
        header.pack(fill="x", padx=16, pady=(14, 10))
        header.pack_propagate(False)

        identity = tk.Frame(header, bg=self.palette.bg)
        identity.pack(side="left", fill="y")
        # Wordmark: a level-meter motif instead of a logo bitmap.
        mark = tk.Canvas(identity, width=46, height=40, bg=self.palette.bg, highlightthickness=0)
        mark.pack(side="left", pady=2)
        for index, weight in enumerate((0.35, 0.6, 0.85, 1.0, 0.7, 0.45)):
            x = 4 + index * 7
            height = int(10 + weight * 24)
            mark.create_rectangle(
                x, 38 - height, x + 5, 38,
                fill=self.palette.level_colour(index / 5.0), outline="",
            )
        titles = tk.Frame(identity, bg=self.palette.bg)
        titles.pack(side="left", padx=(8, 0))
        tk.Label(
            titles, text="LIRIK STUDIO", bg=self.palette.bg, fg=self.palette.fg,
            font=self.fonts.display_l, anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles,
            text="sembilan spesialis \u00b7 satu prompt suno",
            bg=self.palette.bg,
            fg=self.palette.fg_faint,
            font=self.fonts.small,
            anchor="w",
        ).pack(anchor="w")

        actions = tk.Frame(header, bg=self.palette.bg)
        actions.pack(side="right", fill="y")
        self.provider_badge = Badge(actions, "memuat\u2026", self.palette, self.fonts)
        self.provider_badge.pack(side="right", padx=(8, 0))
        IconButton(
            actions, "\u2699", self._open_settings, self.palette, self.fonts,
            tooltip="Pengaturan LLM, routing, pipeline (Ctrl+,)",
        ).pack(side="right")
        IconButton(
            actions, "\u2139", self._show_about, self.palette, self.fonts,
            tooltip="Tentang & alur sembilan agen",
        ).pack(side="right", padx=6)

    def _build_rail(self, parent) -> None:
        container = tk.Frame(parent, bg=self.palette.line, width=M.rail_w)
        parent.add(container, minsize=M.rail_w, width=M.rail_w, stretch="never")
        container.pack_propagate(False)
        inner = tk.Frame(container, bg=self.palette.surface)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        titles = {stage: STAGE_INFO[stage].title for stage in STAGE_ORDER}
        self.rail = StageRail(
            inner,
            STAGE_ORDER,
            titles,
            self.palette,
            self.fonts,
            on_select=self._select_stage,
            width=M.rail_w - 2,
        )
        self.rail.pack(fill="x", padx=1, pady=(6, 0))

        tk.Frame(inner, bg=self.palette.surface, height=6).pack()

        ref_card = ReferencePanel(
            inner,
            self.palette,
            self.fonts,
            on_choose=self._choose_references,
            on_clear=self._clear_references,
        )
        ref_card.pack(fill="x", padx=6, pady=(0, 8))

        self.rail_hint = AutoWrapLabel(inner, "", self.palette, self.fonts, fg=self.palette.fg_faint)
        self.rail_hint.pack(fill="x", padx=10, pady=(0, 10), side="bottom")

    def _build_centre(self, parent) -> None:
        container = tk.Frame(parent, bg=self.palette.line)
        parent.add(container, minsize=M.chat_min, width=760, stretch="always")
        inner = tk.Frame(container, bg=self.palette.surface)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        self.chat_panel = ChatPanel(
            inner,
            self.palette,
            self.fonts,
            on_send=self._on_send,
            on_run=self._start_run,
            on_stop=self._stop_run,
        )
        self.chat_panel.pack(fill="both", expand=True)

    def _build_right(self, parent) -> None:
        container = tk.Frame(parent, bg=self.palette.line)
        parent.add(
            container,
            minsize=M.output_min,
            width=int(M.output_min * 1.35),
            stretch="always",
        )
        container.pack_propagate(False)
        inner = tk.Frame(container, bg=self.palette.surface)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        switch = tk.Frame(inner, bg=self.palette.surface)
        switch.pack(fill="x", padx=M.pad_lg, pady=(14, 6))
        self.view_buttons: dict[str, PillButton] = {}
        for key, label in (
            ("prompt", "Prompt Suno"),
            ("inspect", "Inspeksi"),
            ("log", "Log Agen"),
        ):
            button = PillButton(
                switch, label, (lambda k=key: self._show_right(k)), self.palette,
                self.fonts, kind="ghost", height=28,
            )
            button.pack(side="left", padx=(0, 6))
            self.view_buttons[key] = button

        self.right_stack = tk.Frame(inner, bg=self.palette.surface)
        self.right_stack.pack(fill="both", expand=True, padx=M.pad_lg, pady=(0, M.pad_lg))

        self.prompt_panel = PromptPanel(
            self.right_stack,
            self.palette,
            self.fonts,
            on_export=self._export_run,
            on_open_folder=self._open_run_folder,
            on_copy_all=self._copy_all,
        )
        self.inspector = StageInspector(self.right_stack, self.palette, self.fonts)
        self.agent_log = AgentLogPanel(
            self.right_stack,
            self.palette,
            self.fonts,
            source=lambda cursor: self.session.trace.since(cursor),
            on_export=self._export_log,
            on_clear=self._clear_log,
        )
        self._show_right("prompt")

    def _build_status(self, parent) -> None:
        Divider(parent, self.palette).pack(fill="x")
        bar = tk.Frame(parent, bg=self.palette.bg, height=M.status_h)
        bar.pack(fill="x", padx=16)
        bar.pack_propagate(False)
        self.status_stage = tk.Label(
            bar, text="siap", bg=self.palette.bg, fg=self.palette.fg_muted,
            font=self.fonts.mono_small, anchor="w",
        )
        self.status_stage.pack(side="left", pady=6)
        self.status_metrics = tk.Label(
            bar, text="", bg=self.palette.bg, fg=self.palette.fg_faint,
            font=self.fonts.mono_small, anchor="e",
        )
        self.status_metrics.pack(side="right", pady=6)

    # ------------------------------------------------------------------ shortcuts
    def _bind_shortcuts(self) -> None:
        self.bind("<Control-comma>", lambda _e: self._open_settings())
        self.bind("<Control-Return>", lambda _e: self._start_run())
        self.bind("<Control-s>", lambda _e: self._export_run())
        self.bind("<Control-l>", lambda _e: self.chat_panel.composer.focus_input())
        self.bind("<F5>", lambda _e: self._start_run())

    # ------------------------------------------------------------------ chat
    def _on_send(self, text: str) -> None:
        if self.session.running:
            self.chat_panel.view.add_note("Pipeline sedang berjalan; tunggu atau hentikan.", colour="fg_faint")
            return
        self.chat_panel.view.add_message("user", text, at=_clock())
        self.chat_panel.set_busy(True)
        self.status_stage.configure(text="konsultan sedang menjawab\u2026")

        def worker() -> None:
            """Ask the consultant on a worker thread.

            The result is handed over through a queue rather than ``after()`` for two
            independent reasons, and both were live bugs:

            * **Tk is not thread-safe.** Calling ``after()`` from a worker can
              deadlock or corrupt the interpreter's Tcl state; every other worker in
              this app already funnels through ``_drain_once`` on the UI thread, and
              this path was the one exception.
            * **``except ... as exc`` unbinds ``exc`` when the block exits.** A lambda
              evaluated later therefore raised
              ``NameError: cannot access free variable 'exc'``, so the *error* path
              crashed instead of reporting the error - which is exactly the moment a
              user most needs a working message. Capturing the exception into a local
              at raise time is immune to that scoping rule.
            """
            try:
                payload = self.session.send_user_message(text)
            except Exception as error:  # noqa: BLE001 - reported in the transcript
                as_text = f"{type(error).__name__}: {error}"
                self._intake_results.put(("error", as_text))
            else:
                self._intake_results.put(("ok", payload))

        threading.Thread(target=worker, name="lirik-intake", daemon=True).start()

    def _after_intake(self, payload) -> None:
        self.chat_panel.set_busy(False)
        self.chat_panel.view.add_message("assistant", payload["reply"], at=_clock())
        brief = payload.get("brief")
        if brief is not None and brief.open_questions:
            self.chat_panel.view.add_note(
                "Konsultan masih menandai: " + "; ".join(brief.open_questions[:3]), colour="fg_muted"
            )
        if payload.get("ready_to_build"):
            self.chat_panel.view.add_note(
                "Brief sudah siap dibangun. Tekan \"Bangun Prompt Suno\" (Ctrl+Enter).",
                colour="mint",
            )
        self.chat_panel.set_turns(self.session.turn_count, self.session.turn_budget)
        self.status_stage.configure(text="siap")
        self._refresh_brief_stage()

    def _after_intake_error(self, detail: Any) -> None:
        """Report a failed consultant turn. ``detail`` is text, never an exception.

        Passing the message as a string keeps this method safe to call from a queue
        drain long after the original ``except`` block has exited.
        """
        message = str(detail)
        self.chat_panel.set_busy(False)
        self.chat_panel.view.add_note(f"Gagal menghubungi model: {message}", colour="red")
        self.status_stage.configure(text="gagal")
        messagebox.showerror(
            "Gagal memanggil model",
            f"{message}\n\nPeriksa Pengaturan > Provider LLM, lalu tekan Tes koneksi.",
            parent=self,
        )

    def _on_delta(self, stage: str, chunk: str) -> None:
        """Queue streamed text from whichever thread produced it."""
        self._deltas.put((stage, chunk))


    # ------------------------------------------------------------------ references
    def _choose_references(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self, title="Pilih file musik referensi", filetypes=AUDIO_TYPES
        )
        if not paths:
            return
        self._analyse(list(paths))

    def _analyse(self, paths: list[str]) -> None:
        self.status_stage.configure(text="menganalisa audio referensi\u2026")
        self.update_idletasks()
        bundle = self.session.analyse_references(paths)
        self._render_reference(bundle)
        self.status_stage.configure(text="siap")

    def _render_reference(self, bundle: ReferenceBundle) -> None:
        ref_card = None
        for child in self.winfo_children():
            pass
        # The reference card lives inside the rail; find it through the widget tree
        # once and remember it, rather than recomputing on every update.
        if not hasattr(self, "_ref_card"):
            for widget in self._walk(self):
                if isinstance(widget, ReferencePanel):
                    self._ref_card = widget
                    break
        ref_card = getattr(self, "_ref_card", None)
        if ref_card is not None:
            ref_card.set_bundle(bundle)
        if bundle.error:
            self.chat_panel.view.add_note(f"Referensi gagal dibaca: {bundle.error}", colour="red")
        elif bundle.summary:
            self.chat_panel.view.add_note(
                f"Referensi terukur: {bundle.lossless} (kualitas {bundle.quality}/100). "
                "Angka lengkap ada di panel referensi.",
                colour="cyan",
            )

    @staticmethod
    def _walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from StudioApp._walk(child)

    def _clear_references(self) -> None:
        self.session.analyse_references([])
        if hasattr(self, "_ref_card"):
            self._ref_card.set_bundle(self.session.reference)
        self.chat_panel.view.add_note("Referensi dibersihkan.", colour="fg_faint")

    # ------------------------------------------------------------------ run
    def _start_run(self) -> None:
        if self.session.running:
            return
        if self.session.brief is None and not self.session.settings.offline_demo:
            if not messagebox.askyesno(
                "Belum ada brief",
                "Kamu belum bercakap dengan konsultan. Jalankan pipeline dengan brief "
                "kosong? Hasilnya akan generik.",
                parent=self,
            ):
                return
        self._reset_run_ui()
        try:
            self._run_dir = self.session.start_run()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Tidak bisa mulai", str(exc), parent=self)
            return
        self.chat_panel.set_busy(True)
        self.chat_panel.view.add_rule("pipeline")
        self.chat_panel.view.add_note(
            f"Pipeline dimulai. Hasil akan ditulis ke {self._run_dir}", colour="fg_muted"
        )
        self._started_at = time.time()
        self._tick_timer()

    def _stop_run(self) -> None:
        self.session.stop_run()
        self.chat_panel.view.add_note("Permintaan berhenti dikirim\u2026", colour="amber")

    def _reset_run_ui(self) -> None:
        self.rail.reset()
        self._stage_status = {s: "pending" for s in STAGE_ORDER}
        self.prompt_panel.set_prompt(None)
        if self._right_view == "inspect":
            self.inspector.show_stage(self.rail.selected or STAGE_ORDER[0], PipelineState(), "pending")

    # The pipeline runs on a worker thread. Tk is not thread-safe: calling
    # `after()` or touching a widget from that thread can deadlock the main loop.
    # So the worker only ever pushes into thread-safe queues, and the UI thread's
    # ``_drain`` pump does all the widget work. That is the whole contract.

    def _handle_event(self, event: PipelineEvent) -> None:
        kind = event.kind
        stage = event.stage
        if kind == "stage_start":
            self._stage_status[stage] = "running"
            self.rail.set_status(stage, "running", "berjalan\u2026")
            self.status_stage.configure(text=f"{STAGE_INFO[stage].title}\u2026")
            self.chat_panel.view.add_note(
                f"\u25b8 {STAGE_INFO[stage].title} dimulai", colour="fg_muted"
            )
            if self._right_view == "inspect":
                self.inspector.show_stage(stage, self._current_state(), "running")
        elif kind == "stage_done":
            if is_review_stage(stage):
                return
            self._stage_status[stage] = "done"
            note = truncate(event.message, 34)
            self.rail.set_status(stage, "done", note)
            self.chat_panel.view.add_note(
                f"  \u2713 {STAGE_INFO[stage].title}: {truncate(event.message, 150)}",
                colour="mint",
            )
            self._refresh_brief_stage()
        elif kind == "gates":
            failed = event.data.get("failures") or []
            colour = "amber" if failed else "mint"
            label = ", ".join(failed) if failed else "semua gate lulus"
            self.chat_panel.view.add_note(
                f"  gate mekanis: {label} (skor {event.data.get('score', 0):.0f}/100)", colour=colour
            )
        elif kind == "review":
            verdict = event.data.get("verdict") or {}
            passed = bool(verdict.get("passed"))
            self._stage_status[stage] = "done" if passed else "failed"
            self.rail.set_status(
                stage, "done" if passed else "failed",
                truncate(event.message, 30),
            )
            self.chat_panel.view.add_message(
                "reviewer",
                f"{STAGE_INFO[stage].title}: {event.message}\n\n"
                + _verdict_digest(verdict),
                at=_clock(),
            )
            if self._right_view == "inspect":
                self.inspector.show_stage(stage, self._current_state(), "done" if passed else "failed")
        elif kind == "rollback":
            target = event.data.get("target", "")
            self.chat_panel.view.add_note(
                f"\u21b6 MUNDUR ke {STAGE_INFO.get(target, STAGE_INFO[stage]).title} "
                f"(revisi {event.data.get('verdict', {}).get('overall_score', 0):.0f}/100 \u2192 perbaikan)",
                colour="amber",
            )
            if event.data.get("repeat"):
                # The most useful signal in the whole log: the same defect survived a
                # previous rewrite of the same stage, so another cosmetic pass will not
                # help. Surfacing it here is what stops a silent repeat loop.
                self.chat_panel.view.add_note(
                    "    \u26a0 masalah yang SAMA sudah pernah diperbaiki dan masih gagal "
                    "\u2014 putaran ini diminta mengubah pendekatan, bukan kosmetik.",
                    colour="red",
                )
            for name in (event.data.get("gates_failed") or [])[:4]:
                self.chat_panel.view.add_note(f"    gate gagal: {name}", colour="red")
            for issue in (event.data.get("issues") or [])[:4]:
                self.chat_panel.view.add_note(
                    f"    [{issue.get('severity')}] {truncate(str(issue.get('message', '')), 120)}",
                    colour="fg_muted",
                )
            if target:
                self.rail.select(target)
        elif kind == "warn":
            self.chat_panel.view.add_note(f"\u26a0 {truncate(event.message, 160)}", colour="amber")
        elif kind == "debug":
            pass
        elif kind == "error":
            self.chat_panel.view.add_note(f"\u2715 {truncate(event.message, 200)}", colour="red")
        elif kind == "thread_done":
            self._finish_run(event.message)
        self._refresh_prompt_panel()

    def _finish_run(self, run_dir: str) -> None:
        self.chat_panel.set_busy(False)
        self._stop_timer()
        result = self.session.result
        self._last_state = result.state if result else self._current_state()
        if result is not None and result.prompt is not None:
            self._refresh_prompt_panel(force=True)
            if result.passed:
                self.chat_panel.view.add_rule("selesai")
                self.chat_panel.view.add_note(
                    f"LULUS. Prompt siap dipakai. Revisi: {result.state.revisions}. "
                    f"File tersimpan di {run_dir}",
                    colour="mint",
                )
                self._show_right("prompt")
            else:
                self.chat_panel.view.add_note(
                    f"Selesai tanpa verdict final penuh. {result.stopped_reason} "
                    f"Karya terbaik tetap tersedia di panel Prompt Suno.",
                    colour="amber",
                )
                self._show_right("prompt")
        else:
            self.chat_panel.view.add_note(
                "Run berhenti tanpa prompt akhir. Lihat panel Inspeksi untuk detail.",
                colour="red",
            )
        self.status_stage.configure(text="siap")

    # ------------------------------------------------------------------ timers/refresh
    def _tick_timer(self) -> None:
        if not self.session.running:
            self._timer_job = None
            self._update_status_metrics()
            return
        self._update_status_metrics()
        self._timer_job = self.after(500, self._tick_timer)

    def _stop_timer(self) -> None:
        if self._timer_job is not None:
            try:
                self.after_cancel(self._timer_job)
            except tk.TclError:  # pragma: no cover
                pass
            self._timer_job = None

    def _update_status_metrics(self) -> None:
        parts: list[str] = []
        if self._started_at:
            elapsed = time.time() - self._started_at
            parts.append(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        state = self._current_state()
        parts.append(f"revisi {state.revisions}")
        tokens = sum(h.prompt_tokens + h.completion_tokens for h in state.history)
        if tokens:
            parts.append(f"token~{tokens}")
        if self._run_dir is not None:
            parts.append(f"run {self._run_dir.name}")
        self.status_metrics.configure(text="   ".join(parts))

    def _current_state(self) -> PipelineState:
        if self.session.orchestrator is not None:
            return self.session.orchestrator.state
        return self._last_state

    def _refresh_brief_stage(self) -> None:
        brief = self.session.brief
        if brief is None:
            return
        self.rail.set_status("intake", "done", truncate(brief.working_title or "brief siap", 28))
        self._stage_status["intake"] = "done"
        if self.rail.selected in {"", "intake"}:
            self.inspector.show_stage("intake", self._current_state(), "done")

    def _refresh_prompt_panel(self, force: bool = False) -> None:
        state = self._current_state()
        if force or state.prompt is not None:
            self.prompt_panel.set_prompt(state.prompt, state.prompt_review)

    def _drain(self) -> None:
        """Schedule the pump that moves worker-thread events into the widgets."""
        try:
            self._drain_once()
        except Exception:  # pragma: no cover - never kill the UI loop
            pass
        # Rescheduling after destroy() raises TclError, so stop cleanly instead.
        try:
            if self.winfo_exists():
                self.after(120, self._drain)
        except tk.TclError:  # pragma: no cover - window already gone
            pass

    def _drain_once(self) -> int:
        """One pump pass: streamed deltas first, then pipeline events.

        Tk is single-threaded, so this is the only place a pipeline event may
        reach a widget. It is a separate method so tests can drive the exact code
        path the running app uses instead of re-implementing it.
        """
        handled = 0
        while True:
            try:
                stage, chunk = self._deltas.get_nowait()
            except queue.Empty:
                break
            if stage == "intake":
                self.chat_panel.view.stream_chunk(chunk)
        while True:
            try:
                kind, value = self._intake_results.get_nowait()
            except queue.Empty:
                break
            if kind == "ok":
                self._after_intake(value)
            else:
                self._after_intake_error(value)
            handled += 1
        for event in self.session.drain_events():
            self._handle_event(event)
            handled += 1
        # The agent log is written by worker threads, so it is polled the same way
        # the queues are rather than pushed: a widget must never be touched off the
        # UI thread.
        if self.agent_log is not None:
            self.agent_log.refresh()
        return handled

    # ------------------------------------------------------------------ right pane
    def _show_right(self, key: str) -> None:
        self._right_view = key
        panes = {
            "prompt": self.prompt_panel,
            "inspect": self.inspector,
            "log": self.agent_log,
        }
        for frame in panes.values():
            frame.pack_forget()
        target = panes.get(key, self.prompt_panel)
        target.pack(fill="both", expand=True)
        for name, button in self.view_buttons.items():
            button.configure(bg=self.palette.accent if name == key else self.palette.surface_alt)
            button.redraw()
        if key == "inspect":
            stage = self.rail.selected or self._earliest_active_stage()
            self.inspector.show_stage(stage, self._current_state(), self._stage_status.get(stage, "pending"))
        if key == "log":
            self.agent_log.refresh()

    def _earliest_active_stage(self) -> str:
        for stage in STAGE_ORDER:
            if self._stage_status.get(stage) in {"running", "failed"}:
                return stage
        done = [s for s in STAGE_ORDER if self._stage_status.get(s) == "done"]
        return done[-1] if done else STAGE_ORDER[0]

    def _select_stage(self, stage: str) -> None:
        self._show_right("inspect")
        self.inspector.show_stage(stage, self._current_state(), self._stage_status.get(stage, "pending"))

    # ------------------------------------------------------------------ actions
    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self,
            self.palette,
            self.fonts,
            self.session.settings,
            self.settings_store,
            on_saved=self._apply_settings,
        )
        dialog.focus_set()

    def _apply_settings(self, settings: AppSettings) -> None:
        self.session.settings = settings
        self._sync_provider_badge()
        mode = "demo offline" if settings.offline_demo else "provider aktif"
        self.chat_panel.view.add_note(
            f"Pengaturan disimpan ({mode}). "
            f"Revisi maksimum {settings.max_revisions}, percobaan per tahap {settings.stage_attempts}.",
            colour="cyan",
        )

    def _sync_provider_badge(self) -> None:
        status = self.session.provider_status()
        problem = status.startswith(("Belum", "Provider bermasalah"))
        if self.session.settings.offline_demo:
            status, problem = "demo offline", False
        self.provider_badge.set(
            truncate(status, 42),
            self.palette.amber if problem else self.palette.mint,
            "\u26a0" if problem else "\u25cf",
        )

    def _show_about(self) -> None:
        from ..domain.graph import describe_graph

        messagebox.showinfo(
            "Tentang Lirik Studio",
            "Lirik Studio\n\nSembilan agen spesialis menyusun satu prompt Suno.\n\n"
            + describe_graph()
            + "\n\nSetiap reviewer dapat memundurkan proses ke tahap akar masalah "
            "(composer, lyricist, arranger, analyst, atau intake), dan keputusan itu "
            "selalu diverifikasi gate mekanis di Python.",
            parent=self,
        )

    def _export_run(self) -> None:
        state = self._current_state()
        if state.prompt is None:
            messagebox.showinfo("Belum ada prompt", "Jalankan pipeline terlebih dahulu.", parent=self)
            return
        base = self.session.settings.output_dir or str(user_config_dir() / "runs")
        target = filedialog.askdirectory(
            parent=self, title="Pilih folder penyimpanan", initialdir=base
        )
        if not target:
            return
        directory = Path(target) / (self._run_dir.name if self._run_dir else "lirik-run")
        try:
            if self.session.orchestrator is not None:
                self.session.orchestrator.save_outputs(directory)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "suno_prompt.txt").write_text(
                    state.prompt.to_suno_block(), encoding="utf-8"
                )
        except OSError as exc:
            messagebox.showerror("Gagal menyimpan", str(exc), parent=self)
            return
        self.session.settings.last_export_dir = str(directory)
        self.session.save_settings()
        self.chat_panel.view.add_note(f"Hasil disimpan ke {directory}", colour="mint")

    def _open_run_folder(self) -> None:
        target = None
        if self._run_dir is not None and Path(self._run_dir).exists():
            target = Path(self._run_dir)
        elif self.session.settings.last_export_dir:
            target = Path(self.session.settings.last_export_dir)
        if target is None or not target.exists():
            messagebox.showinfo(
                "Belum ada folder", "Belum ada run yang disimpan.", parent=self
            )
            return
        _open_in_file_manager(target)

    def _export_log(self) -> None:
        """Write the session log next to the run outputs, and offer a copy."""
        trace = self.session.trace
        if not len(trace):
            messagebox.showinfo("Log kosong", "Belum ada aktivitas agen untuk diekspor.", parent=self)
            return
        target = self._run_dir
        if target is None:
            base = self.session.settings.output_dir or str(user_config_dir() / "runs")
            target = Path(base) / f"log-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            target.mkdir(parents=True, exist_ok=True)
            (target / "agent_log.jsonl").write_text(trace.to_jsonl(), encoding="utf-8")
            (target / "agent_log.md").write_text(trace.to_markdown(), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Gagal menyimpan log", str(exc), parent=self)
            return
        self.chat_panel.view.add_note(
            f"Log agen disimpan ke {target} (agent_log.jsonl + agent_log.md)", colour="mint"
        )

    def _clear_log(self) -> None:
        if not messagebox.askyesno(
            "Bersihkan log",
            "Kosongkan log sesi ini? Berkas run yang sudah tersimpan tidak terhapus.",
            parent=self,
        ):
            return
        self.session.trace.clear()
        # forget(), not clear(): the recorder itself is empty now, so the cursor
        # should go back to zero and stay there.
        self.agent_log.forget()
        self.chat_panel.view.add_note("Log agen dibersihkan.", colour="fg_faint")

    def _copy_all(self) -> None:
        self.prompt_panel.copy_all()
        self.chat_panel.view.add_note("Prompt lengkap disalin ke clipboard.", colour="mint")

    # ------------------------------------------------------------------ lifecycle
    def _on_close(self) -> None:
        if self.session.running:
            if not messagebox.askyesno(
                "Pipeline berjalan",
                "Pipeline masih berjalan. Hentikan dan keluar?",
                parent=self,
            ):
                return
            self.session.stop_run()
        try:
            settings = self.session.settings
            settings.window = {
                "w": self.winfo_width(),
                "h": self.winfo_height(),
                "x": self.winfo_x(),
                "y": self.winfo_y(),
                # Remember both sashes so the layout the user chose comes back.
                "panes": [
                    int(self._shell.sash_coord(index)[0])
                    for index in (0, 1)
                    if self._shell.sash_coord(index)
                ],
            }
            self.settings_store.save(settings)
        except Exception:  # pragma: no cover - closing must not raise
            pass
        self.destroy()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def is_review_stage(stage: str) -> bool:
    from ..domain.graph import is_reviewer

    return is_reviewer(stage)


def _verdict_digest(verdict: dict) -> str:
    lines: list[str] = []
    dimensions = verdict.get("dimensions") or {}
    if dimensions:
        weak = sorted(
            ((k, float(v)) for k, v in dimensions.items()), key=lambda kv: kv[1]
        )[:3]
        lines.append(
            "Dimensi terlemah: "
            + ", ".join(f"{name.replace('_', ' ')} {value:.1f}/10" for name, value in weak)
        )
    gates = verdict.get("gates_failed") or []
    if gates:
        lines.append("Gate gagal: " + ", ".join(gates))
    restart = verdict.get("restart_from")
    if restart:
        title = STAGE_INFO.get(restart, None)
        lines.append(f"Akar masalah menurut reviewer: {title.title if title else restart}")
    issues = verdict.get("issues") or []
    for issue in issues[:6]:
        fix = f" \u2192 {issue.get('fix')}" if issue.get("fix") else ""
        lines.append(f"- [{issue.get('severity', 'major')}] {issue.get('message', '')}{fix}")
    strengths = verdict.get("strengths") or []
    if strengths:
        lines.append("Dipertahankan: " + "; ".join(strengths[:3]))
    return "\n".join(lines)


def _clock() -> str:
    return time.strftime("%H:%M")


def _open_in_file_manager(path: Path) -> None:
    target = str(path)
    try:
        if os.name == "nt":
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":  # pragma: no cover - platform specific
            subprocess.Popen(["open", target])
        else:  # pragma: no cover - platform specific
            subprocess.Popen(["xdg-open", target])
    except Exception:  # pragma: no cover - best effort
        pass


def run_app(offline: bool = False) -> None:
    """Entry point used by ``python -m suno_agents``."""
    app = StudioApp(offline=offline)
    app.chat_panel.view.add_message("assistant", WELCOME, at=_clock())
    app.session.seed_conversation(WELCOME)
    if offline:
        app.chat_panel.view.add_note(
            "Mode demo offline aktif: agen tiruan, tanpa panggilan ke provider.",
            colour="amber",
        )
    app.mainloop()


__all__ = ["StudioApp", "run_app", "WELCOME"]

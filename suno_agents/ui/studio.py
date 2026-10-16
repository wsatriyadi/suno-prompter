"""The studio surfaces: reference intake, stage inspector, and the final prompt.

Three panels, each answering one question:

* :class:`ReferencePanel` - "what did the app measure in my reference files?"
  Numbers, a lossless verdict badge, and the honest caveats, because the whole
  point of measuring is to stop the prompt promising fidelity that is not there.
* :class:`StageInspector` - "what did this specific agent actually produce?"
  Every stage's artifact, plus the reviewer's dimension meters and issue list.
* :class:`PromptPanel` - "what do I paste into Suno?" Four fields, character
  counters, copy buttons, and the lossless checklist.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..domain.graph import LIMITS, STAGE_INFO
from ..domain.models import (
    Arrangement,
    CreativeBrief,
    LyricSheet,
    MusicBlueprint,
    PipelineState,
    ReviewVerdict,
    SunoPrompt,
)
from ..quality import text as T
from ..quality.gates import PASS_MARKS, pass_criteria
from .theme import Fonts, Metrics, Palette, truncate
from .widgets import (
    AutoWrapLabel,
    Badge,
    Card,
    Chip,
    Divider,
    LevelMeter,
    PillButton,
    ScrollFrame,
    SectionLabel,
)

M = Metrics()


def _copy_to_clipboard(widget: tk.Misc, text: str) -> None:
    widget.clipboard_clear()
    widget.clipboard_append(text)
    widget.update_idletasks()


class FieldBox(tk.Frame):
    """One copyable Suno field: label, counter, body, copy button."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        label: str,
        hint: str = "",
        height: int = 6,
        limit: int = 0,
        mono: bool = False,
    ) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts
        self._limit = limit
        self._value = ""

        head = tk.Frame(self, bg=palette.surface)
        head.pack(fill="x")
        tk.Label(
            head, text=label.upper(), bg=palette.surface, fg=palette.fg_faint,
            font=fonts.label, anchor="w",
        ).pack(side="left")
        self.counter = tk.Label(
            head, text="", bg=palette.surface, fg=palette.fg_muted,
            font=fonts.mono_small, anchor="e",
        )
        self.counter.pack(side="right")
        self.copy_button = PillButton(
            head,
            "Salin",
            lambda: _copy_to_clipboard(self, self._value),
            palette,
            fonts,
            kind="ghost",
            height=22,
            glyph="\u29c9",
            tooltip=f"Copy {label} to clipboard",
        )
        self.copy_button.pack(side="right", padx=(0, 8))

        self.text = tk.Text(
            self,
            height=height,
            wrap="word",
            bg=palette.surface_alt,
            fg=palette.fg if not mono else palette.cyan,
            font=fonts.mono_small if mono else fonts.body,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.line_soft,
            highlightcolor=palette.line,
            padx=10,
            pady=8,
            spacing3=2,
        )
        self.text.pack(fill="both", expand=True, pady=(4, 0))
        self.text.configure(state="disabled")
        if hint:
            AutoWrapLabel(
                self, hint, palette, fonts, fg=palette.fg_faint,
            ).pack(fill="x", pady=(3, 0))

    def set_value(self, value: str) -> None:
        self._value = value or ""
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self._value)
        self.text.configure(state="disabled")
        self._sync_counter()

    def get_value(self) -> str:
        """Current field value, as last set (not as displayed)."""
        return self._value

    def _sync_counter(self) -> None:
        length = len(self._value)
        if self._limit:
            ratio = length / self._limit
            colour = (
                self._palette.red
                if ratio > 1.0
                else self._palette.amber
                if ratio > 0.9
                else self._palette.fg_muted
            )
            self.counter.configure(text=f"{length} / {self._limit}", fg=colour)
        else:
            self.counter.configure(text=str(length))


class ReferencePanel(Card):
    """Reference audio intake: choose files, measure, show the verdict."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        on_choose: Callable[[], None],
        on_clear: Callable[[], None],
    ) -> None:
        super().__init__(master, palette, fonts, eyebrow="Referensi", title="Analisa File Musik")
        self._palette = palette
        self._fonts = fonts
        self._on_choose = on_choose

        row = tk.Frame(self.body, bg=palette.surface)
        row.pack(fill="x")
        self.choose_button = PillButton(
            row, "Pilih file", on_choose, palette, fonts, kind="ghost",
            glyph="\u2295", height=28, tooltip="mp3, wav, m4a, flac, ogg, aiff",
        )
        self.choose_button.pack(side="left")
        PillButton(
            row, "Bersihkan", on_clear, palette, fonts, kind="ghost",
            glyph="\u2715", height=28, tooltip="Hapus semua referensi",
        ).pack(side="left", padx=6)
        self.verdict_badge = Badge(row, "belum ada referensi", palette, fonts)
        self.verdict_badge.pack(side="right")

        self.files_label = AutoWrapLabel(
            self.body,
            "Tidak ada file. Kirim file musik sebagai referensi (opsional).",
            palette,
            fonts,
        )
        self.files_label.pack(fill="x", pady=(8, 4))

        self.metrics = tk.Text(
            self.body,
            height=10,
            wrap="word",
            bg=palette.surface_alt,
            fg=palette.fg,
            font=fonts.mono_small,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.line_soft,
            padx=10,
            pady=8,
        )
        self.metrics.pack(fill="both", expand=True, pady=(4, 0))
        self.metrics.tag_configure("head", foreground=palette.cyan, font=fonts.body_bold)
        self.metrics.tag_configure("warn", foreground=palette.amber)
        self.metrics.tag_configure("bad", foreground=palette.red)
        self.metrics.tag_configure("good", foreground=palette.mint)
        self.metrics.tag_configure("dim", foreground=palette.fg_muted)
        self.metrics.configure(state="disabled")

    # ------------------------------------------------------------------ api
    def set_bundle(self, bundle) -> None:
        """Render a :class:`~suno_agents.pipeline.session.ReferenceBundle`."""
        if not bundle.paths:
            self.files_label.set_text(
                "Tidak ada file. Kirim file musik sebagai referensi (opsional)."
            )
            self.verdict_badge.set("belum ada referensi", self._palette.fg_muted)
            self._set_text("")
            return

        names = ", ".join(Path(p).name for p in bundle.paths[:4])
        if len(bundle.paths) > 4:
            names += f" (+{len(bundle.paths) - 4} file lain)"
        self.files_label.set_text(names)

        if bundle.error:
            self.verdict_badge.set("gagal dibaca", self._palette.red, "\u2715")
            self._set_text(bundle.error)
            return

        colour = {
            "lossless": self._palette.mint,
            "lossy": self._palette.amber,
            "lossy-transcoded": self._palette.red,
            "mixed": self._palette.amber,
        }.get(bundle.lossless, self._palette.fg_muted)
        self.verdict_badge.set(
            f"{bundle.lossless} \u00b7 {bundle.quality}/100", colour
        )
        self._set_text(bundle.summary, bundle)

    def _set_text(self, text: str, bundle=None) -> None:
        self.metrics.configure(state="normal")
        self.metrics.delete("1.0", "end")
        if not text:
            self.metrics.configure(state="disabled")
            return
        for line in text.split("\n"):
            tag = ""
            lowered = line.lower()
            if lowered.startswith(("lossless verdict", "===", "---")):
                tag = "head"
            elif "warning" in lowered or "lossy" in lowered:
                tag = "warn"
            elif "verdict" in lowered and "lossless" in lowered and bundle is not None:
                tag = "good" if bundle.lossless == "lossless" else "warn"
            elif lowered.endswith(":"):
                tag = "dim"
            self.metrics.insert("end", f"{line}\n", tag or ())
        self.metrics.configure(state="disabled")


class StageInspector(Card):
    """Shows the artifact of whichever stage is selected on the rail."""

    def __init__(self, master, palette: Palette, fonts: Fonts) -> None:
        super().__init__(master, palette, fonts, eyebrow="Inspeksi", title="Pilih tahap")
        self._palette = palette
        self._fonts = fonts

        self.status_row = tk.Frame(self.body, bg=palette.surface)
        self.status_row.pack(fill="x")
        self.status_badge = Badge(self.status_row, "menunggu", palette, fonts)
        self.status_badge.pack(side="left")
        self.score_meter = LevelMeter(
            self.status_row, palette, fonts, slots=20, height=8, caption="skor review"
        )
        self.score_meter.pack(side="right", fill="x", expand=True, padx=(12, 0))

        self.role_label = AutoWrapLabel(self.body, "", palette, fonts)
        self.role_label.pack(fill="x", pady=(8, 6))

        self.spec_label = AutoWrapLabel(self.body, "", palette, fonts, fg=palette.cyan)
        self.spec_label.pack(fill="x", pady=(0, 8))

        Divider(self.body, palette).pack(fill="x", pady=(0, 8))

        self.body_text = tk.Text(
            self.body,
            height=18,
            wrap="word",
            bg=palette.surface_alt,
            fg=palette.fg,
            font=fonts.mono_small,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.line_soft,
            padx=10,
            pady=8,
        )
        self.body_text.pack(fill="both", expand=True)
        self.body_text.tag_configure("h", foreground=palette.cyan, font=fonts.body_bold)
        self.body_text.tag_configure("k", foreground=palette.fg_faint, font=fonts.label)
        self.body_text.tag_configure("v", foreground=palette.fg, font=fonts.mono_small)
        self.body_text.tag_configure("bad", foreground=palette.red)
        self.body_text.tag_configure("warn", foreground=palette.amber)
        self.body_text.tag_configure("good", foreground=palette.mint)
        self.body_text.tag_configure("dim", foreground=palette.fg_muted)
        self.body_text.configure(state="disabled")

        self.issues_frame = tk.Frame(self.body, bg=palette.surface)
        self.issues_frame.pack(fill="x", pady=(8, 0))

    # ------------------------------------------------------------------ api
    def show_stage(self, stage: str, state: PipelineState, status: str = "pending") -> None:
        info = STAGE_INFO.get(stage)
        if info is None:
            return
        self.set_title(info.title)
        self.role_label.set_text(f"{info.role} - menghasilkan {info.produces}")
        self.spec_label.set_text(info.specialty)
        colour = {
            "pending": self._palette.fg_faint,
            "running": self._palette.accent,
            "done": self._palette.mint,
            "failed": self._palette.red,
        }.get(status, self._palette.fg_muted)
        self.status_badge.set(
            {"pending": "menunggu", "running": "berjalan", "done": "selesai", "failed": "gagal"}.get(
                status, status
            ),
            colour,
        )
        verdict = self._verdict_for(stage, state)
        self.score_meter.set_value(
            (verdict.overall_score / 100.0) if verdict else 0.0,
            f"{verdict.overall_score:.0f}/100" if verdict else "-",
        )
        for child in self.issues_frame.winfo_children():
            child.destroy()
        self._render(stage, state, verdict)
        self._render_requirement(stage, verdict)

    def _render_requirement(self, stage: str, verdict: ReviewVerdict | None) -> None:
        """State what the stage needs to pass, next to what it actually scored.

        A verdict panel that shows "revise 79/100" without the requirement is only
        half an answer, and a user reading it has to guess the rest.
        """
        if stage not in PASS_MARKS:
            return
        box = tk.Frame(self.issues_frame, bg=self._palette.surface)
        box.pack(fill="x", pady=(10, 0))
        SectionLabel(box, "Syarat lulus", self._palette, self._fonts).pack(anchor="w")
        for line in pass_criteria(stage).split("\n")[1:]:
            AutoWrapLabel(
                box,
                "\u2022 " + line,
                self._palette,
                self._fonts,
                fg=self._palette.fg_muted,
            ).pack(fill="x")
        if verdict is not None:
            status = (
                f"Status: LULUS pada {verdict.overall_score:.1f}/100"
                if verdict.passed
                else f"Status: BELUM LULUS pada {verdict.overall_score:.1f}/100"
            )
            colour = self._palette.mint if verdict.passed else self._palette.amber
            gate_note = (
                "semua gate keras lulus"
                if not verdict.gates_failed
                else "gate gagal: " + ", ".join(verdict.gates_failed)
            )
            tk.Label(
                box,
                text=f"{status}  \u00b7  {gate_note}",
                bg=self._palette.surface,
                fg=colour,
                font=self._fonts.body_bold,
                anchor="w",
            ).pack(anchor="w", pady=(4, 0))

    def _verdict_for(self, stage: str, state: PipelineState) -> ReviewVerdict | None:
        return {
            "lyric_reviewer": state.lyric_review,
            "music_reviewer": state.music_review,
            "prompt_reviewer": state.prompt_review,
        }.get(stage)

    # ------------------------------------------------------------------ renderers
    def _write(self, text: str, tag: str = "") -> None:
        self.body_text.insert("end", text, tag or ())

    def _render(self, stage: str, state: PipelineState, verdict: ReviewVerdict | None) -> None:
        self.body_text.configure(state="normal")
        self.body_text.delete("1.0", "end")
        renderer = {
            "intake": self._render_brief,
            "audio_analyst": self._render_reference,
            "composer": self._render_blueprint,
            "lyricist": self._render_lyrics,
            "arranger": self._render_arrangement,
            "prompt_engineer": self._render_prompt,
        }.get(stage)
        if renderer is None:
            self._render_review(stage, verdict)
        else:
            renderer(state)
        self.body_text.configure(state="disabled")
        if verdict is not None:
            self._render_issues(verdict)

    def _render_brief(self, state: PipelineState) -> None:
        brief: CreativeBrief | None = state.brief
        if brief is None:
            self._write("Belum ada brief. Mulai percakapan dengan konsultan.", "dim")
            return
        self._write("BRIEF TERSIMPAN\n", "h")
        self._write(brief.summary() + "\n")
        if brief.open_questions:
            self._write("\nPERTANYAAN TERBUKA\n", "h")
            for item in brief.open_questions:
                self._write(f"- {item}\n", "warn")

    def _render_reference(self, state: PipelineState) -> None:
        if not state.reference_summary:
            self._write(
                "Tidak ada referensi audio. Agen ini akan melewati interpretasi dan "
                "menyatakan bahwa tidak ada data terukur.\n",
                "dim",
            )
            return
        self._write(
            f"VERDICT LOSSLESS: {state.reference_lossless} | SKOR: {state.reference_quality}/100\n\n",
            "good" if state.reference_lossless == "lossless" else "warn",
        )
        self._write(state.reference_summary + "\n")

    def _render_blueprint(self, state: PipelineState) -> None:
        blueprint: MusicBlueprint | None = state.blueprint
        if blueprint is None:
            self._write("Belum ada blueprint musik.", "dim")
            return
        self._write(blueprint.summary() + "\n")

    def _render_lyrics(self, state: PipelineState) -> None:
        sheet: LyricSheet | None = state.lyrics
        if sheet is None:
            self._write("Belum ada lirik.", "dim")
            return
        self._write(
            f"JUDUL: {sheet.working_title or '-'}  |  BAHASA: {sheet.language or '-'}\n", "h"
        )
        if sheet.title_options:
            self._write("Opsi judul: " + " \u00b7 ".join(sheet.title_options[:5]) + "\n")
        self._write(f"VOKAL: {sheet.vocal.describe()}\n", "good")
        if sheet.instrumental:
            self._write(f"INSTRUMENTAL: {sheet.instrumental_direction}\n", "good")
        self._write(f"HOOK: {sheet.hook_line or '-'}\n")
        self._write(f"ARC: {sheet.emotional_arc or '-'}\n\n")
        for section in sheet.sections:
            cue = f" [{section.performance_cue}]" if section.performance_cue else ""
            self._write(f"[{section.name}]{cue}\n", "h")
            for line in section.lines:
                syllable_tag = "warn" if line.syllables > 16 else ""
                self._write(f"  {line.text}", syllable_tag)
                self._write(f"   ({line.syllables} syll)\n", "dim")
            if section.backing_vocals:
                self._write(f"  backing: {section.backing_vocals}\n", "dim")
            self._write("\n")

    def _render_arrangement(self, state: PipelineState) -> None:
        arrangement: Arrangement | None = state.arrangement
        if arrangement is None:
            self._write("Belum ada peta aransemen.", "dim")
            return
        self._write(arrangement.summary() + "\n")

    def _render_prompt(self, state: PipelineState) -> None:
        prompt: SunoPrompt | None = state.prompt
        if prompt is None:
            self._write("Belum ada prompt Suno.", "dim")
            return
        budget = prompt.character_budget or {}
        self._write("BUDGET KARAKTER\n", "h")
        for key in ("title", "style", "exclude", "lyrics"):
            limit = {
                "title": LIMITS["title_chars"],
                "style": LIMITS["style_chars"],
                "exclude": LIMITS["exclude_chars"],
                "lyrics": LIMITS["lyrics_chars"],
            }[key]
            used = int(budget.get(key, 0))
            tag = "bad" if used > limit else "warn" if used > limit * 0.9 else ""
            self._write(f"  {key:8s} {used:5d} / {limit}\n", tag)
        self._write(f"  descriptors: {budget.get('descriptors', '-')}\n")
        if prompt.lossless_checklist:
            self._write("\nCHECKLIST LOSSLESS\n", "h")
            for item in prompt.lossless_checklist:
                self._write(f"- {item}\n")
        if prompt.provider_notes:
            self._write("\nCATATAN\n", "h")
            self._write(prompt.provider_notes + "\n", "dim")

    def _render_review(self, stage: str, verdict: ReviewVerdict | None) -> None:
        if verdict is None:
            self._write("Reviewer belum berjalan untuk tahap ini.", "dim")
            return
        self._write(
            f"VERDICT: {verdict.verdict.upper()}  \u00b7  {verdict.overall_score:.1f}/100\n", "h"
        )
        if verdict.gates_failed:
            self._write(
                f"GATE GAGAL ({len(verdict.gates_failed)}): {', '.join(verdict.gates_failed)}\n",
                "bad",
            )
        else:
            self._write("Semua gate mekanis lulus.\n", "good")
        self._write(f"Target mundur: {verdict.restart_from or '-'}\n\n", "warn")
        if verdict.reasoning:
            self._write(verdict.reasoning + "\n")
        if verdict.strengths:
            self._write("\nKEKUATAN YANG HARUS DIPERTAHANKAN\n", "h")
            for item in verdict.strengths:
                self._write(f"- {item}\n", "good")

    def _render_issues(self, verdict: ReviewVerdict) -> None:
        """Dimension meters plus the issue list, drawn as real widgets."""
        if verdict.dimensions:
            box = tk.Frame(self.issues_frame, bg=self._palette.surface)
            box.pack(fill="x")
            SectionLabel(box, "Dimensi penilaian", self._palette, self._fonts).pack(anchor="w")
            for name, value in verdict.dimensions.items():
                row = tk.Frame(box, bg=self._palette.surface)
                row.pack(fill="x", pady=1)
                tk.Label(
                    row,
                    text=truncate(name.replace("_", " "), 22),
                    bg=self._palette.surface,
                    fg=self._palette.fg_muted,
                    font=self._fonts.small,
                    width=22,
                    anchor="w",
                ).pack(side="left")
                meter = LevelMeter(row, self._palette, self._fonts, slots=16, height=6)
                meter.pack(side="left", fill="x", expand=True, padx=(0, 8))
                tk.Label(
                    row,
                    text=f"{value:.1f}",
                    bg=self._palette.surface,
                    fg=self._palette.level_colour(value / 10.0),
                    font=self._fonts.mono_small,
                    width=4,
                    anchor="e",
                ).pack(side="right")
                meter.set_value(value / 10.0)
        if verdict.issues:
            box = tk.Frame(self.issues_frame, bg=self._palette.surface)
            box.pack(fill="x", pady=(10, 0))
            SectionLabel(box, "Temuan", self._palette, self._fonts).pack(anchor="w")
            for issue in verdict.sorted_issues()[:10]:
                row = tk.Frame(box, bg=self._palette.surface)
                row.pack(fill="x", pady=1)
                colour = {
                    "blocker": self._palette.red,
                    "major": self._palette.amber,
                    "minor": self._palette.cyan,
                    "polish": self._palette.fg_faint,
                }.get(issue.severity, self._palette.fg_muted)
                tk.Label(
                    row, text=issue.severity[:4].upper(), bg=self._palette.surface,
                    fg=colour, font=self._fonts.label, width=5, anchor="w",
                ).pack(side="left")
                AutoWrapLabel(
                    row,
                    f"{issue.message}" + (f"  →  {issue.fix}" if issue.fix else ""),
                    self._palette,
                    self._fonts,
                    fg=self._palette.fg,
                ).pack(side="left", fill="x", expand=True)


class PromptPanel(Card):
    """The deliverable: four copyable fields, checklist, export actions."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        on_export: Callable[[], None],
        on_open_folder: Callable[[], None],
        on_copy_all: Callable[[], None],
    ) -> None:
        super().__init__(master, palette, fonts, eyebrow="Hasil akhir", title="Prompt Suno")
        self._palette = palette
        self._fonts = fonts
        self.prompt: SunoPrompt | None = None

        actions = tk.Frame(self.body, bg=palette.surface)
        actions.pack(fill="x", pady=(0, 8))
        self.export_button = PillButton(
            actions, "Simpan run", on_export, palette, fonts, kind="ghost",
            glyph="\u2913", height=26, tooltip="Tulis semua artefak ke folder run",
        )
        self.export_button.pack(side="left")
        PillButton(
            actions, "Buka folder", on_open_folder, palette, fonts, kind="ghost",
            glyph="\u25a1", height=26, tooltip="Buka folder hasil di Explorer",
        ).pack(side="left", padx=6)
        PillButton(
            actions, "Salin semua", on_copy_all, palette, fonts, kind="primary",
            glyph="\u29c9", height=26, tooltip="Copy the whole paste-ready block",
        ).pack(side="right")

        self.review_badge = Badge(actions, "belum direview", palette, fonts)
        self.review_badge.pack(side="right", padx=8)

        scroll = ScrollFrame(self.body, palette)
        scroll.pack(fill="both", expand=True)

        self.title_box = FieldBox(
            scroll.inner, palette, fonts, label="Title", height=2,
            limit=LIMITS["title_chars"],
        )
        self.title_box.pack(fill="x", pady=(0, 8))
        self.style_box = FieldBox(
            scroll.inner,
            palette,
            fonts,
            label="Style of Music",
            hint=(
                f"200 karakter pertama paling menentukan. Target 5-8 deskriptor, "
                f"tanpa tanda kurung, tanpa klaim mastering."
            ),
            height=7,
            limit=LIMITS["style_chars"],
        )
        self.style_box.pack(fill="x", pady=(0, 8))
        self.exclude_box = FieldBox(
            scroll.inner,
            palette,
            fonts,
            label="Exclude Styles",
            hint="Tempat yang benar untuk semua negasi, termasuk kunci gender vokal.",
            height=3,
            limit=LIMITS["exclude_chars"],
        )
        self.exclude_box.pack(fill="x", pady=(0, 8))
        self.lyrics_box = FieldBox(
            scroll.inner,
            palette,
            fonts,
            label="Lyrics",
            hint="Setiap section ditandai [Tag] di baris sendiri, satu baris kosong antar section.",
            height=18,
            limit=LIMITS["lyrics_chars"],
        )
        self.lyrics_box.pack(fill="x", pady=(0, 8))

        self.checklist = tk.Text(
            scroll.inner,
            height=8,
            wrap="word",
            bg=palette.surface_alt,
            fg=palette.fg_muted,
            font=fonts.small,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.line_soft,
            padx=10,
            pady=8,
        )
        self.checklist.pack(fill="x", pady=(0, 4))
        self.checklist.tag_configure("head", foreground=palette.cyan, font=fonts.body_bold)
        self.checklist.tag_configure("item", foreground=palette.fg, lmargin1=10, lmargin2=22)
        self.checklist.configure(state="disabled")

        self.chips = tk.Frame(scroll.inner, bg=palette.surface)
        self.chips.pack(fill="x", pady=(8, 0))

    # ------------------------------------------------------------------ api
    def set_prompt(self, prompt: SunoPrompt | None, verdict: ReviewVerdict | None = None) -> None:
        self.prompt = prompt
        if prompt is None:
            for box in (self.title_box, self.style_box, self.exclude_box, self.lyrics_box):
                box.set_value("")
            self._set_checklist("")
            self.review_badge.set("belum direview", self._palette.fg_muted)
            for child in self.chips.winfo_children():
                child.destroy()
            return
        self.title_box.set_value(prompt.title)
        self.style_box.set_value(prompt.style)
        self.exclude_box.set_value(prompt.exclude_style_text)
        self.lyrics_box.set_value(prompt.lyrics)
        checklist = "\n".join(prompt.lossless_checklist)
        if prompt.provider_notes:
            checklist += ("\n\nCATATAN\n" if checklist else "CATATAN\n") + prompt.provider_notes
        self._set_checklist(checklist)
        self._render_chips(prompt)
        if verdict is not None:
            colour = (
                self._palette.mint
                if verdict.passed
                else self._palette.amber
                if verdict.overall_score >= 70
                else self._palette.red
            )
            self.review_badge.set(
                f"{verdict.verdict.upper()} \u00b7 {verdict.overall_score:.0f}/100", colour
            )

    def _set_checklist(self, text: str) -> None:
        self.checklist.configure(state="normal")
        self.checklist.delete("1.0", "end")
        for line in (text or "").split("\n"):
            tag = "head" if line and line.isupper() and len(line) < 40 else "item"
            if line.startswith("- "):
                self.checklist.insert("end", "\u2022 " + line[2:] + "\n", "item")
            else:
                self.checklist.insert("end", line + "\n", tag)
        self.checklist.configure(state="disabled")

    def _render_chips(self, prompt: SunoPrompt) -> None:
        for child in self.chips.winfo_children():
            child.destroy()
        coverage = T.descriptor_coverage(prompt.style[: LIMITS["style_critical_zone"]])
        SectionLabel(
            self.chips, "zona kritis 200 karakter", self._palette, self._fonts,
            colour=self._palette.fg_faint,
        ).pack(anchor="w", pady=(0, 4))
        row = tk.Frame(self.chips, bg=self._palette.surface)
        row.pack(fill="x")
        for family, present in coverage.items():
            Chip(
                row,
                family,
                self._palette,
                self._fonts,
                colour=self._palette.bg if present else self._palette.fg_faint,
                background=self._palette.mint if present else self._palette.surface_high,
            ).pack(side="left", padx=(0, 4))

    def copy_all(self) -> None:
        if self.prompt is not None:
            _copy_to_clipboard(self, self.prompt.to_suno_block())


__all__ = ["FieldBox", "PromptPanel", "ReferencePanel", "StageInspector"]

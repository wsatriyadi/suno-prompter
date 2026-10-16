"""The agent log panel: what was asked, what came back, what was decided.

The shape is deliberate and comes from the question users actually ask when a run
misbehaves: *"what did it send, what did it get, and why did it choose that?"* So
the panel is a master/detail split rather than one long scroll:

    +----------------------------+--------------------------------------+
    | log entries (list)         | selected entry, in full              |
    | time  stage  model  status | - request (system + user)            |
    | 16:07 composer  9.2s  ok   | - reply                              |
    | 16:08 lyricist  4.1s empty | - hidden reasoning, when present     |
    | 16:08 gate      revise     | - decision notes and problems        |
    +----------------------------+--------------------------------------+

Entries are appended incrementally from a worker thread, so the panel reads new
records by index and never rebuilds the whole list mid-run.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any, Callable, Sequence

from ..domain.graph import STAGE_INFO, STAGE_ORDER
from ..llm.trace import AgentDecision, LLMExchange, TraceEntry, filter_entries, summarise
from .theme import Fonts, Metrics, Palette, truncate
from .widgets import (
    AutoWrapLabel,
    Badge,
    Card,
    Divider,
    LevelMeter,
    PillButton,
    ScrollFrame,
    SectionLabel,
)

M = Metrics()

FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "Semua"),
    ("exchanges", "Permintaan"),
    ("decisions", "Keputusan"),
    ("problems", "Masalah"),
)

STATUS_COLOURS = {
    "ok": "mint",
    "truncated": "amber",
    "empty": "amber",
    "error": "red",
    "reply": "fg_muted",
    "parsed": "mint",
    "repair": "amber",
    "accepted_with_problems": "amber",
    "failed": "red",
    "gate": "cyan",
    "note": "fg_faint",
}


class LogList(ScrollFrame):
    """The master list. Rows are built once and appended, never rebuilt."""

    ROW_HEIGHT = 34

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        on_select: Callable[[int], None],
    ) -> None:
        super().__init__(master, palette)
        self._palette = palette
        self._fonts = fonts
        self._on_select = on_select
        self._rows: list[tk.Frame] = []
        self._selected = -1
        self._entries: list[TraceEntry] = []
        self._empty = tk.Label(
            self.inner,
            text=(
                "Belum ada aktivitas. Log akan terisi begitu agen memanggil LLM - "
                "setiap permintaan, balasan, dan keputusan tercatat di sini."
            ),
            bg=palette.surface,
            fg=palette.fg_faint,
            font=fonts.small,
            justify="left",
            anchor="w",
            wraplength=340,
        )
        self._empty.pack(fill="x", padx=8, pady=10)

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        for row in self._rows:
            row.destroy()
        self._rows.clear()
        self._entries.clear()
        self._selected = -1
        self._empty.pack(fill="x", padx=8, pady=10)

    def append(self, entries: Sequence[TraceEntry]) -> None:
        """Add new entries to the list view."""
        if entries and self._empty.winfo_ismapped():
            self._empty.pack_forget()
        for entry in entries:
            index = len(self._entries)
            self._entries.append(entry)
            row = self._build_row(entry, index)
            row.pack(fill="x", padx=4, pady=1)
            self._rows.append(row)
        self.scroll_to_bottom()

    def set_entries(self, entries: Sequence[TraceEntry]) -> None:
        """Replace the visible list (used when a filter changes)."""
        self.reset()
        self.append(entries)

    def select(self, index: int) -> None:
        self._selected = index
        for position, row in enumerate(self._rows):
            row.configure(bg=self._palette.surface_high if position == index else self._palette.surface)
            for child in row.winfo_children():
                try:
                    child.configure(bg=row["bg"])
                except tk.TclError:  # pragma: no cover - canvas children
                    pass
        if 0 <= index < len(self._entries):
            self._on_select(index)

    # ------------------------------------------------------------------ paint
    def _build_row(self, entry: TraceEntry, index: int) -> tk.Frame:
        row = tk.Frame(self.inner, bg=self._palette.surface, cursor="hand2")
        accent = getattr(
            self._palette, STATUS_COLOURS.get(getattr(entry, "status", "") or entry.outcome, "fg_faint"),
            self._palette.fg_muted,
        )
        bar = tk.Frame(row, bg=accent, width=2, height=self.ROW_HEIGHT - 4)
        bar.pack(side="left", fill="y", padx=(2, 6), pady=2)
        body = tk.Frame(row, bg=self._palette.surface)
        body.pack(side="left", fill="x", expand=True, pady=2)

        top = tk.Label(
            body,
            text=getattr(entry, "at", ""),
            bg=self._palette.surface,
            fg=self._palette.fg_faint,
            font=self._fonts.mono_small,
            anchor="w",
        )
        top.pack(anchor="w")
        name = entry.stage or "-"
        title = STAGE_INFO[name].title if name in STAGE_INFO else name
        if entry.kind == "exchange":
            meta = f"{title} \u00b7 {truncate(entry.model, 20)} \u00b7 {entry.duration_s:.1f}s"
        else:
            meta = f"{title} \u00b7 {entry.outcome}"
        tk.Label(
            body,
            text=meta,
            bg=self._palette.surface,
            fg=self._palette.fg,
            font=self._fonts.small,
            anchor="w",
        ).pack(anchor="w")
        right = tk.Label(
            row,
            text=getattr(entry, "status", "") or getattr(entry, "outcome", ""),
            bg=self._palette.surface,
            fg=accent,
            font=self._fonts.label,
            anchor="e",
        )
        right.pack(side="right", padx=6)

        for widget in (row, body, top, right):
            widget.bind("<Button-1>", lambda _e, i=index: self.select(i))
        return row


class LogDetail(Card):
    """The selected entry, rendered in full."""

    def __init__(self, master, palette: Palette, fonts: Fonts) -> None:
        super().__init__(master, palette, fonts, eyebrow="Detail", title="Pilih satu baris")
        self._palette = palette
        self._fonts = fonts

        head = tk.Frame(self.body, bg=palette.surface)
        head.pack(fill="x")
        self.status_badge = Badge(head, "menunggu", palette, fonts)
        self.status_badge.pack(side="left")
        self.token_label = tk.Label(
            head, text="", bg=palette.surface, fg=palette.fg_faint,
            font=fonts.mono_small, anchor="e",
        )
        self.token_label.pack(side="right")

        self.meta = KeyValueLite(self.body, palette, fonts)
        self.meta.pack(fill="x", pady=(6, 6))
        Divider(self.body, palette).pack(fill="x", pady=(0, 6))

        self.sections = ScrollFrame(self.body, palette)
        self.sections.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ api
    def show(self, entry: TraceEntry | None) -> None:
        for child in self.sections.inner.winfo_children():
            child.destroy()
        self.meta.clear()
        if entry is None:
            self.set_title("Pilih satu baris")
            self.status_badge.set("menunggu", self._palette.fg_faint)
            self.token_label.configure(text="")
            return

        status = getattr(entry, "status", "") or getattr(entry, "outcome", "")
        label = entry.stage
        title = STAGE_INFO[label].title if label in STAGE_INFO else (label or "Catatan sesi")
        self.set_title(f"{title} \u00b7 {getattr(entry, 'at', '')}")
        self.status_badge.set(
            status,
            getattr(self._palette, STATUS_COLOURS.get(status, "fg_muted"), self._palette.fg_muted),
        )

        if entry.kind == "exchange":
            self._render_exchange(entry)
        else:
            self._render_decision(entry)

    # ------------------------------------------------------------------ exchange
    #: Max visible lines per text block. A system prompt is thousands of
    #: characters, and letting it take the whole pane pushes the reply - the thing
    #: people came to read - off the bottom of the window.
    MAX_BLOCK_LINES = 12

    def _render_exchange(self, entry: LLMExchange) -> None:
        self.meta.add("provider", entry.provider)
        self.meta.add("model", entry.model)
        self.meta.add("label", entry.label)
        self.meta.add("durasi", f"{entry.duration_s:.2f}s")
        self.meta.add(
            "parameter",
            f"T={entry.temperature} \u00b7 max_tokens={entry.max_tokens} \u00b7 "
            f"json={entry.json_mode} \u00b7 stream={entry.stream}",
            mono=False,
        )
        if entry.finish_reason:
            self.meta.add("finish_reason", entry.finish_reason)
        self.token_label.configure(
            text=(
                f"prompt {entry.prompt_tokens} \u00b7 completion {entry.completion_tokens} "
                f"\u00b7 total {entry.total_tokens}"
            )
        )
        if entry.error:
            self._block("Kesalahan", entry.error, "red")
        system_prompt = entry.system_prompt
        if system_prompt:
            self._block("System prompt (yang dikirim)", system_prompt, "fg_muted")
        user_message = entry.user_message
        if user_message:
            self._block("Pesan user (yang dikirim)", user_message, "fg")
        if entry.response:
            self._block("Balasan LLM", entry.response, "cyan")
        else:
            self._block("Balasan LLM", "(kosong)", "amber")
        if entry.reasoning:
            self._block("Reasoning tersembunyi", entry.reasoning, "violet")
        # Bring the answer into view: reading the prompt is what you do second.
        self.sections.scroll_to_bottom()

    # ------------------------------------------------------------------ decision
    def _render_decision(self, entry: AgentDecision) -> None:
        self.meta.add("stage", entry.stage)
        self.meta.add("percobaan", str(entry.attempt))
        self.meta.add("hasil", entry.outcome)
        if entry.model:
            self.meta.add("model", entry.model)
        self.token_label.configure(text="")
        self._block("Keputusan", entry.detail, "fg")
        if entry.problems:
            self._block(
                f"Catatan ({len(entry.problems)})",
                "\n".join(f"\u2022 {p}" for p in entry.problems),
                "amber",
            )

    # ------------------------------------------------------------------ helpers
    def _block(self, title: str, text: str, colour: str) -> None:
        holder = tk.Frame(self.sections.inner, bg=self._palette.surface)
        holder.pack(fill="x", pady=(0, 8))
        SectionLabel(
            holder, title, self._palette, self._fonts,
            colour=getattr(self._palette, colour, self._palette.fg_muted),
        ).pack(anchor="w")
        box = tk.Text(
            holder,
            height=max(3, min(self.MAX_BLOCK_LINES, text.count("\n") + 2)),
            wrap="word",
            bg=self._palette.surface_alt,
            fg=getattr(self._palette, colour, self._palette.fg),
            font=self._fonts.mono_small,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=self._palette.line_soft,
            padx=8,
            pady=6,
        )
        box.pack(fill="x", pady=(2, 0))
        box.insert("1.0", text)
        box.configure(state="disabled")
        copy = PillButton(
            holder,
            "Salin",
            lambda: _copy(box, text),
            self._palette,
            self._fonts,
            kind="ghost",
            height=20,
            glyph="\u29c9",
            tooltip=f"Copy {title}",
        )
        copy.pack(anchor="e", pady=(2, 0))


class KeyValueLite(tk.Frame):
    """Compact key/value grid; the studio's KeyValue is fixed-width, this wraps."""

    def __init__(self, master, palette: Palette, fonts: Fonts) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts
        self._row = 0
        self.columnconfigure(1, weight=1)

    def add(self, key: str, value: str, *, mono: bool = True) -> None:
        tk.Label(
            self, text=key.upper(), bg=self._palette.surface, fg=self._palette.fg_faint,
            font=self._fonts.label, anchor="w",
        ).grid(row=self._row, column=0, sticky="w", padx=(0, 10), pady=1)
        tk.Label(
            self, text=value, bg=self._palette.surface, fg=self._palette.fg,
            font=self._fonts.mono_small if mono else self._fonts.small,
            anchor="w", justify="left",
        ).grid(row=self._row, column=1, sticky="w", pady=1)
        self._row += 1

    def clear(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._row = 0


def _copy(widget: tk.Misc, text: str) -> None:
    widget.clipboard_clear()
    widget.clipboard_append(text)
    widget.update_idletasks()


class AgentLogPanel(tk.Frame):
    """Toolbar + master/detail body. Owns no data: it asks its source for entries.

    Reading through a callable (rather than holding a list) is what lets the panel
    show a live run: the recorder is appended to by worker threads while the UI
    polls by index.
    """

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        source: Callable[[int], tuple[list[TraceEntry], int]] | None = None,
        on_export: Callable[[], None] | None = None,
        on_clear: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts
        self._source = source
        self._cursor = 0
        self._known_total = 0
        self._mode = "all"
        self._stage = ""
        self._all: list[TraceEntry] = []

        toolbar = tk.Frame(self, bg=palette.surface)
        toolbar.pack(fill="x", padx=M.pad_lg, pady=(14, 6))
        tk.Label(
            toolbar, text="Log Agen", bg=palette.surface, fg=palette.fg,
            font=fonts.display_s, anchor="w",
        ).pack(side="left")

        self.summary = tk.Label(
            toolbar, text="", bg=palette.surface, fg=palette.fg_faint,
            font=fonts.mono_small, anchor="e",
        )
        self.summary.pack(side="right")

        controls = tk.Frame(self, bg=palette.surface)
        controls.pack(fill="x", padx=M.pad_lg, pady=(0, 8))
        self.filter_buttons: dict[str, PillButton] = {}
        for key, label in FILTERS:
            button = PillButton(
                controls, label, (lambda k=key: self.set_filter(k)), palette, fonts,
                kind="ghost", height=26,
            )
            button.pack(side="left", padx=(0, 6))
            self.filter_buttons[key] = button

        self.stage_var = tk.StringVar(value="semua tahap")
        self.stage_menu = tk.OptionMenu(controls, self.stage_var, "semua tahap")
        self.stage_menu.configure(
            bg=palette.surface_alt, fg=palette.fg, font=fonts.small,
            highlightthickness=0, bd=0, activebackground=palette.surface_high,
            activeforeground=palette.accent, width=22, anchor="w",
        )
        self.stage_menu["menu"].configure(
            bg=palette.surface_alt, fg=palette.fg, font=fonts.small,
            activebackground=palette.accent, activeforeground=palette.bg,
        )
        self._fill_stage_menu()
        self.stage_menu.pack(side="left", padx=(0, 6), pady=1)

        PillButton(
            controls, "Ekspor", on_export or (lambda: None), palette, fonts,
            kind="ghost", height=26, glyph="\u2913",
            tooltip="Tulis log ke berkas JSONL + Markdown",
        ).pack(side="right")
        PillButton(
            controls, "Bersihkan", on_clear or (lambda: None), palette, fonts,
            kind="ghost", height=26, glyph="\u2715",
            tooltip="Kosongkan log sesi ini (tidak menghapus berkas run)",
        ).pack(side="right", padx=(0, 6))

        self._body = tk.Frame(self, bg=palette.surface)
        self._body.pack(fill="both", expand=True, padx=M.pad_lg, pady=(0, M.pad_lg))

        self._list_card = Card(self._body, palette, fonts, eyebrow="Riwayat", title="Aktivitas")
        self.list = LogList(self._list_card.body, palette, fonts, on_select=self._on_select)
        self.list.pack(fill="both", expand=True)

        self._detail_card = LogDetail(self._body, palette, fonts)
        self._stacked = False
        self._apply_layout(side_by_side=True)
        # Re-arranging on resize is what makes this panel usable in a narrow pane:
        # master-detail side by side leaves the detail around 100px there, which is
        # unreadable for a prompt. Stacked, both get the full width.
        self._body.bind("<Configure>", self._on_body_resize, add="+")

    # ------------------------------------------------------------------ layout
    #: Below this body width, stacking beats side-by-side.
    NARROW_WIDTH = 620
    #: Master list width when side by side.
    LIST_WIDTH = 300

    def _on_body_resize(self, event) -> None:
        narrow = event.width < self.NARROW_WIDTH
        if narrow != self._stacked:
            self._apply_layout(not narrow)

    def _apply_layout(self, side_by_side: bool) -> None:
        """Arrange master and detail horizontally or vertically."""
        for widget in (self._list_card, self._detail_card):
            widget.pack_forget()
        self._list_card.pack_propagate(True)
        if side_by_side:
            self._list_card.configure(width=self.LIST_WIDTH)
            self._list_card.pack_propagate(False)
            self._list_card.pack(side="left", fill="y")
            self._detail_card.pack(side="left", fill="both", expand=True, padx=(12, 0))
        else:
            self._list_card.pack(side="top", fill="x")
            self._list_card.configure(height=200)
            self._detail_card.pack(side="top", fill="both", expand=True, pady=(12, 0))
        self._stacked = not side_by_side

    # ------------------------------------------------------------------ data
    def refresh(self, *, force: bool = False) -> int:
        """Pull new entries from the source. Returns how many were added."""
        if self._source is None:
            return 0
        if force:
            self._cursor = 0
            self._all = []
            self.list.reset()
        added, total = self._source(self._cursor)
        self._cursor = total
        self._known_total = max(self._known_total, total)
        if added:
            self._all.extend(added)
            # A filter or stage selection shows a live view of everything known so
            # far, so a new entry never silently fails to appear.
            self._apply_filter()
        self._update_summary()
        return len(added)

    def attach(self, source: Callable[[int], tuple[list[TraceEntry], int]]) -> None:
        self._source = source
        self.refresh(force=True)

    def _apply_filter(self) -> None:
        visible = filter_entries(self._all, self._mode, self._stage)
        self.list.set_entries(visible)
        if visible:
            self.list.select(len(visible) - 1)

    def _update_summary(self) -> None:
        self.summary.configure(text=summarise(self._all) if self._all else "kosong")

    # ------------------------------------------------------------------ controls
    def set_filter(self, mode: str) -> None:
        self._mode = mode
        for key, button in self.filter_buttons.items():
            button.configure(
                bg=self._palette.accent if key == mode else self._palette.surface_alt
            )
            button.redraw()
        self._apply_filter()

    def set_stage(self, stage: str) -> None:
        self._stage = "" if stage == "semua tahap" else stage
        self._apply_filter()

    def _fill_stage_menu(self) -> None:
        menu = self.stage_menu["menu"]
        menu.delete(0, "end")
        menu.add_command(
            label="semua tahap", command=lambda: (self.stage_var.set("semua tahap"), self.set_stage("semua tahap"))
        )
        for stage in STAGE_ORDER:
            title = f"{STAGE_INFO[stage].title}"
            menu.add_command(
                label=title,
                command=lambda s=stage, t=title: (self.stage_var.set(t), self.set_stage(s)),
            )

    def _on_select(self, index: int) -> None:
        visible = filter_entries(self._all, self._mode, self._stage)
        entry = visible[index] if 0 <= index < len(visible) else None
        self.detail_show(entry)

    def detail_show(self, entry: TraceEntry | None) -> None:
        for child in self.winfo_children():
            if isinstance(child, tk.Frame):
                for sub in child.winfo_children():
                    if isinstance(sub, LogDetail):
                        sub.show(entry)
                        return
        # fall back: find it anywhere in the tree
        found = _find_detail(self)
        if found is not None:
            found.show(entry)

    def clear(self) -> None:
        """Empty the view without resurrecting history on the next poll.

        Resetting the cursor to 0 would make the very next refresh re-read every
        entry the recorder still holds - so "clear" would visibly undo itself a
        fraction of a second later. The cursor is advanced past what already exists
        instead, which is the honest meaning of clearing a view: show nothing from
        here on.
        """
        known = self._known_total
        self._all.clear()
        self._cursor = max(self._cursor, known)
        self.list.reset()
        self._update_summary()
        self.detail_show(None)

    def forget(self) -> None:
        """Reset the cursor too, for when the underlying recorder was cleared."""
        self._all.clear()
        self._cursor = 0
        self._known_total = 0
        self.list.reset()
        self._update_summary()
        self.detail_show(None)


def _find_detail(widget: tk.Misc) -> LogDetail | None:
    for child in widget.winfo_children():
        if isinstance(child, LogDetail):
            return child
        found = _find_detail(child)
        if found is not None:
            return found
    return None


__all__ = ["AgentLogPanel", "LogDetail", "LogList", "FILTERS"]

"""The conversation view: transcript, streaming, and the multi-line composer.

Implemented on a ``tk.Text`` rather than a stack of frames: text reflows for free
when the window resizes, stays selectable and copyable, and streaming a token at
a time does not rebuild the widget tree. Message identity is carried by tags, so a
user turn, a consultant reply and a system note are visually distinct without
extra chrome.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from .theme import Fonts, Metrics, Palette, truncate
from .widgets import PillButton

ROLE_SPECS = {
    "user": ("ANDA", "accent"),
    "assistant": ("KONSULTAN", "cyan"),
    "system": ("SISTEM", "fg_faint"),
    "note": ("CATATAN", "fg_muted"),
    "reviewer": ("REVIEWER", "rose"),
    "craft": ("AGEN", "violet"),
}


class Composer(tk.Frame):
    """Multi-line input with a send affordance and Enter/Ctrl+Enter semantics."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        on_send: Callable[[str], None],
        placeholder: str = "",
        rows: int = 3,
    ) -> None:
        super().__init__(master, bg=palette.line, bd=0, highlightthickness=0)
        self._palette = palette
        self._fonts = fonts
        self._on_send = on_send
        self._placeholder = placeholder

        inner = tk.Frame(self, bg=palette.surface)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.text = tk.Text(
            inner,
            height=rows,
            wrap="word",
            bg=palette.surface_alt,
            fg=palette.fg,
            insertbackground=palette.accent,
            selectbackground=palette.accent_dim,
            selectforeground=palette.fg,
            font=fonts.body,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.line_soft,
            highlightcolor=palette.accent,
            padx=10,
            pady=8,
        )
        self.text.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        self.text.bind("<KeyRelease>", lambda _e: self._sync_counter())
        self.text.bind("<Return>", self._on_return)
        self.text.bind("<Shift-Return>", lambda _e: None)
        self.text.bind("<Control-Return>", lambda _e: self._send())

        footer = tk.Frame(inner, bg=palette.surface)
        footer.pack(fill="x", padx=10, pady=(0, 10))
        self.hint = tk.Label(
            footer,
            text="Enter kirim \u00b7 Shift+Enter baris baru",
            bg=palette.surface,
            fg=palette.fg_faint,
            font=fonts.small,
            anchor="w",
        )
        self.hint.pack(side="left")
        self.counter = tk.Label(
            footer,
            text="0",
            bg=palette.surface,
            fg=palette.fg_faint,
            font=fonts.mono_small,
            anchor="e",
        )
        self.counter.pack(side="right", padx=(0, 8))
        self.send_button = PillButton(
            footer,
            "Kirim",
            self._send,
            palette,
            fonts,
            kind="primary",
            glyph="\u21b5",
            height=28,
            tooltip="Kirim pesan ke konsultan",
        )
        self.send_button.pack(side="right")

        self._show_placeholder()

    # ------------------------------------------------------------------ api
    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.text.configure(state=state)
        self.send_button.set_enabled(enabled)

    def focus_input(self) -> None:
        self.text.focus_set()

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def clear(self) -> None:
        self.text.delete("1.0", "end")
        self._show_placeholder()

    def insert_text(self, value: str) -> None:
        self._clear_placeholder()
        self.text.insert("end", value)
        self._sync_counter()
        self.focus_input()

    # ------------------------------------------------------------------ plumbing
    def _show_placeholder(self) -> None:
        if not self._placeholder:
            return
        self.text.configure(fg=self._palette.fg_faint)
        self.text.insert("1.0", self._placeholder)
        self._placeholder_shown = True

    def _clear_placeholder(self) -> None:
        if getattr(self, "_placeholder_shown", False):
            self.text.delete("1.0", "end")
            self.text.configure(fg=self._palette.fg)
            self._placeholder_shown = False

    def _sync_counter(self) -> None:
        value = self.get_text()
        if value and getattr(self, "_placeholder_shown", False):
            self._clear_placeholder()
            value = ""
        self.counter.configure(text=str(len(value)))

    def _on_return(self, _event) -> str:
        # Enter sends; Shift+Enter is allowed through to insert a newline.
        self._send()
        return "break"

    def _send(self) -> None:
        self._clear_placeholder()
        value = self.get_text().strip()
        if not value:
            return
        self.clear()
        self._on_send(value)


class ChatView(tk.Frame):
    """Scrollable transcript with streaming support."""

    def __init__(self, master, palette: Palette, fonts: Fonts) -> None:
        super().__init__(master, bg=palette.line, bd=0, highlightthickness=0)
        self._palette = palette
        self._fonts = fonts
        self._stream_tag = ""
        self._stream_open = False

        inner = tk.Frame(self, bg=palette.surface)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.text = tk.Text(
            inner,
            wrap="word",
            bg=palette.surface,
            fg=palette.fg,
            insertbackground=palette.accent,
            selectbackground=palette.surface_high,
            selectforeground=palette.fg,
            font=fonts.chat,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=14,
            pady=12,
            spacing1=1,
            spacing3=3,
            cursor="arrow",
        )
        self.scrollbar = tk.Scrollbar(
            inner,
            orient="vertical",
            command=self.text.yview,
            width=10,
            bg=palette.surface_high,
            troughcolor=palette.surface,
            activebackground=palette.accent,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.text.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self._configure_tags()
        self.text.configure(state="disabled")

    # ------------------------------------------------------------------ tags
    def _configure_tags(self) -> None:
        p, f = self._palette, self._fonts
        self.text.tag_configure("role", font=f.label, foreground=p.fg_faint, spacing1=10, spacing3=2)
        self.text.tag_configure("body", font=f.chat, foreground=p.fg, lmargin1=0, lmargin2=0, spacing3=4)
        self.text.tag_configure("body_user", font=f.chat, foreground=p.fg, lmargin1=10, lmargin2=10)
        self.text.tag_configure("meta", font=f.mono_small, foreground=p.fg_faint)
        self.text.tag_configure("strong", font=f.body_bold, foreground=p.fg)
        self.text.tag_configure("bullet", font=f.chat, foreground=p.fg, lmargin1=18, lmargin2=28)
        self.text.tag_configure("quote", font=f.chat, foreground=p.fg_muted, lmargin1=18, lmargin2=18)
        self.text.tag_configure("code", font=f.mono_small, foreground=p.cyan, background=p.surface_alt)
        self.text.tag_configure("rule", foreground=p.line, spacing1=6, spacing3=6)
        for role, (_label, colour) in ROLE_SPECS.items():
            self.text.tag_configure(f"role_{role}", foreground=getattr(p, colour, p.fg_muted))

    # ------------------------------------------------------------------ api
    def add_message(self, role: str, text: str, *, at: str = "") -> None:
        """Append one complete message."""
        self.text.configure(state="normal")
        self._write_role(role, at)
        self._write_body(text, role)
        self.text.configure(state="disabled")
        self._scroll_end()

    def add_note(self, text: str, *, colour: str = "") -> None:
        """Append a system/status line, no role header."""
        self.text.configure(state="normal")
        tag = "meta"
        if colour:
            name = f"note_{colour}"
            self.text.tag_configure(name, font=self._fonts.small, foreground=getattr(self._palette, colour, colour))
            tag = name
        self.text.insert("end", f"{text}\n", tag)
        self.text.configure(state="disabled")
        self._scroll_end()

    def add_rule(self, label: str = "") -> None:
        self.text.configure(state="normal")
        line = "\u2500" * 8
        self.text.insert("end", f"{line} {label} {line}\n" if label else f"{line}\n", "rule")
        self.text.configure(state="disabled")
        self._scroll_end()

    def begin_stream(self, role: str = "assistant") -> None:
        """Start a message that will be filled in by successive chunks."""
        self.text.configure(state="normal")
        self._write_role(role, "")
        self._stream_tag = f"stream_{role}"
        self.text.tag_configure(self._stream_tag, font=self._fonts.chat, foreground=self._palette.fg)
        self._stream_open = True
        self.text.configure(state="disabled")
        self._scroll_end()

    def stream_chunk(self, chunk: str) -> None:
        if not self._stream_open:
            self.begin_stream("assistant")
        self.text.configure(state="normal")
        self.text.insert("end", chunk, self._stream_tag)
        self.text.configure(state="disabled")
        self._scroll_end()

    def end_stream(self) -> None:
        if not self._stream_open:
            return
        self.text.configure(state="normal")
        self.text.insert("end", "\n", "body")
        self.text.configure(state="disabled")
        self._stream_open = False
        self._stream_tag = ""
        self._scroll_end()

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    # ------------------------------------------------------------------ rendering
    def _write_role(self, role: str, at: str) -> None:
        label, _colour = ROLE_SPECS.get(role, (role.upper(), "fg_muted"))
        stem = "\u25b8"
        stamp = f"   {at}" if at else ""
        self.text.insert("end", f"{stem} {label}{stamp}\n", ("role", f"role_{role}"))

    def _write_body(self, text: str, role: str) -> None:
        body_tag = "body_user" if role == "user" else "body"
        for raw_line in str(text or "").split("\n"):
            line = raw_line.rstrip()
            if not line.strip():
                self.text.insert("end", "\n", body_tag)
                continue
            stripped = line.strip()
            if stripped.startswith(("- ", "* ", "\u2022 ")):
                self.text.insert("end", f"\u2022 {stripped[2:].strip()}\n", "bullet")
            elif stripped.startswith("> "):
                self.text.insert("end", f"{stripped[2:].strip()}\n", "quote")
            elif stripped.startswith("[") and stripped.endswith("]"):
                self.text.insert("end", f"{stripped}\n", "code")
            elif stripped.startswith("#"):
                self.text.insert(
                    "end", f"{stripped.lstrip('# ').strip()}\n", ("strong", body_tag)
                )
            else:
                self.text.insert("end", f"{line}\n", body_tag)

    def _scroll_end(self) -> None:
        self.text.see("end")


class ChatPanel(tk.Frame):
    """Transcript + composer + the two actions that drive the app."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        on_send: Callable[[str], None],
        on_run: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts

        header = tk.Frame(self, bg=palette.surface)
        header.pack(fill="x", padx=Metrics().pad_lg, pady=(14, 8))
        tk.Label(
            header,
            text="Ruang Konsultasi",
            bg=palette.surface,
            fg=palette.fg,
            font=fonts.display_s,
            anchor="w",
        ).pack(side="left")
        self.turn_label = tk.Label(
            header,
            text="",
            bg=palette.surface,
            fg=palette.fg_faint,
            font=fonts.mono_small,
            anchor="e",
        )
        self.turn_label.pack(side="right")

        self.view = ChatView(self, palette, fonts)
        self.view.pack(fill="both", expand=True, padx=Metrics().pad_lg)

        self.composer = Composer(
            self,
            palette,
            fonts,
            on_send=on_send,
            placeholder=(
                "Ceritakan lagu yang kamu inginkan\u2026 misalnya: "
                "\"lagu galau tapi tetap bisa dipakai buat perjalanan pulang, "
                "penyanyi perempuan, ada bagian melayu\""
            ),
            rows=3,
        )
        self.composer.pack(fill="x", padx=Metrics().pad_lg, pady=(10, 6))

        actions = tk.Frame(self, bg=palette.surface)
        actions.pack(fill="x", padx=Metrics().pad_lg, pady=(0, 14))
        self.run_button = PillButton(
            actions,
            "Bangun Prompt Suno",
            on_run,
            palette,
            fonts,
            kind="primary",
            glyph="\u25b6",
            height=34,
            tooltip="Jalankan 9 agen sampai prompt Suno selesai",
        )
        self.run_button.pack(side="left")
        self.stop_button = PillButton(
            actions,
            "Hentikan",
            on_stop,
            palette,
            fonts,
            kind="danger",
            glyph="\u25a0",
            height=34,
            tooltip="Berhenti setelah tahap yang sedang berjalan",
        )
        self.stop_button.pack(side="left", padx=8)
        self.stop_button.set_enabled(False)

    def set_busy(self, busy: bool) -> None:
        self.run_button.set_enabled(not busy)
        self.stop_button.set_enabled(busy)
        self.composer.set_enabled(not busy)

    def set_turns(self, used: int, budget: int) -> None:
        if used <= 0:
            self.turn_label.configure(text="")
            return
        self.turn_label.configure(text=truncate(f"putaran {used}/{budget}", 24))


__all__ = ["ChatPanel", "ChatView", "Composer"]

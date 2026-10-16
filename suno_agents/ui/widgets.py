"""Hand-drawn Tk widgets.

Tk's built-in widgets are instantly recognisable and impossible to style
consistently across platforms, so the panels, buttons, meters and the signal rail
are all drawn on canvases. Each widget takes the palette and fonts as arguments
and has no other dependencies, which keeps them testable and reusable.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Iterable, Sequence

from .theme import Fonts, Metrics, Palette, STAGE_GLYPHS, STATUS_COLOURS, truncate


class _Hoverable:
    """Mixin adding enter/leave/press bookkeeping to a canvas widget."""

    HOVER: str = "hover"
    NORMAL: str = "normal"
    DISABLED: str = "disabled"
    PRESSED: str = "pressed"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - tk plumbing
        super().__init__(*args, **kwargs)
        self._state = self.NORMAL
        self._enabled = True
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, _event) -> None:
        if not self._enabled:
            return
        self._state = self.HOVER
        try:
            self.configure(cursor="hand2")
        except tk.TclError:  # pragma: no cover
            pass
        self.redraw()

    def _on_leave(self, _event) -> None:
        self._state = self.NORMAL
        try:
            self.configure(cursor="")
        except tk.TclError:  # pragma: no cover
            pass
        self.redraw()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._state = self.NORMAL if enabled else self.DISABLED
        try:
            self.configure(cursor="hand2" if enabled else "")
        except tk.TclError:  # pragma: no cover
            pass
        self.redraw()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def redraw(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# buttons
# --------------------------------------------------------------------------- #


class PillButton(_Hoverable, tk.Canvas):
    """Flat, rounded, keyboard-accessible button drawn on a canvas."""

    def __init__(
        self,
        master,
        text: str,
        command: Callable[[], None],
        palette: Palette,
        fonts: Fonts,
        *,
        kind: str = "primary",  # primary | ghost | danger
        width: int | None = None,
        height: int = 30,
        radius: int = 6,
        glyph: str = "",
        tooltip: str = "",
    ) -> None:
        self._text = text
        self._command = command
        self._palette = palette
        self._fonts = fonts
        self._kind = kind
        self._radius = radius
        self._glyph = glyph
        self._tooltip_text = tooltip
        label = f"{glyph}  {text}".strip() if glyph else text
        font = fonts.body_bold if kind == "primary" else fonts.body
        measured = font.measure(label) + 26
        super().__init__(
            master,
            width=width or measured,
            height=height,
            highlightthickness=0,
            bd=0,
            bg=palette.surface,
            takefocus=1,
        )
        self._focused = False
        self.bind("<Button-1>", self._on_press, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")
        self.bind("<Return>", lambda _e: self._invoke(), add="+")
        self.bind("<space>", lambda _e: self._invoke(), add="+")
        self.bind("<FocusIn>", self._on_focus_in, add="+")
        self.bind("<FocusOut>", self._on_focus_out, add="+")
        self.bind("<Configure>", lambda _e: self.redraw(), add="+")
        if tooltip:
            Tooltip(self, tooltip, palette, fonts)
        self.redraw()

    # ------------------------------------------------------------------ state
    def _on_focus_in(self, _event) -> None:
        self._focused = True
        self.redraw()

    def _on_focus_out(self, _event) -> None:
        self._focused = False
        self.redraw()

    def _on_press(self, _event) -> None:
        if not self._enabled:
            return
        self.focus_set()
        self._state = self.PRESSED
        self.redraw()

    def _on_release(self, event) -> None:
        if not self._enabled:
            return
        inside = 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height()
        self._state = self.HOVER if inside else self.NORMAL
        self.redraw()
        if inside:
            self._invoke()

    def _invoke(self) -> None:
        if self._enabled and self._command is not None:
            self._command()

    def set_text(self, text: str, glyph: str | None = None) -> None:
        self._text = text
        if glyph is not None:
            self._glyph = glyph
        self.redraw()

    # ------------------------------------------------------------------ paint
    def _colours(self) -> tuple[str, str, str]:
        """Return ``(fill, outline, text)`` for the current state."""
        p = self._palette
        if not self._enabled:
            return p.surface_alt, p.line_soft, p.fg_faint
        if self._kind == "primary":
            fill = {
                self.NORMAL: p.accent,
                self.HOVER: "#ffbe45",
                self.PRESSED: p.accent_dim,
            }[self._state]
            return fill, fill, "#12151a"
        if self._kind == "danger":
            fill = {
                self.NORMAL: p.surface_alt,
                self.HOVER: "#3a2226",
                self.PRESSED: "#4a2a2f",
            }[self._state]
            return fill, p.red, p.red
        fill = {
            self.NORMAL: p.surface_alt,
            self.HOVER: p.surface_high,
            self.PRESSED: p.surface,
        }[self._state]
        return fill, p.line, p.fg

    def redraw(self) -> None:
        self.delete("all")
        width = self.winfo_width() or int(self["width"])
        height = self.winfo_height() or int(self["height"])
        fill, outline, fg = self._colours()
        self._round_rect(1, 1, width - 1, height - 1, self._radius, fill, outline)
        if self._focused:
            self._round_rect(2, 2, width - 3, height - 3, self._radius - 1, "", self._palette.cyan)
        label = f"{self._glyph}  {self._text}".strip() if self._glyph else self._text
        font = self._fonts.body_bold if self._kind == "primary" else self._fonts.body
        self.create_text(
            width // 2 + 1,
            height // 2,
            text=truncate(label, max(6, (width - 20) // 7)),
            fill=fg,
            font=font,
            anchor="center",
        )

    def _round_rect(self, x0, y0, x1, y1, radius, fill, outline) -> None:
        radius = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
        points = [
            x0 + radius, y0, x1 - radius, y0, x1, y0, x1, y0 + radius,
            x1, y1 - radius, x1, y1, x1 - radius, y1, x0 + radius, y1,
            x0, y1, x0, y1 - radius, x0, y0 + radius, x0, y0,
        ]
        self.create_polygon(
            points, smooth=True, splinesteps=12, fill=fill or "", outline=outline or ""
        )


class IconButton(PillButton):
    """Square glyph-only button for toolbars."""

    def __init__(self, master, glyph: str, command, palette, fonts, *, tooltip: str = "", size: int = 30):
        super().__init__(
            master,
            text="",
            command=command,
            palette=palette,
            fonts=fonts,
            kind="ghost",
            width=size,
            height=size,
            glyph=glyph,
            tooltip=tooltip,
        )


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #


class Card(tk.Frame):
    """Surface panel with a hairline border and an optional eyebrow + title."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        title: str = "",
        eyebrow: str = "",
        pad: int = 12,
        **kwargs,
    ) -> None:
        super().__init__(
            master, bg=palette.line, bd=0, highlightthickness=0, **kwargs
        )
        self._palette = palette
        self._fonts = fonts
        self.inner = tk.Frame(self, bg=palette.surface, bd=0, highlightthickness=0)
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)
        self.header = tk.Frame(self.inner, bg=palette.surface)
        self.header.pack(fill="x", padx=pad, pady=(pad, 0))
        if eyebrow:
            tk.Label(
                self.header,
                text=eyebrow.upper(),
                bg=palette.surface,
                fg=palette.fg_faint,
                font=fonts.label,
                anchor="w",
            ).pack(side="top", anchor="w")
        self.title_label: tk.Label | None = None
        if title:
            self.title_label = tk.Label(
                self.header,
                text=title,
                bg=palette.surface,
                fg=palette.fg,
                font=fonts.display_s,
                anchor="w",
            )
            self.title_label.pack(side="top", anchor="w")
        self.body = tk.Frame(self.inner, bg=palette.surface)
        self.body.pack(fill="both", expand=True, padx=pad, pady=(8, pad))

    def set_title(self, title: str) -> None:
        if self.title_label is not None:
            self.title_label.configure(text=title)


class Divider(tk.Frame):
    """One-pixel horizontal or vertical rule."""

    def __init__(self, master, palette: Palette, *, vertical: bool = False, colour: str | None = None):
        super().__init__(
            master,
            bg=colour or palette.line_soft,
            height=1 if not vertical else 0,
            width=0 if not vertical else 1,
        )
        if vertical:
            self.configure(width=1, height=0)


class AutoWrapLabel(tk.Label):
    """Label whose wrap width follows its actual width.

    A fixed ``wraplength`` is a silent clipping bug the moment the panel is
    narrower than the author assumed - and here the same widget sits in a rail,
    an inspector and a settings page of three different widths.
    """

    def __init__(self, master, text: str, palette: Palette, fonts: Fonts, **kwargs) -> None:
        self._pad = int(kwargs.pop("inner_pad", 2))
        kwargs.setdefault("justify", "left")
        kwargs.setdefault("anchor", "w")
        kwargs.setdefault("bg", palette.surface)
        kwargs.setdefault("fg", palette.fg_muted)
        kwargs.setdefault("font", fonts.small)
        super().__init__(master, text=text, **kwargs)
        self.bind("<Configure>", self._sync_wrap, add="+")

    def _sync_wrap(self, event) -> None:
        width = max(40, event.width - self._pad)
        # cget returns a Tcl_Obj on some Tk builds, so coerce defensively.
        try:
            current = int(float(str(self.cget("wraplength"))))
        except (TypeError, ValueError):
            current = -1
        if current != width:
            self.configure(wraplength=width)

    def set_text(self, text: str) -> None:
        self.configure(text=text)


class SectionLabel(tk.Label):
    """Letterspaced uppercase micro-label, used to name regions of the UI."""

    def __init__(self, master, text: str, palette: Palette, fonts: Fonts, colour: str | None = None):
        spaced = " ".join(text.upper())
        super().__init__(
            master,
            text=spaced,
            bg=palette.surface,
            fg=colour or palette.fg_faint,
            font=fonts.label,
            anchor="w",
        )


class Badge(tk.Canvas):
    """Small status pill: text, optional glyph, one colour."""

    def __init__(
        self,
        master,
        text: str,
        palette: Palette,
        fonts: Fonts,
        *,
        colour: str | None = None,
        glyph: str = "",
        pad: int = 8,
        height: int = 20,
    ) -> None:
        self._text = text
        self._glyph = glyph
        self._palette = palette
        self._fonts = fonts
        self._colour = colour or palette.fg_muted
        self._pad = pad
        super().__init__(
            master,
            height=height,
            highlightthickness=0,
            bd=0,
            bg=palette.surface,
        )
        self.bind("<Configure>", lambda _e: self.redraw())
        self.redraw()

    def set(self, text: str, colour: str | None = None, glyph: str | None = None) -> None:
        self._text = text
        if colour:
            self._colour = colour
        if glyph is not None:
            self._glyph = glyph
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        label = f"{self._glyph} {self._text}".strip() if self._glyph else self._text
        font = self._fonts.label
        height = int(self["height"])
        width = font.measure(label) + self._pad * 2
        # Only resize when the width actually changed. Setting a canvas width from
        # inside a <Configure> handler re-enters this method forever otherwise,
        # which freezes the whole application.
        if int(float(self["width"])) != width:
            self.configure(width=width)
        self.create_rectangle(
            self._pad, 3, self._pad + width, height - 3, fill=self._colour, outline=""
        )
        self.create_text(
            self._pad,
            height // 2,
            text=label,
            fill=self._palette.bg,
            font=font,
            anchor="w",
        )


# --------------------------------------------------------------------------- #
# data display
# --------------------------------------------------------------------------- #


class LevelMeter(tk.Canvas):
    """Segmented meter. Hue encodes level via the spectral ramp: a signal readout,
    not decoration."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        slots: int = 24,
        height: int = 8,
        caption: str = "",
        show_value: bool = True,
    ) -> None:
        self._palette = palette
        self._fonts = fonts
        self._slots = slots
        self._caption = caption
        self._show_value = show_value
        self._fraction = 0.0
        self._value_text = ""
        super().__init__(
            master,
            height=height + (16 if caption else 0),
            highlightthickness=0,
            bd=0,
            bg=palette.surface,
        )
        self.bind("<Configure>", lambda _e: self.redraw())
        self.redraw()

    def set_value(self, fraction: float, text: str | None = None) -> None:
        self._fraction = max(0.0, min(1.0, float(fraction or 0.0)))
        if text is not None:
            self._value_text = text
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(40, self.winfo_width())
        height = int(self["height"])
        if self._caption:
            self.create_text(
                0, 0, text=self._caption.upper(), fill=self._palette.fg_faint,
                font=self._fonts.label, anchor="nw",
            )
            if self._show_value and self._value_text:
                self.create_text(
                    width, 0, text=self._value_text, fill=self._palette.fg,
                    font=self._fonts.mono_small, anchor="ne",
                )
            top = 16
        else:
            top = 0
        bar_h = max(6, height - top)
        gap = 2
        slot_w = max(2, (width - gap * (self._slots - 1)) / self._slots)
        lit = int(round(self._fraction * self._slots))
        for index in range(self._slots):
            x0 = index * (slot_w + gap)
            on = index < lit
            colour = (
                self._palette.level_colour(index / max(1, self._slots - 1))
                if on
                else self._palette.surface_high
            )
            self.create_rectangle(x0, top, x0 + slot_w, top + bar_h, fill=colour, outline="")


class Chip(tk.Canvas):
    """Read-only tag chip used for descriptors, exclusions and keywords."""

    def __init__(
        self,
        master,
        text: str,
        palette: Palette,
        fonts: Fonts,
        *,
        colour: str | None = None,
        background: str | None = None,
    ) -> None:
        self._text = text
        self._palette = palette
        self._fonts = fonts
        self._colour = colour or palette.fg
        self._bg = background or palette.surface_high
        font = fonts.mono_small
        # tkinter canvases cannot measure before realising; use font metrics.
        width = font.measure(text) + 16
        super().__init__(
            master, width=width, height=20, highlightthickness=0, bd=0, bg=palette.surface
        )
        self.create_rectangle(0, 3, width, 17, fill=self._bg, outline="")
        self.create_text(8, 10, text=text, fill=self._colour, font=font, anchor="w")


class KeyValue(tk.Frame):
    """Monospace key/value readout with the values right-aligned in a column."""

    def __init__(self, master, palette: Palette, fonts: Fonts, pad: int = 8) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts
        self._row = 0
        self._pad = pad
        self.columnconfigure(1, weight=1)

    def add(self, key: str, value: str, *, colour: str | None = None, mono: bool = True) -> None:
        tk.Label(
            self,
            text=key.upper(),
            bg=self._palette.surface,
            fg=self._palette.fg_faint,
            font=self._fonts.label,
            anchor="w",
        ).grid(row=self._row, column=0, sticky="w", pady=1, padx=(0, self._pad))
        tk.Label(
            self,
            text=value,
            bg=self._palette.surface,
            fg=colour or self._palette.fg,
            font=self._fonts.mono_small if mono else self._fonts.small,
            anchor="e",
            justify="right",
        ).grid(row=self._row, column=1, sticky="e", pady=1)
        self._row += 1
        return self

    def clear(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._row = 0


class ScrollFrame(tk.Frame):
    """Vertically scrollable frame with a themed scrollbar and mouse wheel."""

    def __init__(self, master, palette: Palette, **kwargs) -> None:
        super().__init__(master, bg=palette.surface, **kwargs)
        self._palette = palette
        self.canvas = tk.Canvas(
            self, bg=palette.surface, highlightthickness=0, bd=0
        )
        self.scrollbar = tk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview, width=10,
            bg=palette.surface_high, troughcolor=palette.surface,
            activebackground=palette.accent, bd=0, highlightthickness=0,
            relief="flat",
        )
        self.inner = tk.Frame(self.canvas, bg=palette.surface)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind_all_wheel()

    def _on_scroll(self, first: str, last: str) -> None:
        self.scrollbar.set(first, last)
        # Hide the scrollbar entirely when everything fits: less chrome.
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.scrollbar.pack_forget()
        else:
            self.scrollbar.pack(side="right", fill="y")

    def _on_inner_configure(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def bind_all_wheel(self) -> None:
        def on_wheel(event) -> None:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            self.canvas.yview_scroll(delta, "units")

        for widget in (self.canvas, self.inner):
            widget.bind("<MouseWheel>", on_wheel, add="+")

    def scroll_to_bottom(self) -> None:
        self.update_idletasks()
        self.canvas.yview_moveto(1.0)


class Tooltip:
    """Delayed hover tooltip, used sparingly for icon-only controls."""

    def __init__(self, widget: tk.Misc, text: str, palette: Palette, fonts: Fonts, delay: int = 450):
        self._widget = widget
        self._text = text
        self._palette = palette
        self._fonts = fonts
        self._delay = delay
        self._after: str | None = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after = self._widget.after(self._delay, self._show)

    def _cancel(self) -> None:
        if self._after is not None:
            try:
                self._widget.after_cancel(self._after)
            except tk.TclError:  # pragma: no cover
                pass
            self._after = None

    def _show(self) -> None:
        if self._window is not None:
            return
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        window = tk.Toplevel(self._widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(window, bg=self._palette.line, bd=0)
        frame.pack()
        tk.Label(
            frame,
            text=self._text,
            bg=self._palette.surface_high,
            fg=self._palette.fg,
            font=self._fonts.small,
            padx=8,
            pady=4,
            justify="left",
        ).pack(padx=1, pady=1)
        self._window = window

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


# --------------------------------------------------------------------------- #
# the signal rail
# --------------------------------------------------------------------------- #


class StageRail(tk.Canvas):
    """The signature element: a vertical signal path of the nine stages.

    Reads like a mixing-console channel strip: glyph, title, status lamp, and a
    level bar that fills as the stage produces its artifact. The connector between
    stages lights up as the pipeline advances, so the review/rollback loop is
    visible as a signal travelling back up the rail.
    """

    ROW_H = 44
    HEADER_H = 28

    def __init__(
        self,
        master,
        stages: Sequence[str],
        titles: dict[str, str],
        palette: Palette,
        fonts: Fonts,
        *,
        on_select: Callable[[str], None] | None = None,
        width: int = 232,
    ) -> None:
        self._stages = list(stages)
        self._titles = titles
        self._palette = palette
        self._fonts = fonts
        self._on_select = on_select
        self._selected = ""
        self._status: dict[str, str] = {stage: "pending" for stage in self._stages}
        self._levels: dict[str, float] = {stage: 0.0 for stage in self._stages}
        self._notes: dict[str, str] = {}
        self._pulse = 0
        self._pulse_job: str | None = None
        pad = Metrics().pad_sm
        super().__init__(
            master,
            width=width,
            height=self.HEADER_H + self.ROW_H * len(self._stages) + pad,
            highlightthickness=0,
            bd=0,
            bg=palette.surface,
        )
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._on_click)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda _e: self.redraw())
        self.redraw()

    # ------------------------------------------------------------------ api
    def set_status(self, stage: str, status: str, note: str = "") -> None:
        if stage not in self._status:
            return
        self._status[stage] = status
        if note:
            self._notes[stage] = note
        if status == "running":
            self._levels[stage] = 0.15
            self._start_pulse()
        elif status == "done":
            self._levels[stage] = 1.0
            self._stop_pulse_if_idle()
        elif status == "failed":
            self._levels[stage] = 1.0
            self._stop_pulse_if_idle()
        self.redraw()

    def set_level(self, stage: str, fraction: float) -> None:
        if stage in self._levels:
            self._levels[stage] = max(0.0, min(1.0, fraction))
            self.redraw()

    def reset(self) -> None:
        for stage in self._stages:
            self._status[stage] = "pending"
            self._levels[stage] = 0.0
            self._notes[stage] = ""
        self._selected = ""
        self._stop_pulse_if_idle()
        self.redraw()

    def select(self, stage: str) -> None:
        self._selected = stage
        self.redraw()
        if self._on_select is not None:
            self._on_select(stage)

    @property
    def selected(self) -> str:
        return self._selected

    # ------------------------------------------------------------------ pulse
    def _start_pulse(self) -> None:
        if self._pulse_job is not None:
            return
        self._pulse = 0
        self._tick()

    def _tick(self) -> None:
        running = [s for s in self._stages if self._status[s] == "running"]
        if not running:
            self._pulse_job = None
            return
        self._pulse = (self._pulse + 1) % 12
        phase = (self._pulse / 12.0) * 0.5
        for stage in running:
            self._levels[stage] = 0.12 + phase
        self.redraw()
        self._pulse_job = self.after(90, self._tick)

    def _stop_pulse_if_idle(self) -> None:
        if not any(status == "running" for status in self._status.values()):
            if self._pulse_job is not None:
                try:
                    self.after_cancel(self._pulse_job)
                except tk.TclError:  # pragma: no cover
                    pass
                self._pulse_job = None

    # ------------------------------------------------------------------ input
    def _stage_at(self, y: int) -> str:
        index = (y - self.HEADER_H) // self.ROW_H
        if 0 <= index < len(self._stages):
            return self._stages[int(index)]
        return ""

    def _on_click(self, event) -> None:
        stage = self._stage_at(event.y)
        if stage:
            self.select(stage)

    def _on_motion(self, event) -> None:
        stage = self._stage_at(event.y)
        try:
            self.configure(cursor="hand2" if stage else "")
        except tk.TclError:  # pragma: no cover
            pass
        if stage != self._hover:
            self._hover = stage
            self.redraw()

    _hover: str = ""

    # ------------------------------------------------------------------ paint
    def redraw(self) -> None:
        self.delete("all")
        p = self._palette
        width = self.winfo_width() or int(self["width"])
        pad = Metrics().pad_sm
        self.create_text(
            pad,
            6,
            text="P I P E L I N E",
            fill=p.fg_faint,
            font=self._fonts.label,
            anchor="nw",
        )
        running = [s for s in self._stages if self._status[s] == "running"]
        counter = f"{sum(1 for s in self._stages if self._status[s] == 'done')}/{len(self._stages)}"
        self.create_text(
            width - pad, 6, text=counter, fill=p.fg_muted, font=self._fonts.mono_small, anchor="ne"
        )

        for index, stage in enumerate(self._stages):
            top = self.HEADER_H + index * self.ROW_H
            status = self._status[stage]
            colour = self._colours_for(status)
            selected = stage == self._selected
            hovered = stage == self._hover

            if selected:
                self.create_rectangle(
                    pad - 4, top + 2, width - pad + 4, top + self.ROW_H - 2,
                    fill=p.surface_high, outline="",
                )
            elif hovered:
                self.create_rectangle(
                    pad - 4, top + 2, width - pad + 4, top + self.ROW_H - 2,
                    fill=p.surface_alt, outline="",
                )

            # connector from the previous stage
            if index:
                connector = p.line
                if self._status[self._stages[index - 1]] == "done":
                    connector = p.mint if status != "pending" else p.line
                self.create_line(
                    pad + 7, top - 6, pad + 7, top + 4, fill=connector, width=1
                )

            glyph_x = pad + 7
            glyph_y = top + self.ROW_H // 2 - 3
            if status == "running":
                ring = p.surface_high
                self.create_rectangle(
                    glyph_x - 5, glyph_y - 5, glyph_x + 5, glyph_y + 5, fill=ring, outline=colour
                )
            elif status == "done":
                self.create_rectangle(
                    glyph_x - 5, glyph_y - 5, glyph_x + 5, glyph_y + 5, fill=colour, outline=""
                )
            else:
                self.create_rectangle(
                    glyph_x - 5, glyph_y - 5, glyph_x + 5, glyph_y + 5,
                    fill=p.surface_alt, outline=p.line,
                )
            self.create_text(
                glyph_x,
                glyph_y,
                text=STAGE_GLYPHS.get(stage, "\u25cf"),
                fill=p.bg if status == "done" else colour,
                font=self._fonts.mono_small,
            )

            title = truncate(self._titles.get(stage, stage), max(10, (width - 70) // 8))
            self.create_text(
                pad + 22,
                top + 12,
                text=title,
                fill=p.fg if status != "pending" or selected else p.fg_muted,
                font=self._fonts.body_bold if (selected or status == "running") else self._fonts.body,
                anchor="nw",
            )
            note = truncate(self._notes.get(stage, ""), max(12, (width - 70) // 6))
            if note:
                self.create_text(
                    pad + 22, top + 27, text=note, fill=p.fg_faint,
                    font=self._fonts.small, anchor="nw",
                )

            # level bar along the bottom of the row
            bar_y = top + self.ROW_H - 8
            bar_w = width - pad * 2 - 22
            self.create_rectangle(
                pad + 22, bar_y, pad + 22 + bar_w, bar_y + 3, fill=p.line_soft, outline=""
            )
            level = self._levels.get(stage, 0.0)
            if level > 0:
                self.create_rectangle(
                    pad + 22,
                    bar_y,
                    pad + 22 + max(2, bar_w * level),
                    bar_y + 3,
                    fill=colour,
                    outline="",
                )

        if running:
            self.create_text(
                pad, self.HEADER_H + self.ROW_H * len(self._stages) + 2,
                text="sedang berjalan: " + ", ".join(self._titles.get(s, s) for s in running),
                fill=p.accent, font=self._fonts.small, anchor="nw",
            )

    def _colours_for(self, status: str) -> str:
        p = self._palette
        token = STATUS_COLOURS.get(status, "fg_faint")
        return getattr(p, token, p.fg_muted)


__all__ = [
    "AutoWrapLabel",
    "Badge",
    "Card",
    "Chip",
    "Divider",
    "IconButton",
    "KeyValue",
    "LevelMeter",
    "PillButton",
    "ScrollFrame",
    "SectionLabel",
    "StageRail",
    "Tooltip",
]

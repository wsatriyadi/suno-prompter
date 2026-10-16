"""Visual language of Lirik Studio.

The subject is a mixing desk in a dark room: near-black neutral surfaces, faint
hairline rules, one warm amber accent for the master bus, and a rationed spectral
ramp where hue encodes a signal property (loudness, level, energy) rather than
decorating. Typography is the one risk: condensed display faces at small sizes
with generous letterspacing for labels, paired with a monospace for every number
the app reports, because this tool is full of measurements and numbers should
line up in columns.

Nothing here imports the rest of the app, so the palette and metrics can be
unit-tested without a display.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Palette:
    """Named colours. Dark first: this is a studio tool."""

    bg: str = "#0e1013"
    surface: str = "#161a20"
    surface_alt: str = "#1c2129"
    surface_high: str = "#242a34"
    line: str = "#2b323d"
    line_soft: str = "#20262e"
    fg: str = "#e8ecf2"
    fg_muted: str = "#98a3b3"
    fg_faint: str = "#5f6b7a"
    accent: str = "#f5a524"  # master bus amber
    accent_dim: str = "#8a5c14"
    cyan: str = "#39c2d7"  # timing
    cyan_dim: str = "#1e5f6b"
    mint: str = "#4ec9a4"  # pass / level good
    amber: str = "#e8b13c"  # caution
    red: str = "#e5646b"  # fail
    violet: str = "#a98bff"  # craft
    rose: str = "#ef8fb0"  # review
    spectrum: tuple[str, ...] = (
        "#2f6f7d",
        "#3d9a8f",
        "#5fbf7f",
        "#9ccb63",
        "#d8c34a",
        "#efa93c",
        "#e5723c",
        "#dc4f5b",
    )

    def stage_colour(self, accent: str) -> str:
        """Resolve a stage accent token (``c1``..``c6``) to a hex colour."""
        ramp = {
            "c1": self.accent,
            "c2": self.cyan,
            "c3": self.violet,
            "c4": self.rose,
            "c5": self.mint,
            "c6": self.amber,
        }
        return ramp.get(accent, self.accent)

    def level_colour(self, fraction: float) -> str:
        """Map 0..1 onto the spectral ramp (used for scores and meters)."""
        fraction = max(0.0, min(1.0, fraction))
        index = int(round(fraction * (len(self.spectrum) - 1)))
        return self.spectrum[index]


@dataclass(frozen=True)
class Typography:
    """Resolved font families with graceful fallbacks."""

    display: str = "Segoe UI"
    body: str = "Segoe UI"
    mono: str = "Consolas"
    display_condensed: str = "Segoe UI"
    available: tuple[str, ...] = ()


def resolve_typography(root: tk.Misc | None = None) -> Typography:
    """Pick the best installed faces once, at startup."""
    try:
        families = set(tkfont.families(root))
    except tk.TclError:  # pragma: no cover - headless / no font system
        return Typography()

    def first(*candidates: str, default: str) -> str:
        for candidate in candidates:
            if candidate in families:
                return candidate
        return default

    display = first(
        "Segoe UI Variable Display",
        "Segoe UI Semibold",
        "Bahnschrift SemiBold",
        "Segoe UI",
        "Helvetica Neue",
        default="Segoe UI",
    )
    condensed = first(
        "Bahnschrift SemiCondensed",
        "Bahnschrift",
        "Segoe UI Variable Small",
        display,
        default=display,
    )
    body = first("Segoe UI Variable Text", "Segoe UI", "Helvetica Neue", default="Segoe UI")
    mono = first(
        "Cascadia Mono",
        "Cascadia Code",
        "JetBrains Mono",
        "Consolas",
        "Menlo",
        "DejaVu Sans Mono",
        default="Courier",
    )
    return Typography(
        display=display,
        body=body,
        mono=mono,
        display_condensed=condensed,
        available=tuple(sorted(families)),
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Metrics:
    """Spacing and sizing scale. 4px grid, nothing arbitrary."""

    unit: int = 4
    radius: int = 6
    pad_xs: int = 4
    pad_sm: int = 8
    pad_md: int = 12
    pad_lg: int = 20
    pad_xl: int = 32
    header_h: int = 60
    rail_w: int = 268
    rail_collapsed_w: int = 56
    status_h: int = 32
    chat_min: int = 380
    output_min: int = 460


@dataclass
class Fonts:
    """Concrete ``Font`` objects, built once and reused everywhere."""

    display_l: tkfont.Font
    display_m: tkfont.Font
    display_s: tkfont.Font
    body: tkfont.Font
    body_bold: tkfont.Font
    small: tkfont.Font
    label: tkfont.Font
    mono: tkfont.Font
    mono_small: tkfont.Font
    mono_big: tkfont.Font
    chat: tkfont.Font
    raw: tuple[str, int] = ("TkFixedFont", 10)


def build_fonts(typo: Typography) -> Fonts:
    """Create the font set. Call once, after the root window exists."""
    return Fonts(
        display_l=tkfont.Font(family=typo.display, size=22, weight="bold"),
        display_m=tkfont.Font(family=typo.display, size=15, weight="bold"),
        display_s=tkfont.Font(family=typo.display_condensed, size=12, weight="bold"),
        body=tkfont.Font(family=typo.body, size=10),
        body_bold=tkfont.Font(family=typo.body, size=10, weight="bold"),
        small=tkfont.Font(family=typo.body, size=9),
        label=tkfont.Font(family=typo.display_condensed, size=8, weight="bold"),
        mono=tkfont.Font(family=typo.mono, size=10),
        mono_small=tkfont.Font(family=typo.mono, size=9),
        mono_big=tkfont.Font(family=typo.mono, size=13, weight="bold"),
        chat=tkfont.Font(family=typo.body, size=10),
        raw=(typo.mono, 10),
    )


# --------------------------------------------------------------------------- #
# stage identity
# --------------------------------------------------------------------------- #

#: Icon glyph per stage, drawn with the mono font so it always renders.
STAGE_GLYPHS: dict[str, str] = {
    "intake": "\u25cf",
    "audio_analyst": "\u223f",
    "composer": "\u266b",
    "lyricist": "\u270e",
    "lyric_reviewer": "\u2696",
    "arranger": "\u21c4",
    "music_reviewer": "\u2699",
    "prompt_engineer": "\u25b6",
    "prompt_reviewer": "\u2713",
}

STATUS_COLOURS = {
    "pending": "fg_faint",
    "running": "accent",
    "done": "mint",
    "failed": "red",
    "skipped": "fg_faint",
}


def truncate(text: str, limit: int, ellipsis: str = "\u2026") -> str:
    """Shorten text for a single line, respecting the ellipsis width."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip(" ,;.") + ellipsis


def wrap_words(text: str, width: int) -> list[str]:
    """Greedy word wrap; used by the canvas-based transcript renderer."""
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if length + addition > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += addition
    if current:
        lines.append(" ".join(current))
    return lines


def meter_blocks(fraction: float, slots: int = 10, ramp: Iterable[str] = ()) -> list[bool]:
    """Which meter slots are lit for ``fraction`` (0..1)."""
    fraction = max(0.0, min(1.0, fraction))
    lit = int(round(fraction * slots))
    return [i < lit for i in range(slots)]


class StatusCode(str, Enum):
    """Stage lifecycle, shared by the rail, the status line and the tests."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


def level_colour(palette: Palette, fraction: float) -> str:
    """Map 0..1 onto the palette's spectral ramp.

    Standalone counterpart to :meth:`Palette.level_colour` so tests and
    non-widget code can colour a value without constructing a widget.
    """
    return palette.level_colour(fraction)


def status_colour(palette: Palette, status: str | StatusCode) -> str:
    """Resolve a status token to a palette colour, tolerating unknowns."""
    token = status.value if isinstance(status, StatusCode) else str(status)
    attribute = STATUS_COLOURS.get(token, "fg_faint")
    return getattr(palette, attribute, palette.fg_muted)


__all__ = [
    "Palette",
    "Typography",
    "Metrics",
    "Fonts",
    "STAGE_GLYPHS",
    "STATUS_COLOURS",
    "StatusCode",
    "build_fonts",
    "level_colour",
    "meter_blocks",
    "resolve_typography",
    "status_colour",
    "truncate",
    "wrap_words",
]

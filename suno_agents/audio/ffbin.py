"""Locate the ``ffmpeg`` / ``ffprobe`` executables used by the audio engine.

Resolution order per binary:

1. the ``FFMPEG_BIN`` / ``FFPROBE_BIN`` environment override (absolute path or
   a bare executable name that is looked up on ``PATH``),
2. a copy bundled with a frozen build (``sys._MEIPASS``), which is what makes the
   packaged .exe able to analyse audio without asking the user to install ffmpeg,
3. a copy sitting next to the executable, so a user can drop a newer ffmpeg into
   the app folder without rebuilding,
4. :func:`shutil.which` on ``PATH``.

The successful lookup is cached process-wide; :func:`reset_cache` drops it
(tests that repoint the environment variables need that).
"""

from __future__ import annotations

import os
import shutil
import sys

from .errors import AudioEngineError

__all__ = [
    "FFMPEG_ENV",
    "FFPROBE_ENV",
    "bundled_search_dirs",
    "require_ffmpeg",
    "reset_cache",
]

#: Environment variables that override the ``PATH`` lookup.
FFMPEG_ENV = "FFMPEG_BIN"
FFPROBE_ENV = "FFPROBE_BIN"

INSTALL_HINT = (
    "Install ffmpeg (it ships ffprobe) and put both on PATH, or point "
    "FFMPEG_BIN / FFPROBE_BIN at the executables, or drop ffmpeg.exe and "
    "ffprobe.exe into a folder named 'ffmpeg' beside this application. "
    "Windows: winget install Gyan.FFmpeg - macOS: brew install ffmpeg - "
    "Debian/Ubuntu: sudo apt install ffmpeg"
)

_cached: tuple[str, str] | None = None


def bundled_search_dirs() -> list[str]:
    """Directories to check for a bundled ``ffmpeg``/``ffprobe``, in priority order.

    Empty in a normal source checkout, populated in a frozen build. Kept separate
    from the resolution below so a diagnostics command can report exactly where it
    looked instead of guessing.
    """
    dirs: list[str] = []
    bundle = getattr(sys, "_MEIPASS", "")
    if bundle:
        dirs.extend([os.path.join(bundle, "ffmpeg"), bundle])
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        dirs.extend([os.path.join(exe_dir, "ffmpeg"), exe_dir])
    seen: list[str] = []
    for path in dirs:
        if path and path not in seen and os.path.isdir(path):
            seen.append(path)
    return seen


def _resolve(env_var: str, executable: str) -> str | None:
    """Return the path for *executable*, honouring the override then the bundle."""
    override = os.environ.get(env_var, "").strip()
    if override:
        if os.path.isfile(override):
            return override
        found = shutil.which(override)
        if found:
            return found
        raise AudioEngineError(
            f"{env_var}={override!r} does not point at an executable. {INSTALL_HINT}"
        )
    for directory in bundled_search_dirs():
        for name in (executable, f"{executable}.exe"):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
    return shutil.which(executable)


def require_ffmpeg() -> tuple[str, str]:
    """Return ``(ffmpeg, ffprobe)`` paths, resolving and caching on first use.

    Raises :class:`AudioEngineError` with an install hint when either binary is
    missing from ``PATH``.
    """
    global _cached
    if _cached is None:
        ffmpeg = _resolve(FFMPEG_ENV, "ffmpeg")
        ffprobe = _resolve(FFPROBE_ENV, "ffprobe")
        missing = [
            name
            for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
            if not path
        ]
        if missing:
            raise AudioEngineError(
                f"{' and '.join(missing)} not found on PATH. {INSTALL_HINT}"
            )
        _cached = (ffmpeg, ffprobe)  # type: ignore[assignment]
    return _cached


def reset_cache() -> None:
    """Forget the cached lookup so the next call re-resolves the binaries."""
    global _cached
    _cached = None

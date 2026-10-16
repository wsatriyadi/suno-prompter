"""Container/stream facts for a reference file, straight from ``ffprobe``.

:func:`probe_file` never raises: a missing file, an unreadable stream or a
machine without ffmpeg all come back as an :class:`AudioProbe` whose ``error``
field carries a short message, so batch callers can keep going with whatever
they did understand.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from .errors import AudioEngineError
from .ffbin import require_ffmpeg
from .models import LOSSLESS_CODECS, AudioProbe

__all__ = ["probe_file"]

_MAX_ERROR_CHARS = 200


def _short_error(raw: object, *, tail: bool = True) -> str:
    """Collapse *raw* into one short line (stderr tail, or message head)."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "replace")
    else:
        text = str(raw or "")
    text = " ".join(text.split())
    return text[-_MAX_ERROR_CHARS:] if tail else text[:_MAX_ERROR_CHARS]


def _as_int(value: object) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _merge_tags(*sources: object) -> dict[str, str]:
    """Merge tag dicts, lower-casing keys; later sources win."""
    merged: dict[str, str] = {}
    for tags in sources:
        if not isinstance(tags, dict):
            continue
        for key, value in tags.items():
            if key is None:
                continue
            merged[str(key).lower()] = "" if value is None else str(value)
    return merged


def probe_file(path: str | Path) -> AudioProbe:
    """Return what ``ffprobe`` knows about *path*.

    Fills codec/container/sample-rate/channels/bit-rate/duration/bits-per-sample
    and merged tags (lower-cased keys, string values). ``lossless`` is true when
    the codec is one of :data:`suno_agents.audio.models.LOSSLESS_CODECS`.
    """
    target = Path(path)
    probe = AudioProbe(path=str(target))

    if not target.exists():
        probe.error = "file not found"
        return probe

    try:
        _, ffprobe = require_ffmpeg()
    except AudioEngineError as exc:
        probe.error = _short_error(exc, tail=False)
        return probe

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(target),
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL
        )
    except OSError as exc:
        probe.error = _short_error(f"could not run ffprobe: {exc}", tail=False)
        return probe

    if proc.returncode != 0:
        probe.error = _short_error(proc.stderr) or (
            f"ffprobe exited with code {proc.returncode}"
        )
        return probe

    try:
        info = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except (json.JSONDecodeError, AttributeError):
        probe.error = "ffprobe returned invalid JSON"
        return probe

    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    stream = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"),
        None,
    )
    if stream is None:
        probe.error = "no audio stream in file"
        return probe

    codec = str(stream.get("codec_name") or "unknown").lower()
    probe.codec = codec
    probe.container = str(fmt.get("format_name") or "unknown")
    probe.sample_rate = _as_int(stream.get("sample_rate")) or 0
    probe.channels = _as_int(stream.get("channels")) or 0
    probe.bit_rate = _as_int(stream.get("bit_rate")) or _as_int(fmt.get("bit_rate"))

    duration = _as_float(fmt.get("duration"))
    if duration is None:
        duration = _as_float(stream.get("duration"))
    probe.duration = float(duration or 0.0)

    bits = _as_int(stream.get("bits_per_raw_sample"))
    if bits is None:
        bits = _as_int(stream.get("bits_per_sample"))
    probe.bits_per_raw_sample = bits

    probe.tags = _merge_tags(fmt.get("tags"), stream.get("tags"))
    probe.lossless = codec in LOSSLESS_CODECS
    return probe

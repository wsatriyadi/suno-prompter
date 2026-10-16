"""Decode any ffmpeg-readable file into numpy arrays over an ffmpeg pipe.

Samples come back as float32 exactly as decoded (full scale = 1.0); no peak
normalisation is applied. Mono downmixes with ``-ac 1``, stereo downmixes (or
upmixes) with ``-ac 2``; a genuinely mono source is decoded once and duplicated
so :func:`decode_stereo` always returns two identical rows for it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .errors import AudioEngineError
from .ffbin import require_ffmpeg
from .models import AudioProbe
from .probe import probe_file

__all__ = ["decode_mono", "decode_stereo"]

_STDERR_TAIL = 300
_BYTES_PER_SAMPLE = 4


def _stderr_tail(raw: bytes | None) -> str:
    if not raw:
        return ""
    text = " ".join(raw.decode("utf-8", "replace").split())
    return text[-_STDERR_TAIL:]


def _decode_raw(
    target: Path,
    sr: int | None,
    max_seconds: float | None,
    channels: int,
    known_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Run ffmpeg once and return ``(samples, sample_rate)``.

    ``samples`` is flat float32 in the requested channel layout; the caller
    reshapes. The returned rate is ``sr`` when a resample was asked for, else the
    source's native rate.
    """
    if not target.exists():
        raise AudioEngineError(f"audio file not found: {target}")

    ffmpeg, _ = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(target),
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(int(channels)),
    ]
    if sr is not None:
        cmd += ["-ar", str(int(sr))]
    if max_seconds is not None:
        cmd += ["-t", f"{float(max_seconds):.6f}"]
    cmd.append("pipe:1")

    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL
    )
    if proc.returncode != 0:
        detail = _stderr_tail(proc.stderr) or f"ffmpeg exit code {proc.returncode}"
        raise AudioEngineError(f"ffmpeg could not decode {target.name}: {detail}")

    raw = proc.stdout or b""
    usable = len(raw) - (len(raw) % _BYTES_PER_SAMPLE)
    if usable <= 0:
        detail = _stderr_tail(proc.stderr)
        suffix = f": {detail}" if detail else ""
        raise AudioEngineError(f"ffmpeg produced no audio samples for {target.name}{suffix}")

    samples = np.frombuffer(raw[:usable], dtype="<f4").astype(np.float32, copy=True)

    if sr is not None:
        rate = int(sr)
    else:
        rate = int(known_rate or 0)
        if rate <= 0:
            meta: AudioProbe = probe_file(target)
            rate = int(meta.sample_rate) if meta.sample_rate > 0 else 0
        if rate <= 0:
            raise AudioEngineError(
                f"could not determine the native sample rate of {target.name}"
            )
    return samples, rate


def decode_mono(
    path: str | Path, sr: int | None = None, max_seconds: float | None = None
) -> tuple[np.ndarray, int]:
    """Decode *path* to mono float32 samples shaped ``(n,)``.

    Raw decoded samples scaled to ``[-1, 1]`` (no peak normalisation). Pass *sr*
    to resample, ``None`` to keep the file's native rate. *max_seconds* limits
    how much audio is decoded.
    """
    samples, rate = _decode_raw(Path(path), sr, max_seconds, 1)
    return samples, rate


def decode_stereo(
    path: str | Path, sr: int | None = None, max_seconds: float | None = None
) -> tuple[np.ndarray, int]:
    """Decode *path* to interleaved-free stereo float32 shaped ``(2, n)``.

    More than two source channels are downmixed by ffmpeg; a mono source yields
    two identical rows.
    """
    target = Path(path)
    meta = probe_file(target)
    if meta.channels == 1:
        samples, rate = _decode_raw(
            target, sr, max_seconds, 1, known_rate=meta.sample_rate
        )
        return np.repeat(samples[np.newaxis, :], 2, axis=0), rate

    samples, rate = _decode_raw(
        target, sr, max_seconds, 2, known_rate=meta.sample_rate
    )
    frames = samples.size // 2
    return samples[: frames * 2].reshape(2, frames), rate

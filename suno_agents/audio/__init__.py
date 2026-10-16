"""Reference-audio analysis for the Lirik Studio prompt pipeline.

Public surface: :func:`probe_file`, :func:`decode_mono`, :func:`decode_stereo`,
:func:`analyze` and :func:`build_profile`, with the frozen dataclasses from
:mod:`suno_agents.audio.models`. Every failure is raised as
:class:`AudioEngineError`; importing this package never touches ffmpeg.
"""

from __future__ import annotations

from .analysis import analyze
from .decode import decode_mono, decode_stereo
from .errors import AudioEngineError
from .models import AnalysisMetrics, AudioProbe, ReferenceProfile
from .probe import probe_file
from .profile import build_profile

__all__ = [
    "AudioEngineError",
    "AudioProbe",
    "AnalysisMetrics",
    "ReferenceProfile",
    "probe_file",
    "decode_mono",
    "decode_stereo",
    "analyze",
    "build_profile",
]

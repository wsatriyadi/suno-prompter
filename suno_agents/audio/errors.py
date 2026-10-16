"""Single error type for the audio engine.

Every failure raised by this subsystem (missing ffmpeg, bad file, undecodable
stream) surfaces as :class:`AudioEngineError` so callers never have to import
``subprocess`` to catch our problems.
"""

from __future__ import annotations

__all__ = ["AudioEngineError"]


class AudioEngineError(RuntimeError):
    """Raised when the audio engine cannot probe, decode or analyse a file."""

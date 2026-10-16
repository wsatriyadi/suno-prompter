"""Agent 2 - the audio reference analyst.

The measurements come from :mod:`suno_agents.audio` (ffmpeg + DSP), never from
the model: a language model cannot hear tempo or bitrate. This agent's job is to
turn numbers into musical decisions, and to tell the truth about whether the
reference is actually lossless.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import AUDIO_ANALYST
from ..domain.models import as_float, as_str_list
from .base import Agent
from .prompts import (
    CONSTITUTION,
    CRAFT_MIX,
    JSON_DISCIPLINE,
    SCHEMA_FIELD_NAMES,
    NEED_BRIEF,
    NEED_REFERENCE,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)
from .schemas import REFERENCE_SCHEMA

ANALYST_CRAFT = """\
YOUR CRAFT: READING A SPECTRUM LIKE A PRODUCER
- Tempo and key are given to you by measurement; you interpret them. A 128 kbps
  MP3 sounds glassy and loses cymbals; an upsample to 96 kHz WAV keeps the
  destroyed high end while looking lossless on paper - call that out as
  "lossy-transcoded" and never let the style field promise "airy cymbals" or
  "crisp hi-hats" on top of it.
- Interpreting numbers:
  * spectral centroid under ~1.5 kHz = dark, warm, analog; 1.5-3.5 kHz = balanced,
    modern; above 3.5 kHz = bright, thin, digital-forward.
  * lowpass cutoff 15-16 kHz -> 128 kbps class MP3; 18-19 kHz -> 192-256 kbps;
    above 20.5 kHz -> genuinely lossless.
  * stereo width near 0 = mono or near-mono source; above 0.6 = wide, watch mono
    compatibility.
  * dynamic range under 6 dB = loudness-war brick; crest factor under 8 dB = the
    reference already lost its transients. Do not ask the new song to copy that.
  * tempo feel: 60-76 ballad/heartbeat, 78-95 laid-back, 96-118 mid/driving,
    120-134 dance/pop, 135-170 energetic, 170+ double-time feel.
  * key mode decides emotional default: minor for longing/tension, major for
    resolution/celebration; name the relative-major relationship when useful.
- Transfer rules (this is the real value you add): separate what is worth copying
  (energy curve shape, density, instrument roles, register choices) from what must
  NOT be copied (melody, chord progression identity, lyrics, topline phrasing).
- Always produce at least 3 `arrangement_lessons` phrased as instructions the
  composer can act on, e.g. "chorus adds a 16th-note shaker layer and doubles the
  lead an octave up instead of getting louder".
- `do_not_copy` is a legal and creative safety line. Be specific.
- If the analysis is absent, say so plainly in `interpretation`, set confidence
  below 0.3, and produce generic-but-true guidance. Never fabricate measurements.
"""


class AudioAnalystAgent(Agent):
    stage = AUDIO_ANALYST
    schema = REFERENCE_SCHEMA

    def needs(self, state) -> Sequence[str]:
        return (NEED_BRIEF, NEED_REFERENCE)

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        return build_system(
            CONSTITUTION,
            ANALYST_CRAFT,
            CRAFT_MIX,
            JSON_DISCIPLINE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(REFERENCE_SCHEMA),
            context_block(state, self.needs(state)),
            revision_block(revision_notes, self.stage),
        )

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        if not state.reference_summary:
            return (
                "No audio reference was supplied. Report that honestly in "
                "`interpretation` with confidence below 0.3, and give generic "
                "production guidance that matches the brief."
            )
        return (
            "Interpret the measured analysis for the production team. Ground every "
            "claim in a number that appears in the analysis block."
        )

    def parse(self, payload: Any, state) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("Reference analysis must be a JSON object.")
        return {
            "interpretation": str(payload.get("interpretation", "") or "").strip(),
            "genre_read": as_str_list(payload.get("genre_read")),
            "production_notes": as_str_list(payload.get("production_notes")),
            "arrangement_lessons": as_str_list(payload.get("arrangement_lessons")),
            "transferable_to_new_song": as_str_list(payload.get("transferable_to_new_song")),
            "do_not_copy": as_str_list(payload.get("do_not_copy")),
            "lossless_reality_check": str(payload.get("lossless_reality_check", "") or "").strip(),
            "quality_warnings": as_str_list(payload.get("quality_warnings")),
            "confidence": max(0.0, min(1.0, as_float(payload.get("confidence"), 0.5))),
        }

    def validate(self, artifact: Any) -> list[str]:
        problems: list[str] = []
        if not artifact["interpretation"]:
            problems.append("Field 'interpretation' is empty.")
        if len(artifact["arrangement_lessons"]) < 3 and artifact["confidence"] > 0.3:
            problems.append(
                "Provide at least 3 arrangement_lessons that the composer can act on, "
                "or lower confidence below 0.3 if there is no usable audio."
            )
        return problems

    def summary_block(self, state, artifact: Mapping[str, Any]) -> str:
        """Text appended to ``state.reference_summary`` for downstream stages."""
        lines = [
            "",
            "--- PRODUCER INTERPRETATION ---",
            artifact["interpretation"],
        ]
        if artifact["genre_read"]:
            lines.append(f"Genre read: {', '.join(artifact['genre_read'])}")
        for lesson in artifact["arrangement_lessons"]:
            lines.append(f"Lesson: {lesson}")
        for note in artifact["production_notes"]:
            lines.append(f"Production: {note}")
        for item in artifact["transferable_to_new_song"]:
            lines.append(f"Transferable: {item}")
        for item in artifact["do_not_copy"]:
            lines.append(f"Do NOT copy: {item}")
        if artifact["lossless_reality_check"]:
            lines.append(f"Lossless reality: {artifact['lossless_reality_check']}")
        for warning in artifact["quality_warnings"]:
            lines.append(f"Warning: {warning}")
        return "\n".join(lines)


__all__ = ["AudioAnalystAgent"]

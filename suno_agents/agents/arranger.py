"""Agent 6 - the arranger that makes the lyrics actually singable.

The bridge between words and melody. Grid arithmetic (bar numbers, beats,
seconds, syllables per beat) is recomputed in Python because a misplaced line is
the single most common reason a Suno render sounds rushed or garbled.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import ARRANGER
from ..domain.models import Arrangement, as_float, as_int, as_str_list
from ..quality.text import repetition_ratio
from .base import Agent
from .lyricist import count_syllables
from .prompts import (
    CONSTITUTION,
    CRAFT_MELODY,
    CRAFT_PROSODY,
    JSON_DISCIPLINE,
    SCHEMA_FIELD_NAMES,
    NEED_BLUEPRINT,
    NEED_BRIEF,
    NEED_LYRIC_REVIEW,
    NEED_LYRICS,
    NEED_REFERENCE,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)
from .schemas import ARRANGEMENT_SCHEMA

ARRANGER_CRAFT = """\
YOUR CRAFT: FITTING WORDS TO TIME
- You receive a blueprint with a bar grid and a lyric sheet with syllable counts.
  Your job is the marriage: every sung line gets a start bar, a start beat, a
  duration, a melodic contour, and honest stress treatment.
- Budget honestly. A 4/4 bar at 100 BPM lasts 2.4 s; a comfortable sung line of
  8 syllables needs about one bar. Two bars per line invites long notes; half a
  bar per line forces sixteenth-note delivery and reads as rap.
- Mandatory checks per line, and report them in `stress_conflicts` when they fail:
  * syllables fit the space you assigned (report syllables and syl/beat).
  * the word's natural stress lands on a strong beat (1 and 3 in 4/4), not on the
    'and' of 2, unless the displacement is a deliberate syncopation you name.
  * the long-note syllable is an open vowel (a, e, o, ou, ay) - closed syllables
    on long notes sound swallowed.
  * no more than ~7 syllables without a breath opportunity in a mid-tempo song.
- Resolve what you can yourself and list each resolution in `resolved_by`
  ("line 3 shortened to 7 syllables by dropping 'sangat'"). Only leave an entry
  in `conflicts` when the fix belongs to another stage - e.g. the line carries
  two ideas and must be rewritten by the lyricist, or the section is too short
  and the composer must add bars. Say which, in the conflict text.
- `hook_placement`: name the bar where the hook lands and what changes underneath
  (drop the drums, double the top line, stop the band for one beat).
- Transition plan: what happens in the last bar of each section that makes the
  next one feel inevitable - a fill, a held note, a filter sweep, a riser, or a
  dead stop. Dead stops are the cheapest great transition.
- Instrumental breaks: name the bars where the voice drops out and what carries
  the melody instead.
"""


class ArrangerAgent(Agent):
    stage = ARRANGER
    schema = ARRANGEMENT_SCHEMA

    def needs(self, state) -> Sequence[str]:
        return (
            NEED_BRIEF,
            NEED_REFERENCE,
            NEED_BLUEPRINT,
            NEED_LYRICS,
            NEED_LYRIC_REVIEW,
        )

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        return build_system(
            CONSTITUTION,
            ARRANGER_CRAFT,
            CRAFT_PROSODY,
            CRAFT_MELODY,
            JSON_DISCIPLINE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(ARRANGEMENT_SCHEMA),
            self._grid_block(state),
            context_block(state, self.needs(state)),
            revision_block(revision_notes, self.stage),
        )

    def _grid_block(self, state) -> str:
        """Give the model the exact grid and syllable budgets it must fill."""
        blueprint = state.blueprint
        sheet = state.lyrics
        if blueprint is None:
            return "GRID: (no blueprint available)"
        tempo = blueprint.tempo_bpm or 100.0
        lines = [
            f"GRID: {blueprint.total_bars} bars total, {blueprint.time_signature} at "
            f"{tempo:.0f} BPM. One bar = {60.0 / tempo * 4:.2f} s.",
            "Section map (start bar -> end bar) and the lyric lines that must fit:",
        ]
        cursor = 1
        for section in blueprint.sections:
            end = cursor + section.bars - 1
            entry = f"  [{section.name}] bars {cursor}-{end} (energy {section.energy}/10)"
            if sheet is not None:
                match = next(
                    (s for s in sheet.sections if s.name.strip().lower() == section.name.strip().lower()),
                    None,
                )
                if match is not None and match.lines:
                    counts = [
                        count_syllables(l.text, sheet.language)
                        for l in match.lines
                        if l.text.strip()
                    ]
                    budget = int(section.bars * 4 * (60.0 / tempo) / (60.0 / tempo) * 0)
                    per_line = max(1, int(section.bars / max(1, len(counts)))) if counts else 0
                    entry += (
                        f" | {len(counts)} lines, syllables {counts}, "
                        f"average {sum(counts) / len(counts):.1f}, "
                        f"~{per_line} bars available per line"
                    )
                    del budget
            lines.append(entry)
            cursor = end + 1
        return "\n".join(lines)

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        return (
            "Map every lyric line onto the bar grid. For each line give start_bar, "
            "start_beat, bars, syllables, syllables_per_beat, melodic_motion, contour, "
            "emphasis, breath_mark and harmony_support. List any stress conflict you "
            "could not resolve in `conflicts`, and say which stage owns the fix.\n"
            "Reply with one JSON object only."
        )

    def parse(self, payload: Any, state) -> Arrangement:
        if not isinstance(payload, Mapping):
            raise ValueError("Arrangement must be a JSON object.")
        arrangement = Arrangement.from_dict(payload)
        sheet = state.lyrics
        if sheet is not None:
            for mapping in arrangement.mappings:
                if mapping.line.strip():
                    mapping.syllables = count_syllables(mapping.line, sheet.language)
        _recompute_grid(arrangement, state)
        return arrangement

    def validate(self, artifact: Arrangement) -> list[str]:
        problems = list(artifact.validate())
        for mapping in artifact.mappings:
            if mapping.syllables and mapping.syllables_per_beat > 4.5:
                problems.append(
                    f"Line '{mapping.line[:40]}...' needs "
                    f"{mapping.syllables_per_beat:.1f} syllables per beat - un-singable. "
                    "Give it more bars or mark the conflict."
                )
                break
        if not artifact.transition_plan.strip():
            problems.append("Field 'transition_plan' is empty.")
        return problems


def _recompute_grid(arrangement: Arrangement, state) -> None:
    """Recompute timing from the blueprint grid; the model never owns arithmetic."""
    blueprint = state.blueprint
    if blueprint is None:
        return
    tempo = blueprint.tempo_bpm or 100.0
    beats_per_bar = 4.0
    try:
        beats_per_bar = float(str(blueprint.time_signature).split("/")[0]) or 4.0
    except (ValueError, IndexError):
        beats_per_bar = 4.0
    seconds_per_bar = beats_per_bar * 60.0 / tempo

    arrangement.total_bars = arrangement.total_bars or blueprint.total_bars
    if arrangement.total_bars != blueprint.total_bars:
        arrangement.total_bars = blueprint.total_bars
    arrangement.total_seconds = round(blueprint.total_bars * seconds_per_bar, 1)

    # Section start bars, recomputed so every downstream number agrees.
    starts: dict[str, int] = {}
    cursor = 1
    for section in blueprint.sections:
        starts[section.name.strip().lower()] = cursor
        cursor += section.bars

    for mapping in arrangement.mappings:
        key = mapping.section.strip().lower()
        start = starts.get(key)
        if start is not None:
            offset = max(0, mapping.start_bar - 1)
            # Clamp into the section so a hallucinated bar number cannot escape.
            section_bars = next(
                (s.bars for s in blueprint.sections if s.name.strip().lower() == key),
                arrangement.total_bars,
            )
            mapping.start_bar = start + min(offset, max(0, section_bars - 1))
        else:
            mapping.start_bar = max(1, min(mapping.start_bar, arrangement.total_bars))
        mapping.start_beat = max(1.0, min(float(mapping.start_beat or 1.0), beats_per_bar))
        mapping.bars = as_float(mapping.bars, 1.0) or 1.0
        mapping.start_beat = round(mapping.start_beat, 2)
        if mapping.syllables:
            sung_beats = mapping.bars * beats_per_bar - (mapping.start_beat - 1.0)
            mapping.syllables_per_beat = round(
                mapping.syllables / max(0.5, sung_beats), 2
            )

    if arrangement.vocal_entry_bar and state.lyrics is not None:
        arrangement.vocal_entry_bar = max(
            1, min(arrangement.vocal_entry_bar, arrangement.total_bars)
        )
    arrangement.section_timeline = [
        {
            "section": section.name,
            "start_bar": starts.get(section.name.strip().lower(), 0),
            "bars": section.bars,
            "energy": section.energy,
            "seconds": round(section.bars * seconds_per_bar, 1),
            "chords": [c.symbol for c in section.chords],
            "dynamics": section.dynamics,
        }
        for section in blueprint.sections
    ]
    if not arrangement.total_seconds:
        arrangement.total_seconds = round(arrangement.total_bars * seconds_per_bar, 1)
    if arrangement.hook_bar and state.lyrics is not None:
        arrangement.hook_bar = max(1, min(arrangement.hook_bar, arrangement.total_bars))


__all__ = ["ArrangerAgent", "repetition_ratio"]

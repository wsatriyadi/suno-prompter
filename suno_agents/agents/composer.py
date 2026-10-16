"""Agent 3 - the composer: harmony, groove, motif, arc.

Timing arithmetic is done in Python, not by the model: bar counts, section start
beats, and song duration are recomputed from tempo and meter afterwards, because
LLMs routinely get that arithmetic wrong and a wrong grid poisons the arranger.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import COMPOSER
from ..domain.models import MusicBlueprint, as_float, as_int, as_str_list, normalise_key
from .base import Agent
from .prompts import (
    CONSTITUTION,
    CRAFT_HARMONY,
    CRAFT_MELODY,
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
from .schemas import BLUEPRINT_SCHEMA

COMPOSER_CRAFT = """\
YOUR CRAFT: WRITING MUSIC THAT HOLDS TOGETHER
- You write for a vocalist and a hook, not for a chord chart. Every harmonic move
  must make the topline easier to sing or more interesting.
- Form first. Pick a form that fits the brief (e.g. Intro-Verse-PreChorus-Chorus-
  Verse-PreChorus-Chorus-Bridge-FinalChorus-Outro, or a club build with a drop),
  assign bars, then fill harmony. Typical pop/rock: verse 8, pre 4, chorus 8,
  bridge 4-8, intro 4-8, outro 4-8, total 150-230 seconds. A 90-second song and a
  7-minute song are both allowed - but never 3 bars of chorus.
- Same chords in every chorus is fine and usually better. Change the *energy*,
  the arrangement, and the last line instead.
- Groove: describe the actual pattern. "Kick on 1 and 3, snare on 2 and 4,
  closed hats on 8ths with an open hat on the 'and' of 4 at the end of every
  second bar; bass plays root on 1, fifth on the 'and' of 3." That is a groove.
- Motif: give it as scale degrees with rhythm, e.g. "5-6-5-3 (dotted 8th, 16th,
  8th, quarter) rising in the verse, inverted in the bridge".
- `hook_definition` names the exact moment the listener remembers: which bar,
  which words, which interval, and what the band does under it.
- `tension_curve` must be a sequence across the whole song, not a mood word.
- Serve the brief's must_include / must_avoid literally. If the user forbade
  drums, no drums anywhere - not even a "light kick" in the outro.
- If a measured reference exists, use its tempo feel, density, and energy-curve
  shape, and refuse to copy its melody or progressions (see the do-not-copy list).

LENGTH DISCIPLINE - the artifact is parsed as JSON and cut off past the output cap
- Cap each chord list at the chords that actually change: if two bars hold the
  same harmony, list it once with `beats: 8` instead of repeating the entry.
- Prose fields are working notes for the arranger, not liner notes. Budget:
  `groove` 2-3 sentences; `harmonic_language`, `chord_progression_summary`,
  `melodic_motif`, `motif_development`, `hook_definition`, `arrangement_notes` 2
  sentences each; `tension_curve` one clause per section; `theory_rationale`
  2 sentences; `delivery_notes` 1 sentence.
- `melodic_device`, `texture`, `drums`, `bass` and `purpose` are ONE short clause
  each, not paragraphs. The arranger needs the instruction, not the essay.
- Total reply target: under 12000 characters. A precise short blueprint that
  finishes parsing beats an exhaustive one that gets truncated.
"""


class ComposerAgent(Agent):
    stage = COMPOSER
    schema = BLUEPRINT_SCHEMA

    def needs(self, state) -> Sequence[str]:
        return (NEED_BRIEF, NEED_REFERENCE)

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        return build_system(
            CONSTITUTION,
            COMPOSER_CRAFT,
            CRAFT_HARMONY,
            CRAFT_MELODY,
            CRAFT_MIX,
            JSON_DISCIPLINE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(BLUEPRINT_SCHEMA),
            context_block(state, self.needs(state)),
            revision_block(revision_notes, self.stage),
        )

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        brief = state.brief
        tempo = f"{brief.tempo_bpm:.0f}" if brief and brief.tempo_bpm else "your choice"
        return (
            "Design the complete musical blueprint: key, mode, tempo, groove, "
            "section-by-section harmony with roman-numeral functions, melodic motif "
            "and its development, hook definition, and the tension curve.\n"
            f"Tempo to respect: {tempo} BPM. Every section needs a chord list with "
            "`function` filled in.\n"
            "Reply with one JSON object only."
        )

    def parse(self, payload: Any, state) -> MusicBlueprint:
        if not isinstance(payload, Mapping):
            raise ValueError("Blueprint must be a JSON object.")
        blueprint = MusicBlueprint.from_dict(payload)
        if state.brief is not None:
            blueprint = _align_with_brief(blueprint, state.brief)
        _recompute_timing(blueprint)
        return blueprint

    def validate(self, artifact: MusicBlueprint) -> list[str]:
        problems = list(artifact.validate())
        if not artifact.sections:
            return problems
        if any(not section.chords for section in artifact.sections):
            empty = [s.name for s in artifact.sections if not s.chords]
            problems.append(f"Sections missing harmony: {', '.join(empty)}.")
        for section in artifact.sections:
            if any(not chord.function for chord in section.chords):
                problems.append(
                    f"Section '{section.name}': every chord needs a roman-numeral "
                    "`function` (e.g. i7, V, bVII)."
                )
                break
        if artifact.estimated_seconds and not 60 <= artifact.estimated_seconds <= 480:
            problems.append(
                f"Estimated duration {artifact.estimated_seconds:.0f}s is outside 60-480s; "
                "rebalance the bars."
            )
        return problems


def _align_with_brief(blueprint: MusicBlueprint, brief) -> MusicBlueprint:
    """Respect hard user constraints; the model does not get to override them."""
    if brief.tempo_bpm and abs(blueprint.tempo_bpm - brief.tempo_bpm) > 1.0:
        blueprint.tempo_bpm = brief.tempo_bpm
    if brief.time_signature:
        blueprint.time_signature = brief.time_signature
    if brief.key:
        root, mode = normalise_key(brief.key)
        blueprint.key = root
        blueprint.mode = mode
    if brief.instrumentation:
        blueprint.instrumentation = list(brief.instrumentation)
    if brief.must_avoid:
        forbidden = ", ".join(brief.must_avoid).lower()
        notes = [n for n in blueprint.arrangement_notes.split(". ") if n]
        filtered = [n for n in notes if not any(word in n.lower() and word in forbidden for word in brief.must_avoid)]
        if filtered:
            blueprint.arrangement_notes = ". ".join(filtered)
    return blueprint


def _recompute_timing(blueprint: MusicBlueprint) -> MusicBlueprint:
    """Derive bar totals and duration from the model's bar counts and tempo."""
    try:
        beats_per_bar = float(str(blueprint.time_signature).split("/")[0])
    except (ValueError, IndexError):
        beats_per_bar = 4.0
    if beats_per_bar <= 0:
        beats_per_bar = 4.0

    cursor = 0.0
    for section in blueprint.sections:
        section.bars = max(1, as_int(section.bars, 4))
        section.start_beat = cursor
        cursor += section.bars * beats_per_bar
        section.energy = max(0, min(10, as_int(section.energy, 5)))
        if not section.chords:
            continue
        # If the model gave bar offsets that do not fit the section, rescale them.
        for chord in section.chords:
            chord.bar = max(1, as_int(chord.bar, 1))
            if chord.bar > section.bars:
                chord.bar = section.bars

    blueprint.total_bars = sum(section.bars for section in blueprint.sections)
    tempo = as_float(blueprint.tempo_bpm, 0.0)
    if tempo <= 0:
        tempo = 100.0
        blueprint.tempo_bpm = tempo
    blueprint.estimated_seconds = round(blueprint.total_bars * beats_per_bar * 60.0 / tempo, 1)
    if not blueprint.scale_notes:
        blueprint.scale_notes = _scale_for(blueprint.key, blueprint.mode)
    return blueprint


_MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
_MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)
_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _scale_for(root: str, mode: str) -> list[str]:
    try:
        index = _NOTES.index(root)
    except ValueError:
        index = 0
    steps = _MINOR_STEPS if mode == "minor" else _MAJOR_STEPS
    return [_NOTES[(index + step) % 12] for step in steps]


__all__ = ["ComposerAgent"]

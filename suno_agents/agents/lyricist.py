"""Agent 4 - lyricist and vocal architect.

Lyrics plus who sings them. Syllable counts are recomputed in Python from the
lyric text (the model's counts are unreliable), because the arranger balances
syllables against melodic space and a wrong count there produces lines that
cannot be sung.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..domain.graph import LYRICIST
from ..domain.models import LyricSheet, VocalSpec, as_bool, as_int, as_str_list
from .base import Agent
from .prompts import (
    CONSTITUTION,
    CRAFT_LYRICS,
    CRAFT_PROSODY,
    CRAFT_VOCAL,
    JSON_DISCIPLINE,
    SCHEMA_FIELD_NAMES,
    NEED_BLUEPRINT,
    NEED_BRIEF,
    NEED_REFERENCE,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)
from .schemas import LYRIC_SHEET_SCHEMA

LYRICIST_CRAFT = """\
YOUR CRAFT: WORDS THAT CAN BE SUNG
- Write ONLY for the sections the blueprint marks `lyrics_needed`. An instrumental
  break gets no words, ever.
- Match the brief's language exactly. Indonesian lyrics must sound like a person
  wrote them: contractions and everyday words, not textbook sentences. Never mix
  languages inside one line unless the brief asks for it.
- Concrete over abstract, always. Instead of "aku merindukanmu di dalam gelap",
  write the object: "kopi dingin di meja, kursi yang kamu tinggali".
- One idea per line, 6-14 syllables, and keep all lines inside a section within
  +/-2 syllables of each other. Long lines only in instrumental-driven or
  spoken-sung sections, and say so in the section cue.
- Rhyme: prefer ABAB or AABB on the strong end-words, allow slant rhyme. Never
  force a word only to rhyme - if it does not fit, break the rhyme.
- The chorus is the thesis. One line a stranger could quote after one listen.
  The first line of the chorus is usually the title.
- Show evolution: verse 2 escalates or contradicts verse 1, the bridge changes
  the camera, the final chorus lands with a changed line.
- `cue` marks delivery per line: "(whispered)", "(belted)", "(half-spoken)",
  "(held)". Use at most 2 cues per section so it stays a performance, not a
  storyboard. Ad-libs go in `adlib`, max 3 words, no more than one every 2 lines.
- VOCAL SPEC is half your job and often the deciding factor for the render:
  * gender must be explicit: male / female / mixed / instrumental.
  * `voice_type`: describable and castable, e.g. "warm male baritone with a
    raspy edge, roughly G2-F4" or "bright female mezzo, light vibrato, sits
    comfortably in the 200-500 Hz body".
  * `register`: chest / mixed / falsetto / head, plus where the song pushes the
    voice (e.g. "chest in verses, mixed belt on the chorus top note").
  * `timbre`, `delivery`, `vibrato`, `accent`, `age_impression`: one specific word
    each, not a sentence.
  * `harmony_layers` and `doubles`: real arrangement info ("3rds and 6ths under
    the final chorus, octave-down double on the hook line").
- INSTRUMENTAL briefs: set `instrumental` true, leave `sections[].lines` empty,
  and fill `instrumental_direction` with what replaces the voice - e.g. "lead
  melodica carrying the topline, wordless ooh pads in the chorus stack, no
  human syllables at all". Set `vocal.present` false and gender "instrumental".
- Titles: 2-5 words, quotable, 3 options minimum, drawn from the hook.
"""


class LyricistAgent(Agent):
    stage = LYRICIST
    schema = LYRIC_SHEET_SCHEMA

    def needs(self, state) -> Sequence[str]:
        return (NEED_BRIEF, NEED_REFERENCE, NEED_BLUEPRINT)

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        return build_system(
            CONSTITUTION,
            LYRICIST_CRAFT,
            CRAFT_LYRICS,
            CRAFT_PROSODY,
            CRAFT_VOCAL,
            JSON_DISCIPLINE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(LYRIC_SHEET_SCHEMA),
            context_block(state, self.needs(state)),
            revision_block(revision_notes, self.stage),
        )

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        sections = []
        if state.blueprint is not None:
            sections = [s.name for s in state.blueprint.sections]
        instrumental = bool(state.brief and state.brief.instrumental)
        mode = (
            "The brief is INSTRUMENTAL: write no lyric lines, only the instrumental "
            "direction and the vocal texture plan."
            if instrumental
            else "Write the full lyric set for the sections that need words."
        )
        return (
            f"{mode}\nSections from the blueprint: {', '.join(sections) or 'use standard form'}.\n"
            "Return the lyric sheet with per-line syllable counts, section cues, and a "
            "complete vocal specification (gender, voice type, register, timbre, "
            "delivery, harmony layers).\n"
            "Reply with one JSON object only."
        )

    def parse(self, payload: Any, state) -> LyricSheet:
        if not isinstance(payload, Mapping):
            raise ValueError("Lyric sheet must be a JSON object.")
        sheet = LyricSheet.from_dict(payload)
        if state.brief is not None:
            sheet = _align_with_brief(sheet, state.brief)
        # The model's syllable counts are not trustworthy; recompute them.
        for section in sheet.sections:
            for line in section.lines:
                line.text = " ".join(str(line.text).split())
                if line.text:
                    line.syllables = count_syllables(line.text, sheet.language)
        if not sheet.working_title and sheet.title_options:
            sheet.working_title = sheet.title_options[0]
        if not sheet.language:
            sheet.language = (state.brief.language if state.brief else "") or "id"
        if not sheet.vocal.gender and not sheet.instrumental:
            sheet.vocal.gender = (
                state.brief.vocal_gender if state.brief else "unspecified"
            )
        return sheet

    def validate(self, artifact: LyricSheet) -> list[str]:
        problems = list(artifact.validate())
        if artifact.instrumental:
            return problems
        # Line length sanity: a section whose lines swing wildly cannot be sung.
        for section in artifact.sections:
            counts = [l.syllables for l in section.lines if l.text.strip()]
            if len(counts) >= 3 and max(counts) - min(counts) > 6:
                problems.append(
                    f"Section '{section.name}' line lengths swing from {min(counts)} to "
                    f"{max(counts)} syllables; keep them within about 2-4 syllables."
                )
        empty_sections = [s.name for s in artifact.sections if not s.lines and s.name.lower()
                          .startswith(("verse", "chorus", "bridge", "pre"))]
        if empty_sections:
            problems.append(f"These sections need lyrics: {', '.join(empty_sections)}.")
        return problems

    def syllable_targets(self, state) -> dict[str, int]:
        """Syllable budget per section, derived from bars and tempo."""
        budget: dict[str, int] = {}
        if state.blueprint is None:
            return budget
        tempo = state.blueprint.tempo_bpm or 100.0
        for section in state.blueprint.sections:
            if not section.lyrics_needed:
                continue
            try:
                beats_per_bar = float(str(section.start_beat).split("/")[0]) if False else 4.0
            except ValueError:  # pragma: no cover - defensive
                beats_per_bar = 4.0
            seconds = section.bars * beats_per_bar * 60.0 / tempo
            # ~1.8 sung syllables per second at mid tempo is a comfortable ceiling.
            budget[section.name] = max(4, int(seconds * 1.8))
        return budget


def _align_with_brief(sheet: LyricSheet, brief) -> LyricSheet:
    if brief.instrumental:
        sheet.instrumental = True
        sheet.vocal.present = False
        sheet.vocal.gender = "instrumental"
    elif brief.language:
        sheet.language = brief.language
    if brief.vocal_gender in {"male", "female", "duet"} and not sheet.instrumental:
        declared = sheet.vocal.gender
        if declared in {"", "unspecified"}:
            sheet.vocal.gender = brief.vocal_gender
        elif brief.vocal_gender == "duet" and declared != "mixed":
            # A duet request is authoritative: honour it and flag the change.
            sheet.notes = (
                f"{sheet.notes}\nDuet was requested; treated the vocal cast as mixed "
                "(male + female) to satisfy the brief."
            ).strip()
            sheet.vocal.gender = "mixed"
    if brief.mood and not sheet.emotional_arc:
        sheet.emotional_arc = ", ".join(brief.mood)
    return sheet


_VOWEL_GROUP = re.compile(r"[aiueoáéíóúàèìòùâêîôûäëïöü]+", re.IGNORECASE)


def count_syllables(text: str, language: str = "") -> int:
    """Count singable syllables for Indonesian/English-style lyrics.

    Vowel-group counting with silent-e and common-diphthong handling: accurate
    enough for prosody budgets, and deterministic, which is what the arranger
    needs. Never returns 0 for a non-empty line.
    """
    words = re.findall(r"[A-Za-z\u00c0-\u024f']+", text)
    if not words:
        return 0
    total = 0
    for word in words:
        lowered = word.lower()
        groups = _VOWEL_GROUP.findall(lowered)
        count = len(groups)
        if lowered.endswith("e") and not lowered.endswith(("le", "ee", "ye")) and count > 1:
            count -= 1  # silent e in English-style words
        if lowered.endswith("es") and count > 2:
            count -= 1
        total += max(1, count)
    return max(1, total)


__all__ = ["LyricistAgent", "count_syllables"]

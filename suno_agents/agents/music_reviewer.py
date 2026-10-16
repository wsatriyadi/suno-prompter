"""Agent 7 - the music reviewer.

Audits the whole song as one object: structure, harmony, melody, character, and
whether the production is actually achievable. This is the reviewer most likely
to send work back to the composer, and the one that decides whether a song has a
character or is merely competent.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import MUSIC_REVIEWER
from ..quality.gates import PASS_MARKS
from ..domain.models import ReviewVerdict, as_float, as_str_list
from .base import Agent
from .prompts import (
    CONSTITUTION,
    CRAFT_HARMONY,
    CRAFT_MELODY,
    CRAFT_MIX,
    CRAFT_PROSODY,
    JSON_DISCIPLINE,
    CONVERGENCE_RULE,
    SCHEMA_FIELD_NAMES,
    NEED_ARRANGEMENT,
    NEED_BLUEPRINT,
    NEED_BRIEF,
    NEED_LYRICS,
    NEED_MUSIC_REVIEW,
    NEED_REFERENCE,
    REVIEW_SCHEMA,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)

MUSIC_REVIEWER_CRAFT = """\
YOUR CRAFT: HEARING THE SONG BEFORE IT EXISTS
- Reconstruct the song in your head from the blueprint plus the arrangement map,
  then judge what you hear. If you cannot imagine the chorus, that IS the finding.
- Score each dimension 0-10 and justify anything below 7 with the exact bar,
  chord, or line:
  * structure - does the form earn its length? Is the last chorus different?
  * harmony - functional logic, no aimless wandering, cadences placed on purpose?
  * melody_contour - arc over the song, singable leaps, range under control
    (a hook that demands two octaves will break the vocal)?
  * groove_and_rhythm - is the groove specific and consistent, does the meter of
    the words sit on it?
  * energy_arc - real contrast and a reason to keep listening after 60 seconds?
  * hook_strength - is the payoff worth the build?
  * lyrics_integration - do the words land on the notes (syllables, stress,
    long-note vowels, breath), reported by the arranger?
  * character_and_identity - could this be mistaken for a stock loop, or does it
    have a fingerprint (an unusual instrument, an odd bar, a specific vocal
    gesture)?
  * producibility - can a text-to-music model actually render this without the
    result collapsing: not too many simultaneous layers, no impossible
    orchestration, no contradictory instructions?
  * lossless_readiness - will this arrangement survive a clean master: headroom
    for dynamics, no muddy low-mid pile-up, mono-compatible low end, no clipping
    transient demand?
- The classic failures you must hunt for:
  * every chorus identical, so the song peaks at 60 seconds and then repeats.
  * a bridge that is just chorus 3 with different words.
  * melody and harmony fighting: chord tones not landing on strong beats.
  * too many layers at once so the render turns to mush on the chorus.
  * energy that starts at 10 and has nowhere to go.
  * a 3-minute song with only 40 seconds of actual musical ideas.
- `restart_from` must be the ROOT CAUSE stage and only from your legal list:
  * alignment, syllable/stress problems, breath plan -> `arranger`
  * words that do not serve the music, weak topline words, vowel problems ->
    `lyricist`
  * form, harmony, groove, motif, tempo, range -> `composer`
  * the reference was misread -> `audio_analyst`
  * the brief is contradictory or unsatisfiable as written -> `intake`
- You are the last gate before the prompt: if the song is mediocre, say so now,
  because after this stage nobody will hear it again. But "mediocre" must be
  namable: a specific structural, harmonic, melodic or production defect. "I can
  imagine a better song" is not a defect, it is the human condition.

THE PASS DECISION
- `pass` requires: no hard check failed, no scored dimension below 6/10, and your
  `overall_score` at or above the pass mark in your prompt.
- `revise` requires a named defect a composer, lyricist or arranger must fix, with
  the root cause stage. Improvements you can merely describe belong in `issues` at
  "minor" or "polish" severity and do NOT stop the pipeline.
- `reject` is for a wrong premise, not weak execution.
- Your score and your verdict must agree. Logging 8s and 9s across ten dimensions
  and then returning "revise" is self-contradictory: it tells the loop that nothing
  is wrong while refusing to release, which burns the budget without improving the
  song. Be honest at either end, not harsh in one place and generous in the other.
"""

DIMENSIONS = (
    "structure",
    "harmony",
    "melody_contour",
    "groove_and_rhythm",
    "energy_arc",
    "hook_strength",
    "lyrics_integration",
    "character_and_identity",
    "producibility",
    "lossless_readiness",
)


class MusicReviewerAgent(Agent):
    stage = MUSIC_REVIEWER
    # Reviewers answer once: a reprompt costs a full generation and rarely
    # changes the shape of a review.
    parse_retries = 0
    schema = REVIEW_SCHEMA

    def needs(self, state) -> Sequence[str]:
        return (
            NEED_BRIEF,
            NEED_REFERENCE,
            NEED_BLUEPRINT,
            NEED_LYRICS,
            NEED_ARRANGEMENT,
            NEED_MUSIC_REVIEW,
        )

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        return build_system(
            CONSTITUTION,
            MUSIC_REVIEWER_CRAFT,
            CRAFT_HARMONY,
            CRAFT_MELODY,
            CRAFT_PROSODY,
            CRAFT_MIX,
            JSON_DISCIPLINE,
            CONVERGENCE_RULE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(REVIEW_SCHEMA),
            "DIMENSIONS TO SCORE (exactly these keys): " + ", ".join(DIMENSIONS),
            self._targets_block(),
            context_block(state, self.needs(state)),
            revision_block(revision_notes, self.stage),
        )

    def _targets_block(self) -> str:
        from ..domain.graph import STAGE_INFO, legal_restarts

        names = ", ".join(
            f"'{stage}' ({STAGE_INFO[stage].title})" for stage in legal_restarts(self.stage)
        )
        return f"LEGAL restart_from values for you: {names}. Pick exactly one."

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        return (
            f"Pass mark for this stage: {PASS_MARKS[MUSIC_REVIEWER]:.0f}/100 with every "
            "hard check passing and no dimension below 6/10.\n"
            "Audit the complete song: blueprint, lyrics, and the lyric-to-note "
            "arrangement map. Judge it as one piece of music that a stranger will "
            "hear once.\n"
            f"Deterministic pre-checks already run:\n{self._mechanics(state)}\n"
            "Return the structured review as one JSON object."
        )

    def _mechanics(self, state) -> str:
        lines: list[str] = []
        blueprint = state.blueprint
        arrangement = state.arrangement
        if blueprint is not None:
            energies = [s.energy for s in blueprint.sections]
            lines.append(
                f"- form: {len(blueprint.sections)} sections, {blueprint.total_bars} bars, "
                f"{blueprint.estimated_seconds:.0f}s at {blueprint.tempo_bpm:.0f} BPM"
            )
            if energies:
                lines.append(
                    f"- energy range: {min(energies)}-{max(energies)}/10 across sections"
                )
            chords = {c.symbol for s in blueprint.sections for c in s.chords}
            lines.append(f"- distinct chord symbols: {len(chords)} ({', '.join(sorted(chords)[:10])})")
            chorus_count = sum(
                1 for s in blueprint.sections if "chorus" in s.name.lower() or "hook" in s.name.lower()
            )
            lines.append(f"- chorus statements: {chorus_count}")
            duplicate_energies = (
                len(energies) > 2 and len(set(energies)) <= 2
            )
            if duplicate_energies:
                lines.append("- WARNING: the energy value barely varies across sections.")
        if arrangement is not None:
            stressed = [m.line for m in arrangement.mappings if m.stress_conflicts]
            lines.append(
                f"- mapped lines: {len(arrangement.mappings)}, "
                f"unresolved stress conflicts: {len(stressed)}"
            )
            dense = [m for m in arrangement.mappings if m.syllables_per_beat > 3.6]
            if dense:
                lines.append(
                    f"- {len(dense)} lines run above 3.6 syllables/beat (rushed delivery risk)"
                )
            lines.append(f"- hook bar: {arrangement.hook_bar or 'not stated'}")
        if not lines:
            return "- (no musical material available - that alone is a blocker)"
        return "\n".join(lines)

    def parse(self, payload: Any, state) -> ReviewVerdict:
        if not isinstance(payload, Mapping):
            raise ValueError("Review must be a JSON object.")
        verdict = ReviewVerdict.from_dict(payload, stage=self.stage)
        # Only normalise dimensions the model actually reported. Inventing a 0.0 for
        # every missing key used to fabricate a "below the floor" issue for each
        # dimension at once - eight phantom defects that the next agent was then told
        # to fix, while the reviewer's own score said the work was fine. A reviewer
        # that omits its score sheet must be read as "scored via overall_score", not
        # as "scored everything zero".
        verdict.dimensions = {
            name: min(10.0, max(0.0, float(value)))
            for name, value in verdict.dimensions.items()
            if isinstance(value, (int, float)) and value == value
        }
        if not verdict.reasoning:
            verdict.reasoning = "No reasoning supplied."
        verdict.strengths = as_str_list(payload.get("strengths")) or verdict.strengths
        return verdict

    def validate(self, artifact: ReviewVerdict) -> list[str]:
        problems: list[str] = []
        if not artifact.dimensions or all(v == 0 for v in artifact.dimensions.values()):
            problems.append("Score all ten dimensions 0-10: " + ", ".join(DIMENSIONS))
        if artifact.verdict != "pass" and not artifact.issues:
            problems.append("A non-pass verdict requires at least one concrete issue.")
        if artifact.verdict == "pass" and artifact.restart_from:
            problems.append("A passing verdict must not request a restart.")
        return problems


__all__ = ["MusicReviewerAgent", "DIMENSIONS"]

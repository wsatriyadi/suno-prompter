"""Agent 5 - the lyric reviewer.

Scores like a lyric editor at a publishing house: is this worth someone's
attention, can it be sung, does it say something true. It is told, explicitly,
which stages it may send work back to, and that its taste cannot override the
deterministic gates reported to it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import AUDIO_ANALYST, INTAKE, LYRICIST, LYRIC_REVIEWER
from ..domain.models import ReviewVerdict, as_float, as_str_list
from ..quality import text as T
from ..quality.gates import PASS_MARKS
from .base import Agent
from .prompts import (
    CONSTITUTION,
    CRAFT_LYRICS,
    CRAFT_PROSODY,
    CRAFT_VOCAL,
    JSON_DISCIPLINE,
    CONVERGENCE_RULE,
    SCHEMA_FIELD_NAMES,
    NEED_BLUEPRINT,
    NEED_BRIEF,
    NEED_LYRIC_REVIEW,
    NEED_LYRICS,
    NEED_REFERENCE,
    REVIEW_SCHEMA,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)

REVIEWER_CRAFT = """\
YOUR CRAFT: EDITING, NOT APPLAUSE
- You are the person who decides. Two failures are equally yours: waving through
  work you would not put on a record, and refusing to release work that meets its
  brief because you can still imagine a better version. Judge against the pass mark
  and the hard checks, not against an ideal no draft will reach.
- Score each dimension 0-10 honestly, and justify every score below 7 with the
  exact line that caused it (quote it in `evidence`). Do not deflate scores to make
  a point: the number and the verdict must agree.
- Dimension meanings (use exactly these keys):
  * originality - does it avoid the phrases a thousand other songs already use?
  * imagery - concrete objects and scenes versus abstract feeling-words?
  * singability - syllable counts, open vowels on long notes, stress on strong
    beats, breath room. Count the syllables yourself before judging.
  * emotional_truth - does the emotion feel earned and specific, or announced?
  * hook_strength - could a stranger repeat the chorus after one listen?
  * language_quality - grammar, natural word order, no filler, no mixed language.
  * vocal_fit - do gender, voice type, register and delivery actually suit this
    song and this brief?
  * brief_fidelity - did it deliver what the user asked for, including
    must_include / must_avoid?
- `overall_score` is 0-100 and must be consistent with the dimensions.
- Follow the convergence rule below to the letter; `overall_score` and `verdict`
  must agree with each other and with the pass mark in your prompt.
- `restart_from` is the stage that owns the ROOT CAUSE, and you may only choose
  from the legal targets listed for you in the prompt. Diagnose properly:
  * mangled words, weak images, cliche lines, wrong voice -> `lyricist`
  * a song that cannot be sung because the form or melody gives no room, or the
    brief's story is structurally impossible -> `composer`
  * the user's own brief contradicts itself or is too vague to satisfy ->
    `intake`
  * the reference analysis was misread (tempo/key/quality) -> `audio_analyst`
- Do not send work back for taste you could have fixed yourself in the same pass;
  reserve rollbacks for defects with a real root cause.
- Do not rewrite the lyrics for the lyricist. Point at the defect and the
  direction. Three example rewrites as illustrations are welcome in `reasoning`,
  but the artifact must be produced by the lyricist.
"""

DIMENSIONS = (
    "originality",
    "imagery",
    "singability",
    "emotional_truth",
    "hook_strength",
    "language_quality",
    "vocal_fit",
    "brief_fidelity",
)


class LyricReviewerAgent(Agent):
    stage = LYRIC_REVIEWER
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
            NEED_LYRIC_REVIEW,
        )

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        return build_system(
            CONSTITUTION,
            REVIEWER_CRAFT,
            CRAFT_LYRICS,
            CRAFT_PROSODY,
            CRAFT_VOCAL,
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
        return (
            f"LEGAL restart_from values for you: {names}. Pick exactly one, and "
            "justify it with the root cause you found."
        )

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        mechanics = self._mechanics(state)
        return (
            f"Review the lyric sheet for the brief and the blueprint.\n"
            f"Pass mark for this stage: {PASS_MARKS[LYRIC_REVIEWER]:.0f}/100 with every "
            f"hard check passing and no dimension below 6/10.\n"
            "Count the syllables yourself before judging singability, and quote the "
            "offending line in `evidence` for every score below 7.\n"
            f"Deterministic pre-checks already run on the text:\n{mechanics}\n"
            "Return the structured review as one JSON object."
        )

    def _mechanics(self, state) -> str:
        sheet = state.lyrics
        if sheet is None:
            return "- (no lyric sheet available - this is itself a blocker)"
        lines: list[str] = []
        hits = T.find_cliches(sheet.full_text)
        lines.append(
            f"- cliche phrase scan: {len(hits)} hits" + (f" ({', '.join(hits[:6])})" if hits else "")
        )
        for section in sheet.sections:
            counts = [l.syllables for l in section.lines if l.text.strip()]
            if counts:
                lines.append(
                    f"- [{section.name}] {len(counts)} lines, "
                    f"syllables min/median/max = {min(counts)}/"
                    f"{sorted(counts)[len(counts) // 2]}/{max(counts)}"
                )
        lines.append(f"- total sung lines: {len([1 for s in sheet.sections for l in s.lines if l.text.strip()])}")
        lines.append(f"- vocabulary bloat index: {T.reading_level_penalty(sheet.full_text):.2f} (0 = tight)")
        chorus = next(
            (s for s in sheet.sections if "chorus" in s.name.lower() or "hook" in s.name.lower()),
            None,
        )
        if chorus is not None:
            ratio = T.repetition_ratio([l.text for l in chorus.lines])
            lines.append(f"- chorus verbatim repetition: {ratio * 100:.0f}%")
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
            problems.append(
                "Score all eight dimensions 0-10: " + ", ".join(DIMENSIONS)
            )
        if artifact.verdict not in {"pass", "revise", "reject"}:
            problems.append("verdict must be pass, revise or reject.")
        if artifact.verdict != "pass" and not artifact.issues:
            problems.append("A non-pass verdict requires at least one concrete issue.")
        for issue in artifact.issues:
            if not issue.message:
                problems.append("Every issue needs a `message`.")
                break
        return problems


__all__ = ["LyricReviewerAgent", "DIMENSIONS"]

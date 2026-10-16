"""Agent 9 - the Suno prompt QA reviewer.

The last gate. It reviews the prompt the way Suno's own parser will: character
budgets, tag hygiene, descriptor conflicts, negation collisions, vocal-gender
locks, and the risk of a render that comes back flat and over-compressed. It is
the reviewer with the widest rollback rights, because a prompt defect can have
its root cause anywhere upstream.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import LIMITS, PROMPT_REVIEWER
from ..domain.models import ReviewVerdict, as_float, as_str_list
from ..quality import text as T
from ..quality.gates import PASS_MARKS
from .base import Agent
from .prompts import (
    CONSTITUTION,
    CRAFT_MIX,
    JSON_DISCIPLINE,
    CONVERGENCE_RULE,
    SCHEMA_FIELD_NAMES,
    NEED_ARRANGEMENT,
    NEED_BLUEPRINT,
    NEED_BRIEF,
    NEED_LYRICS,
    NEED_MUSIC_REVIEW,
    NEED_PROMPT,
    NEED_REFERENCE,
    REVIEW_SCHEMA,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)

PROMPT_REVIEWER_CRAFT = """\
YOUR CRAFT: THINKING LIKE THE RENDERER
- Re-read the prompt as if you were the model that has to generate the audio with
  no other context. If a descriptor is ambiguous, contradictory, or un-actionable,
  the render will average it away. That is a defect, not a nitpick.
- Score each dimension 0-10 and quote the exact offending text:
  * prompt_fidelity - does the prompt describe the song the team actually built?
  * style_clarity - would a stranger know the genre, mood, tempo feel, and lead
    voice from the first 200 characters alone?
  * descriptor_balance - 5-8 decisive descriptors, no contradictions, no filler?
  * lyrics_format - canonical [Section] tags alone on their line, blank lines
    between sections, no inline tags, no stray bracket text in the style field?
  * exclusion_effectiveness - is the vocal gender locked, are the real failure
    modes excluded, and does the exclude field avoid negating what the style asks
    for?
  * structural_clarity - does the section order in the lyrics field match the
    arrangement map, with instrumental sections clearly marked?
  * vocal_direction - is the requested singer unambiguous (gender, register,
    delivery), and does nothing contradict it?
  * title_hook - is the title short, quotable, and from the hook?
  * render_risk - what is most likely to go wrong in the actual render (Suno
    inventing an intro vocal, a chorus that repeats identically, drums where the
    brief wanted none, garbled lyrics on dense lines, a brickwalled mix)?
  * lossless_readiness - does the prompt avoid impossible mastering promises and
    does the checklist give the user what they need for a true lossless export?
- Character limits are absolute. Suno truncates silently, so an over-limit field
  is a blocker even if the text is beautiful.
- Negations in the style field are unreliable: if it says "no drums", that is a
  defect to fix in the exclude field, not a stylistic choice.
- Promising "airy highs" or "crisp cymbals" over a reference that measured as
  lossy is a fidelity defect: the render will expose the destroyed high end.
- `restart_from` is the ROOT CAUSE, and you may choose any legal target:
  * the prompt simply does not transcribe what exists -> `prompt_engineer`
  * the lyric/nada alignment makes the prompt impossible to format cleanly
    (a section with 30 syllables, no instrumental breaks) -> `arranger`
  * the words themselves are un-singable or cliche -> `lyricist`
  * the form/harmony/tempo cannot be expressed in 5-8 descriptors without
    contradicting itself -> `composer`
  * the measured reference was misread and the prompt now promises the wrong
    fidelity or ignores the tempo/key evidence -> `audio_analyst`
  * the brief is self-contradictory so no prompt can satisfy it -> `intake`
- Only 85+ with every hard check passing means pass. Everything else gets a
  precise, numbered fix list. Never pass a prompt you would not paste into Suno
  yourself - but equally, once it clears 85 with clean checks, ship it and put your
  remaining wishes in `issues` at "minor" or "polish" severity. They will be kept in
  the run record; they must not hold the deliverable hostage.

THE PASS DECISION
- `pass` requires: no hard check failed, no scored dimension below 6/10, and
  `overall_score` at or above the pass mark in your prompt.
- `revise` requires a defect that would make the render come back wrong: an
  ambiguous field, a contradictory descriptor, a lyric line Suno will garble, a
  missing vocal lock. Cite the exact text.
- `reject` is for a prompt whose premise cannot be fixed at the prompt layer.
- Your score and verdict must agree. Ten 9s plus "revise" is a contradiction that
  costs a rebuild of the whole prompt without a reason.
"""

DIMENSIONS = (
    "prompt_fidelity",
    "style_clarity",
    "descriptor_balance",
    "lyrics_format",
    "exclusion_effectiveness",
    "structural_clarity",
    "vocal_direction",
    "title_hook",
    "render_risk",
    "lossless_readiness",
)


class PromptReviewerAgent(Agent):
    stage = PROMPT_REVIEWER
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
            NEED_PROMPT,
        )

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        return build_system(
            CONSTITUTION,
            PROMPT_REVIEWER_CRAFT,
            CRAFT_MIX,
            JSON_DISCIPLINE,
            CONVERGENCE_RULE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(REVIEW_SCHEMA),
            "DIMENSIONS TO SCORE (exactly these keys): " + ", ".join(DIMENSIONS),
            self._targets_block(),
            self._hard_numbers(state),
            context_block(state, self.needs(state)),
            revision_block(revision_notes, self.stage),
        )

    def _targets_block(self) -> str:
        from ..domain.graph import STAGE_INFO, legal_restarts

        names = ", ".join(
            f"'{stage}' ({STAGE_INFO[stage].title})" for stage in legal_restarts(self.stage)
        )
        return f"LEGAL restart_from values for you: {names}. Pick exactly one."

    def _hard_numbers(self, state) -> str:
        """Hand the reviewer the mechanical facts so its scores are grounded."""
        prompt = state.prompt
        if prompt is None:
            return "MECHANICAL FACTS: (no prompt produced - automatic blocker)"
        head = prompt.style[: LIMITS["style_critical_zone"]]
        coverage = T.descriptor_coverage(head)
        missing = [name for name, present in coverage.items() if not present]
        blank_lines_ok = "yes" if "\n\n" in prompt.lyrics else "NO - check it"
        style_negations = T.find_negations(prompt.style)
        promises = T.unsupported_promises(prompt.style)
        tags = T.extract_section_tags(prompt.lyrics)
        conflicts: list[str] = []
        conflicts.extend(T.contradicts_exclusions(prompt.style, prompt.exclude_styles))
        conflicts.extend(T.vocal_direction_conflict(prompt.style, prompt.exclude_styles))
        if state.brief is not None and state.brief.must_avoid:
            conflicts.extend(
                f"user asked to avoid '{hit}'"
                for hit in T.keyword_hits(prompt.style + " " + prompt.lyrics, state.brief.must_avoid)
            )
        lines = [
            "MECHANICAL FACTS (verified by the app - do not dispute these, judge them)",
            f"- title: {len(prompt.title)} chars (limit {LIMITS['title_chars']})",
            f"- style: {len(prompt.style)} chars (limit {LIMITS['style_chars']}), "
            f"{T.count_descriptors(prompt.style)} descriptors",
            f"- exclude: {len(prompt.exclude_style_text)} chars (limit {LIMITS['exclude_chars']}), "
            f"{len(prompt.exclude_styles)} tags",
            f"- lyrics: {len(prompt.lyrics)} chars (limit {LIMITS['lyrics_chars']}), "
            f"{len(tags)} section tags: {', '.join(tags[:12])}",
            f"- descriptor families inside the {LIMITS['style_critical_zone']}-char critical zone: "
            f"{', '.join(name for name, present in coverage.items() if present) or 'none'}"
            + (f" | MISSING: {', '.join(missing)}" if missing else ""),
            f"- bracketed tags found in the style field: {'YES - defect' if T.has_brackets(prompt.style) else 'none'}",
            f"- negations found in the style field: {', '.join(style_negations) if style_negations else 'none'}",
            f"- impossible mastering promises: {', '.join(promises) if promises else 'none'}",
            f"- style/exclude conflicts: {'; '.join(conflicts[:4]) if conflicts else 'none'}",
            f"- reference lossless verdict: {state.reference_lossless} "
            f"(quality {state.reference_quality}/100)",
            f"- lyric field blank-line separation between sections: {blank_lines_ok}",
        ]
        return "\n".join(lines)

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        return (
            f"Pass mark for this stage: {PASS_MARKS[PROMPT_REVIEWER]:.0f}/100 with every "
            "hard check passing and no dimension below 6/10.\n"
            "Review the final Suno prompt as the renderer would. Verify that it "
            "describes the song the team built, that the fields are unambiguous and "
            "within limits, and that the most likely render failures are prevented.\n"
            "Quote the offending text for every score below 7.\n"
            "Return the structured review as one JSON object."
        )

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
        return problems


__all__ = ["PromptReviewerAgent", "DIMENSIONS"]

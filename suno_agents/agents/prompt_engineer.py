"""Agent 8 - the translator into a Suno prompt.

Everything upstream exists so that this agent has nothing left to invent: it
formats. The four Suno fields have hard, mechanical rules (grounded in Suno's
own documented limits) and this stage is where 'a great song' becomes 'a great
render'. Character budgets are enforced in Python afterwards, so the model gets
exact numbers to hit rather than discovering them by failing.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import LIMITS, PROMPT_ENGINEER
from ..domain.models import SunoPrompt, as_str_list
from ..quality import text as T
from .base import Agent
from .prompts import (
    CONSTITUTION,
    CRAFT_MIX,
    JSON_DISCIPLINE,
    SCHEMA_FIELD_NAMES,
    NEED_ARRANGEMENT,
    NEED_BLUEPRINT,
    NEED_BRIEF,
    NEED_LYRICS,
    NEED_MUSIC_REVIEW,
    NEED_REFERENCE,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)
from .schemas import SUNO_PROMPT_SCHEMA

SUNO_CRAFT = f"""\
YOUR CRAFT: THE TWO-BOX INSTRUMENT
Suno gives you four fields. Each has a job, a hard limit, and a failure mode.

1. TITLE (max {LIMITS['title_chars']} chars)
   Two to five words, quotable, almost always lifted from the hook line. No
   parentheses, no "feat.", no emoji, no ALL CAPS.

2. STYLE OF MUSIC (max {LIMITS['style_chars']} chars, aim 320-520)
   - Plain natural language descriptors separated by commas. NEVER brackets - any
     [Tag] in this field is parsed as nothing or as garbage.
   - The first {LIMITS['style_critical_zone']} characters carry most of the weight. Put, in
     order: genre -> era/scene -> mood -> lead voice (type + delivery) -> 1-3
     defining instruments -> production character -> tempo/feel. Everything after
     the critical zone is refinement.
   - {LIMITS['descriptors_min']}-{LIMITS['descriptors_max']} decisive descriptors. Fewer than {LIMITS['descriptors_min']} yields a
     generic render; more than 12 start contradicting each other and the model
     averages them into mush.
   - No negations here. "no drums", "without reverb" belong in the exclude field.
   - No mastering or delivery claims: never "lossless", "24-bit", "mastered",
     "48 kHz", "clean master". Suno renders a mix; mastering is your job later.
   - No singer or band names. Describe the voice instead: "raspy female alto,
     conversational, slightly behind the beat". Names are both useless and a
     legal risk.
   - Front-load tempo as a number when the tempo matters ("92 BPM soul ballad"),
     because Suno follows explicit BPM better than words like "slow".

3. EXCLUDE STYLES (max {LIMITS['exclude_chars']} chars)
   - Comma-separated tags of what must NOT appear. This is the reliable way to
     remove things; inline negation in the style field is not.
   - ALWAYS lock the vocal gender here when vocals exist: excluding the opposite
     gender is what stops Suno flipping the singer. For a female lead: exclude
     "male vocals, male lead". For male: "female vocals, female lead".
   - For instrumental requests, exclude "vocals, lead vocals, singing, humming".
   - Also exclude the specific failure modes of this song: e.g. for a sparse
     ballad exclude "heavy drums, distorted guitars, EDM drop, auto-tuned vocals".
   - Keep it honest: never exclude a genre the arrangement clearly is.

4. LYRICS (max {LIMITS['lyrics_chars']} chars, aim under 3000)
   - Every section on its own line as a bracket tag: [Intro], [Verse 1],
     [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Instrumental Break],
     [Guitar Solo], [Outro].
   - Keep tag names canonical and short; add delivery cues inside the tag only
     when you must ("[Chorus - belted, full band]").
   - One blank line between sections. Never put a section tag in the middle of a
     lyric line.
   - Ad-libs go on their own line in parentheses, max 3 words.
   - Order sections so the form reads top to bottom exactly as the arrangement
     map says. For instrumental sections write the instrumental tag and, if
     useful, a one-line description of what carries the melody.
   - For instrumental songs, the field holds the section map plus short
     performance notes in brackets, nothing that looks like a sung line.

LOSS/LOSSLESS CHECKLIST (fill `lossless_checklist`)
   Give 4-6 concrete instructions for the user's post-render work so the final
   file is actually lossless and release-ready, adapted to this song: e.g.
   "render in WAV/44.1 kHz or higher", "leave -1 dBTP true-peak headroom in the
   DAW", "high-pass the vocal at 90 Hz before compressing", "check the low-mid
   200-400 Hz between kick and bass", "export 24-bit WAV, do not re-encode",
   "verify integrated loudness around -12 to -9 LUFS for streaming". Tailor them
   to what this arrangement actually needs; do not paste a generic list.

DISCIPLINE
- Do not invent music. Every descriptor must be traceable to the blueprint, the
  lyric sheet, or the brief. If the blueprint says brushed drums and a Rhodes,
  the style field says brushed drums and a Rhodes.
- If a budget is tight, cut adjectives before cutting information. Order of cuts:
  redundant mood words, then production adjectives, then tempo words (keep the
  BPM number), never the vocal descriptor or the genre.
- `character_budget` must report the actual character counts you produced for
  each field, as integers.
- `meta_notes` is for the user, not Suno: one short paragraph explaining the two
  or three decisions most likely to affect the render.
"""


class PromptEngineerAgent(Agent):
    stage = PROMPT_ENGINEER
    schema = SUNO_PROMPT_SCHEMA

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
            SUNO_CRAFT,
            CRAFT_MIX,
            JSON_DISCIPLINE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(SUNO_PROMPT_SCHEMA),
            self._field_brief(state),
            context_block(state, self.needs(state)),
            revision_block(revision_notes, self.stage),
        )

    def _field_brief(self, state) -> str:
        """Explicit, computed budgets so the model writes to a target."""
        blueprint = state.blueprint
        sheet = state.lyrics
        reference = state.reference_lossless
        lines = [
            "COMPUTED BUDGETS FOR THIS SONG",
            f"- title: <= {LIMITS['title_chars']} chars",
            f"- style: target 320-520 chars, hard max {LIMITS['style_chars']}; "
            f"first {LIMITS['style_critical_zone']} chars must already contain genre + mood + voice",
            f"- exclude: target 60-140 chars, hard max {LIMITS['exclude_chars']}",
            f"- lyrics: hard max {LIMITS['lyrics_chars']}, aim under 3000",
        ]
        if blueprint is not None:
            lines.append(
                f"- music facts to honour: {blueprint.key} {blueprint.mode}, "
                f"{blueprint.tempo_bpm:.0f} BPM, {blueprint.time_signature}, "
                f"groove '{blueprint.groove}'"
            )
            instruments = ", ".join(blueprint.instrumentation[:8]) or "not specified"
            lines.append(f"- defining instruments: {instruments}")
            form = " > ".join(f"{s.name}({s.bars})" for s in blueprint.sections)
            lines.append(f"- form for the lyrics field: {form}")
        if sheet is not None:
            lines.append(
                "- vocal direction: "
                + ("INSTRUMENTAL (no lead vocal)" if sheet.instrumental else sheet.vocal.describe())
            )
            if sheet.hook_line:
                lines.append(f"- hook line (title should come from here): {sheet.hook_line}")
        if reference in {"lossy", "lossy-transcoded"}:
            lines.append(
                f"- reference audio is {reference}: do NOT promise airy highs, crisp "
                "cymbals, or hi-fi clarity, and note the risk in meta_notes."
            )
        if state.music_review is not None and state.music_review.strengths:
            lines.append(
                "- keep the character the music reviewer praised: "
                + "; ".join(state.music_review.strengths[:3])
            )
        return "\n".join(lines)

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        return (
            "Produce the four Suno fields. Convert the arrangement into the lyrics "
            "field exactly as the form was mapped, write the style field front-loaded "
            "with the decisive descriptors, build the exclude field around the vocal "
            "gender lock plus this song's failure modes, and write the title from the "
            "hook.\nReport your real character counts in `character_budget` and write "
            "the lossless checklist for this specific arrangement.\n"
            "Reply with one JSON object only."
        )

    def parse(self, payload: Any, state) -> SunoPrompt:
        if not isinstance(payload, Mapping):
            raise ValueError("Suno prompt must be a JSON object.")
        prompt = SunoPrompt.from_dict(payload)
        prompt.lyrics = _normalise_lyrics(prompt.lyrics)
        prompt.style = _normalise_style(prompt.style)
        prompt.exclude_styles = [
            tag.strip(" -") for tag in prompt.exclude_styles if str(tag).strip(" -")
        ]
        budget = {
            "title": len(prompt.title),
            "style": len(prompt.style),
            "exclude": len(prompt.exclude_style_text),
            "lyrics": len(prompt.lyrics),
            "style_critical_zone": len(prompt.style[: LIMITS["style_critical_zone"]]),
            "descriptors": T.count_descriptors(prompt.style),
        }
        prompt.character_budget = budget
        if state.lyrics is not None and not prompt.vocal_gender_tag:
            if state.lyrics.instrumental:
                prompt.vocal_gender_tag = "instrumental"
            else:
                prompt.vocal_gender_tag = state.lyrics.vocal.gender
        # Deterministic enforcement of the limits: trimming here beats a failed gate.
        prompt.style = _enforce_style_limit(prompt.style)
        prompt.exclude_styles = _enforce_exclude_limit(prompt.exclude_styles)
        if len(prompt.title) > LIMITS["title_chars"]:
            prompt.title = prompt.title[: LIMITS["title_chars"]].rstrip(" ,-")
        return prompt

    def validate(self, artifact: SunoPrompt) -> list[str]:
        problems = list(artifact.validate())
        if T.has_brackets(artifact.style):
            problems.append("Remove every bracketed tag from the style field.")
        negations = T.find_negations(artifact.style)
        if negations:
            problems.append(
                "Move these negations from the style field into exclude_styles: "
                + ", ".join(negations[:4])
            )
        if not artifact.exclude_styles:
            problems.append(
                "exclude_styles is empty; lock the vocal gender and name the failure "
                "modes you want gone."
            )
        if len(artifact.lossless_checklist) < 4:
            problems.append("Provide 4-6 specific lossless checklist items.")
        return problems


def _normalise_style(style: str) -> str:
    """Tidy the style field without changing its content."""
    text = " ".join(str(style or "").replace("\r", "\n").split())
    # Convert any accidental newline list into a comma-separated descriptor list.
    text = text.replace(";", ",")
    parts = [p.strip(" .") for p in text.split(",")]
    seen: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.lower() in {s.lower() for s in seen}:
            continue
        seen.append(part)
    return ", ".join(seen)


def _enforce_style_limit(style: str, limit: int = LIMITS["style_chars"]) -> str:
    """Trim from the tail: the critical zone must never be touched."""
    if len(style) <= limit:
        return style
    descriptors = T.split_descriptors(style)
    kept: list[str] = []
    length = 0
    for descriptor in descriptors:
        addition = len(descriptor) + (2 if kept else 0)
        if length + addition > limit:
            break
        kept.append(descriptor)
        length += addition
    return ", ".join(kept) if kept else style[:limit].rstrip(" ,")


def _enforce_exclude_limit(
    excludes: Sequence[str], limit: int = LIMITS["exclude_chars"]
) -> list[str]:
    """Keep the highest-value exclusions when the field overflows.

    Vocal-gender locks and vocal removals stay; genre/instrument exclusions go
    last because they are the ones a user can also express elsewhere.
    """
    priority_words = ("vocal", "voice", "singing", "humming", "male", "female", "lead")
    ranked = sorted(
        excludes,
        key=lambda tag: (
            0 if any(word in tag.lower() for word in priority_words) else 1,
            len(tag),
        ),
    )
    kept: list[str] = []
    length = 0
    for tag in ranked:
        addition = len(tag) + (2 if kept else 0)
        if length + addition > limit:
            continue
        kept.append(tag)
        length += addition
    # Restore the user's reading order.
    return [tag for tag in excludes if tag in kept]


def _normalise_lyrics(lyrics: str) -> str:
    """Enforce Suno lyric-field hygiene: tags alone on a line, blank line between blocks."""
    raw_lines = [line.rstrip() for line in str(lyrics or "").replace("\r", "").split("\n")]
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        is_tag = bool(stripped) and stripped.startswith("[") and stripped.endswith("]")
        if is_tag and current:
            blocks.append(current)
            current = [stripped]
            continue
        if is_tag:
            current = [stripped]
            continue
        if not stripped:
            if current and len(current) > 1:
                blocks.append(current)
                current = []
            continue
        # Repair the classic defect: "[Chorus] line one" on a single line.
        if "[" in stripped and stripped.startswith("[") is False:
            index = stripped.find("[")
            head, tail = stripped[:index].strip(), stripped[index:].strip()
            if head:
                current.append(head)
            if tail.endswith("]") or "]" in tail:
                if current and len(current) > 1:
                    blocks.append(current)
                current = [tail]
                continue
        current.append(stripped)
    if current:
        blocks.append(current)
    rendered = ["\n".join(block) for block in blocks if any(x.strip() for x in block)]
    return "\n\n".join(rendered).strip()


__all__ = ["PromptEngineerAgent"]

"""Shared prompt material: the studio constitution every specialist obeys.

The whole point of this app is that nine narrow experts outperform one general
prompt. That only works if the shared law is small, absolute, and identical in
every stage prompt: JSON-only output, no invented facts, no filler descriptions,
and a hard line on what "good music" means here.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# constitution
# --------------------------------------------------------------------------- #

CONSTITUTION = """\
You are one specialist inside LIRIK STUDIO, a nine-agent studio that writes a
finished Suno prompt. You have exactly one job. Do not do the next agent's job.

STUDIO LAW (all of it applies to you)
1. Deliverable over commentary. Produce the artifact, not a discussion of it.
2. Never invent facts you were not given. If a fact is missing, infer it, mark it
   with `inference: true` where the schema allows, and move on. Do not ask
   questions in a JSON field.
3. Specific beats decorative, every time. "Fender Rhodes through a tape echo,
   86 BPM, brushed snare on 2 and 4" beats "warm nostalgic vibes".
4. Musical truth: rhythm is felt against a grid, melody is a shape over time,
   harmony sets expectation and then delays or pays it off, lyrics are sung
   syllables carrying stressed and unstressed beats. Every choice you make must
   serve at least one of those four.
5. Write for the human ear, not for a rubric. If a choice is technically correct
   but boring, it is wrong.
6. No AI tells: no "in a world where", no "tapestry of emotions", no rhyming
   "heart/part/start", no chorus that restates the title four times, no lyrics
   about singing/echoes/shadows/neon unless the brief asked for them.
7. Anti-cliche discipline: banned unless the user asked for them - rain, tears,
   broken glass, dancing in the dark, fire and ice, midnight city, embers.
8. Lossless mindset: assume the target is a clean 24-bit/48 kHz master. That
   means no clipped transients, no muddy 200-400 Hz build-up, real dynamic
   contrast between verses and choruses, and stereo width that is earned by
   arrangement rather than smeared by effects.
9. Language: write creative content (lyrics) in the brief's language. Write every
   JSON field name in English exactly as specified, and keep explanatory fields
   in clear, terse English.
10. Output ONE JSON object and nothing else: no markdown fences, no preamble, no
    trailing notes. The app parses your reply mechanically and rejects prose.
"""

JSON_DISCIPLINE = """\
OUTPUT CONTRACT
- Reply with a single JSON object matching the requested schema.
- Every required key must be present, even when the value is an empty list or "".
- Numbers are numbers (86, not "86 BPM"). Booleans are true/false.
- No trailing commas, no comments, no markdown fences.
- If you are unsure about a value, give your best professional estimate; never
  write "N/A", "unknown", or "tbd" for creative fields.
- BREVITY IS RELIABILITY. Your reply is parsed as JSON, and a reply longer than the
  output budget is truncated mid-object and thrown away entirely - a total loss of
  every stage of work that led here. Prefer one precise clause to three fluent ones.
  Keep prose fields to a sentence or two, and never restate a value you already gave
  in another field. Assume roughly 12000 characters is the ceiling for the whole
  object and aim well under it.
"""

# --------------------------------------------------------------------------- #
# craft knowledge injected into relevant stages
# --------------------------------------------------------------------------- #

CRAFT_MELODY = """\
MELODIC CRAFT
- A motif is 3-7 notes with a clear rhythmic identity. State it once, then
  transform it: transpose, invert, augment, truncate, or displace it to a new
  beat. Repetition without transformation is what makes AI music feel cheap.
- A chorus hook must be singable by someone who heard it once: mostly stepwise
  motion, one memorable leap (4th-6th, rarely larger), and a rhythmic figure that
  sits on the strong beats. Highest note in the song usually lands here.
- Pre-chorus raises tension: rising line, shorter note values, harmony that
  refuses to resolve, bass moving to a chord that is not the tonic.
- Contour over time matters more than any single phrase: verse should orbit a
  narrow range (a 5th or so), chorus opens up, bridge breaks the pattern.
- Never let the melody and the vocal line fight the harmony: strong chord tones
  land on strong beats, non-chord tones resolve by step.
"""

CRAFT_HARMONY = """\
HARMONIC CRAFT
- Functional progressions with intention: tonic establishes, subdominant moves,
  dominant (or its substitution) creates pull, and the resolution can be delayed
  for effect. Write roman numerals in `function`.
- Borrowed chords (bVII, iv in major, Picardy third), secondary dominants (V/vi),
  modal interchange, and pedal points are welcome. Random non-functional stacks
  are not.
- Cadence placement is a structural decision: end the chorus on a resolved
  cadence, deny it at the end of a verse to keep motion going.
- Register and voicing matter: keep the chord tones in a range where the lead
  vocal sits on top rather than in the middle. Say where the voice sits.
- Name extensions precisely: Am7, Fmaj9, Cadd9, G/B, Dsus4 -> D. No "jazzy chord".
"""

CRAFT_PROSODY = """\
PROSODY CRAFT (words meeting notes)
- Count syllables per line, then place them on the grid: a bar of 4/4 comfortably
  holds 4-8 sung syllables at mid tempo; 10+ needs sixteenth-note motion which
  reads as rap or rushed phrasing.
- English/Indonesian lyric stress: place the naturally stressed syllable of a
  word on a strong beat. Never stress a schwa. Never put a one-syllable word that
  carries no meaning on a downbeat if a meaningful word could take it.
- Open vowels sustain: "ou", "ah", "oh", "ay" carry long notes. Closed vowels
  ("it", "up", "she") are for short notes or rhythmic passages.
- Plan breath: roughly every 2 bars at mid tempo, at punctuation or a melodic
  rest. Mark it.
- Melisma only on emotive words, and only 2-4 notes per syllable.
"""

CRAFT_MIX = """\
DELIVERY / LOSSLESS CRAFT
- Describe the arrangement in layers so the master has room: sub (30-60 Hz),
  low (60-250), low-mid (250-500), mid (500-2k), high-mid (2-6k), high (6-12k),
  air (12-20k). Say which layer each instrument occupies.
- Dynamic contrast is the cheapest way to sound expensive: verse 6-7 dB below
  chorus, chorus vocal fully forward, bridge suddenly intimate.
- Stereo: hard-panned doubles for width, mono-compatible low end, no wide bass.
- Target master: integrated loudness around -12 to -9 LUFS for a streaming-ready
  export, true peak under -1 dBTP, crest factor above 8 dB. Never ask Suno for a
  "loud mastered" sound - ask for the mix, then master outside.
"""

CRAFT_LYRICS = """\
LYRIC CRAFT
- One image per line is better than three. Concrete nouns outperform adjectives.
- The chorus must state the song's thesis in a phrase a stranger could repeat.
- Show the change: verse 1 sets the situation, chorus names the cost or the
  claim, verse 2 complicates it, bridge changes the camera angle, last chorus
  lands differently (new line, new harmony, drop the band, or add the answer).
- Keep line lengths in a section within +/-2 syllables of each other so the
  melody does not have to improvise.
- Titles: 2-5 words, quotable, ideally lifted from the hook line.
"""

CRAFT_VOCAL = """\
VOCAL SPECIFICATION
- Gender is a request, not a detail: Suno follows `male`/`female` hard. State it
  explicitly and consistently (style field, vocal spec, exclude field).
- Voice type means something measurable: range (e.g. baritone, roughly G2-F4),
  register habits (chest-dominant, mixed belt, falsetto flips), timbre (husky,
  reedy, smoky, glassy, warm), vibrato (narrow/fast vs wide/slow), and delivery
  (conversational, crooned, belted, spoken-sung, whispered).
- In a duet, define who takes which line and whether they harmonise in 3rds/6ths
  or trade call-and-response.
- Instrumental means no lead vocal at all; describe the vocal texture used as an
  instrument (oohs, wordless pads, chopped syllables) or explicitly forbid it.
"""

# --------------------------------------------------------------------------- #
# context rendering
# --------------------------------------------------------------------------- #

NEED_BRIEF = "brief"
NEED_REFERENCE = "reference"
NEED_BLUEPRINT = "blueprint"
NEED_LYRICS = "lyrics"
NEED_LYRIC_REVIEW = "lyric_review"
NEED_ARRANGEMENT = "arrangement"
NEED_MUSIC_REVIEW = "music_review"
NEED_PROMPT = "prompt"


def _bullet_list(items: Iterable[str], empty: str = "-") -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    return "; ".join(values) if values else empty


def context_block(state, needs: Sequence[str]) -> str:
    """Render exactly the artifacts a stage is allowed to see.

    Keeping this explicit is what stops the composer from reading the lyrics and
    the lyricist from peeking at the prompt: role isolation is a design feature.
    """
    parts: list[str] = []
    for need in needs:
        if need == NEED_BRIEF and state.brief is not None:
            parts.append("=== BRIEF (user intent, authoritative) ===\n" + state.brief.summary())
        elif need == NEED_REFERENCE and state.reference_summary:
            parts.append(
                "=== AUDIO REFERENCE ANALYSIS (measured, authoritative) ===\n"
                f"Lossless verdict: {state.reference_lossless} | "
                f"reference quality score: {state.reference_quality}/100\n"
                + state.reference_summary
            )
        elif need == NEED_BLUEPRINT and state.blueprint is not None:
            parts.append("=== MUSIC BLUEPRINT (composer output) ===\n" + state.blueprint.summary())
        elif need == NEED_LYRICS and state.lyrics is not None:
            parts.append("=== LYRIC SHEET (lyricist output) ===\n" + _lyric_block(state.lyrics))
        elif need == NEED_LYRIC_REVIEW and state.lyric_review is not None:
            parts.append("=== LYRIC REVIEW (previous verdict) ===\n" + _review_block(state.lyric_review))
        elif need == NEED_ARRANGEMENT and state.arrangement is not None:
            parts.append("=== ARRANGEMENT MAP (lyric-note alignment) ===\n" + state.arrangement.summary())
        elif need == NEED_MUSIC_REVIEW and state.music_review is not None:
            parts.append("=== MUSIC REVIEW (previous verdict) ===\n" + _review_block(state.music_review))
        elif need == NEED_PROMPT and state.prompt is not None:
            parts.append(
                "=== CURRENT SUNO PROMPT ===\n"
                f"TITLE: {state.prompt.title}\n"
                f"STYLE ({state.prompt.style_length} chars): {state.prompt.style}\n"
                f"EXCLUDE ({len(state.prompt.exclude_style_text)} chars): "
                f"{state.prompt.exclude_style_text}\n"
                f"LYRICS ({state.prompt.lyrics_length} chars):\n{state.prompt.lyrics}"
            )
    return "\n\n".join(parts)


def _lyric_block(sheet) -> str:
    lines = [
        f"Language: {sheet.language} | Perspective: {sheet.perspective} | Tense: {sheet.tense}",
        f"Hook line: {sheet.hook_line}",
        f"Emotional arc: {sheet.emotional_arc}",
        f"Title options: {_bullet_list(sheet.title_options)}",
        f"Vocal: {sheet.vocal.describe()}",
    ]
    if sheet.instrumental:
        lines.append(f"Instrumental direction: {sheet.instrumental_direction}")
    for section in sheet.sections:
        lines.append(f"[{section.name}] cue: {section.performance_cue or '-'}")
        for line in section.lines:
            count = f" ({line.syllables} syll)" if line.syllables else ""
            cue = f" <{line.cue}>" if line.cue else ""
            adlib = f" +(adlib: {line.adlib})" if line.adlib else ""
            lines.append(f"  {line.text}{count}{cue}{adlib}")
    if sheet.notes:
        lines.append(f"Notes: {sheet.notes}")
    return "\n".join(lines)


def _review_block(verdict) -> str:
    scores = ", ".join(f"{k}={v:.1f}" for k, v in verdict.dimensions.items()) or "-"
    return (
        f"Stage: {verdict.stage} | verdict: {verdict.verdict} | score: {verdict.overall_score:.1f}\n"
        f"Dimensions: {scores}\n"
        f"Gates failed: {_bullet_list(verdict.gates_failed)}\n"
        f"Requested restart: {verdict.restart_from or '-'}\n"
        f"Reasoning: {verdict.reasoning}\n"
        f"Issues:\n{verdict.issue_brief(12)}\n"
        f"Strengths to keep: {_bullet_list(verdict.strengths)}"
    )


def revision_block(notes: Sequence[str], target_stage: str) -> str:
    """What a stage sees when it is re-run after a review failure."""
    if not notes:
        return ""
    body = "\n".join(f"- {note}" for note in notes if note)
    return (
        f"REVISION PASS - you are stage `{target_stage}` and you are being re-run.\n"
        "The reviewer rejected the previous output. Fix every point below, keep "
        "everything that already worked, and do not regress the strengths listed.\n"
        f"{body}\n"
        "Rewrite the FULL artifact, not a patch. Output the same JSON schema."
    )


def json_schema_hint(schema: Mapping[str, object]) -> str:
    """Render a compact, human-readable schema hint for the prompt tail."""
    lines = ["REQUIRED JSON SHAPE"]

    def walk(node: object, indent: int, name: str = "") -> None:
        pad = "  " * indent
        if isinstance(node, Mapping):
            if "type" in node and node.get("type") in {"string", "number", "integer", "boolean"}:
                extra = ""
                if node.get("enum"):
                    extra = f" (one of: {', '.join(map(str, node['enum']))})"
                lines.append(f"{pad}{name}: <{node['type']}>{extra}")
                return
            if node.get("type") == "array":
                lines.append(f"{pad}{name}: [")
                walk(node.get("items", {}), indent + 1)
                lines.append(f"{pad}]")
                return
            for key, value in (node.get("properties") or {}).items():
                walk(value, indent, str(key))
            return
        lines.append(f"{pad}{name}: {node}")

    walk(schema, 0)
    return "\n".join(lines)


CONVERGENCE_RULE = """\
DECIDING pass vs revise - READ THIS BEFORE YOU ANSWER

You are one pass in a loop that has a finite budget. A review that never accepts
anything is as useless as one that accepts everything, because the loop spends its
budget re-reading instead of shipping. So judge on a threshold, not on a feeling.

Set `verdict` to "pass" when BOTH are true:
  1. every hard check you were given passes, and no single scored dimension is
     below 6/10, and
  2. your `overall_score` is at or above the pass mark stated in your prompt.

Set `verdict` to "revise" only when a specific, nameable defect means this artifact
would embarrass the project - an un-singable line, a broken image, an ambiguity the
renderer will resolve the wrong way, a contradiction inside the artifact itself.
Name it, cite it, and say where the root cause lives.

SEVERITY SCALE - use it as written, or the review loses its signal
- `blocker`: the artifact cannot ship at all. Reserve this for the handful of
  defects that break the deliverable, and put them in `verdict: "revise"`.
- `major`: a real defect that should be fixed, but that a listener would not call
  broken. At most 2-3 per review.
- `minor`: worth fixing in some later pass. Does NOT block, and does not need a
  rewrite. Most observations belong here.
- `polish`: a preference. Purely informational.
Measured against a real reviewer: 48 items labelled `major` in 28 consecutive
reviews, with dimensions scored 8/10 in the same reviews. That is severity
inflation, and it makes every "major" meaningless: if everything is major, nothing
is. A dimension scoring 8/10 means the artifact is in good shape - your issues
should read that way too.

Set `verdict` to "reject" when the material's premise is wrong rather than its
execution, so no rewrite of this artifact can save it.

Do NOT use "revise" for:
  * improvements you can describe but that leave the song perfectly releasable
    ("this could be even stronger if...", "one could also...", "the line is good
    but not the best line").
  * polish that belongs to a later production stage.
  * a wish to see a different artistic choice that the brief does not require.
Those belong in `issues` with severity "minor" or "polish", and they do NOT stop
the pipeline. A score at or above the pass mark with clean hard checks IS a pass,
even if you can still imagine a better version - every song can be improved, and
holding work for that reason means nothing ever ships.

Calibration: a 90+ artifact ships as-is. An 85-89 artifact is releasable with the
notes you listed. An 80-84 artifact is releasable but its listed defects must be
named precisely. Below the pass mark, name the root cause and request a revision.
"""


SCHEMA_FIELD_NAMES = """\
FIELD NAMES ARE EXACT
- Use these key spellings verbatim: `stage`, `verdict`, `passed`, `overall_score`,
  `dimensions`, `gates_failed`, `issues`, `restart_from`, `reasoning`, `strengths`.
- `dimensions` is a JSON OBJECT whose keys are exactly the dimension names listed
  for your stage, each holding a number 0-10:
  {"dimensions": {"<dimension>": 8.5, ...}}. An empty `dimensions` object is a
  failed review: the score sheet is the substance of your report.
- `restart_from` is a single stage id string from your legal list, or "" when you
  pass. Do not invent a stage name and do not return a list.
- `verdict` is exactly one of "pass", "revise", "reject".
- Do not rename keys, do not nest the payload, do not return a wrapper object.
"""


def build_system(*blocks: str) -> str:
    """Join prompt blocks with clear separators."""
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


# --------------------------------------------------------------------------- #
# reusable JSON schema fragments
# --------------------------------------------------------------------------- #

REVIEW_ISSUE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "severity": {"type": "string", "enum": ["blocker", "major", "minor", "polish"]},
        "dimension": {"type": "string"},
        "message": {"type": "string"},
        "fix": {"type": "string"},
        "evidence": {"type": "string"},
        "restart_from": {"type": "string"},
    },
}

REVIEW_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "stage": {"type": "string"},
        # `dimensions` is required and must be an OBJECT keyed by the exact
        # dimension names listed for the stage: a model that omits it leaves the
        # scoring sheet empty and the review loses its substance.
        "verdict": {"type": "string", "enum": ["pass", "revise", "reject"]},
        "passed": {"type": "boolean"},
        "overall_score": {"type": "number"},
        "dimensions": {"type": "object"},
        "gates_failed": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": REVIEW_ISSUE_SCHEMA},
        "restart_from": {"type": "string"},
        "reasoning": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
    },
}

__all__ = [
    "CONSTITUTION",
    "JSON_DISCIPLINE",
    "CRAFT_MELODY",
    "CRAFT_HARMONY",
    "CRAFT_PROSODY",
    "CRAFT_MIX",
    "CRAFT_LYRICS",
    "CRAFT_VOCAL",
    "NEED_BRIEF",
    "NEED_REFERENCE",
    "NEED_BLUEPRINT",
    "NEED_LYRICS",
    "NEED_LYRIC_REVIEW",
    "NEED_ARRANGEMENT",
    "NEED_MUSIC_REVIEW",
    "NEED_PROMPT",
    "REVIEW_SCHEMA",
    "CONVERGENCE_RULE",
    "SCHEMA_FIELD_NAMES",
    "REVIEW_ISSUE_SCHEMA",
    "build_system",
    "context_block",
    "json_schema_hint",
    "revision_block",
]

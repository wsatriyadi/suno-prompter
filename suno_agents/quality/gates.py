"""Deterministic hard gates for every artifact.

A gate is a fact, not an opinion. The reviewers score taste; these gates decide
what is *allowed*. When any gate fails the orchestrator refuses to accept the
run and uses the gate's ``restart_from`` to send work back, regardless of what
the reviewer's prose said. That is what makes "reviewer menentukan mundur dari
mana" predictable instead of model-dependent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..domain.graph import (
    ARRANGER,
    AUDIO_ANALYST,
    COMPOSER,
    INTAKE,
    LIMITS,
    LYRICIST,
    PROMPT_ENGINEER,
)
from ..domain.models import (
    Arrangement,
    CreativeBrief,
    LyricSheet,
    MusicBlueprint,
    ReviewIssue,
    SunoPrompt,
)
from . import text as T

def is_soft(result: GateResult) -> bool:
    """True when a failed gate must not veto a pass (see :class:`GateResult`)."""
    return (not result.ok) and result.soft


#: Default pass mark per stage. Reviewers may not pass a stage below this.
#:
#: How to choose one: it should sit below a score the reviewer would want to
#: *improve* and above one it would call *broken*. Set too low and mediocre songs
#: ship; set too high and the reviewer's own "revise" impulse outranks the loop's
#: ability to converge.
PASS_MARKS: dict[str, float] = {
    "lyric_reviewer": 80.0,
    "music_reviewer": 80.0,
    "prompt_reviewer": 85.0,
}

#: Score at or above which a clean-gate review is treated as an acceptance even
#: though the model said "revise", *provided* one real repair pass has already
#: happened. The model is the critic; the loop should not chase "make me say pass
#: on principle" forever.
#:
#: This is not a bypass, it is a stopping rule, and it is deliberately guarded:
#:   * only applies from the stage's second attempt onward, so the reviewer always
#:     gets one full repair round to act on its own findings;
#:   * requires no hard gate and no low-dimension failure;
#:   * requires the margin above the pass mark shown in ``_ACCEPT_MARGIN``.
#:
#: It exists because of a measured live failure: a reviewer produced 85.0 and then
#: 90.7 with zero hard gate failures and kept saying "revise", so the pipeline would
#: have spun until the revision budget was exhausted without ever delivering, even
#: though its own numbers said the work was ready.
ACCEPTANCE_SCORES: dict[str, float] = {
    "lyric_reviewer": 84.0,
    "music_reviewer": 84.0,
    "prompt_reviewer": 88.0,
}

#: Minimum attempt index for the acceptance rule above (0-based).
ACCEPTANCE_MIN_ATTEMPT = 1

#: Minimum dimension score a reviewer may award and still pass.
DIMENSION_FLOOR = 6.0


@dataclass
class GateResult:
    """Outcome of one check.

    ``soft`` marks a failure that costs score and is reported, but may never veto
    a pass. The distinction has to live on the *result*, not on the code, because
    one gate routinely contains both kinds of judgement: "the lyrics field has no
    section tags at all" is a fact that breaks the render, while "the lyrics field
    has fewer tags than ideal" is taste. Found the hard way - a global code list
    forced a whole gate to be one or the other, so a calibration complaint about
    descriptor count blocked three live generations in a row while the reviewer was
    scoring that same dimension 8/10 and asking to ship.
    """

    code: str
    ok: bool
    message: str = ""
    fix: str = ""
    restart_from: str = ""
    severity: str = "major"
    soft: bool = False

    def to_issue(self, dimension: str = "") -> ReviewIssue:
        return ReviewIssue(
            code=self.code,
            severity=self.severity,
            dimension=dimension or "gate",
            message=self.message,
            fix=self.fix,
            evidence=self.code,
            restart_from=self.restart_from,
        )


def _fail(
    code: str,
    message: str,
    *,
    restart_from: str,
    fix: str = "",
    severity: str = "blocker",
    soft: bool = False,
) -> GateResult:
    return GateResult(code, False, message, fix, restart_from, severity, soft)


def _ok(code: str) -> GateResult:
    return GateResult(code, True)


# --------------------------------------------------------------------------- #
# gate sets
# --------------------------------------------------------------------------- #


#: A quoted fragment is the only reliable sign that a requirement is text a lyric is
#: expected to contain. Everything else - structure, style, rhythm, mood, quality
#: direction - describes what the song should BE, and is judged by a reviewer reading
#: the whole thing, never by a substring search.
_QUOTED_RE = re.compile(r"""["'\u2018\u2019\u201c\u201d]([^"'\u2018\u2019\u201c\u201d]{2,80})["'\u2018\u2019\u201c\u201d]""")


def literal_requirements(items: Sequence[str]) -> list[tuple[str, str]]:
    """Return ``(requirement, quoted_phrase)`` pairs that can be checked as text.

    An earlier version tried to classify each requirement as "structural" or
    "literal" with a keyword list, and it failed exactly where it mattered: a real
    brief asked for "authentic Caribbean Spanish idioms and cadence" and "empowering
    affirmation phrases", both parsed as literal, so the gate demanded that a lyric
    contain those sentences verbatim. It could not, so the gate failed forever with
    an identical message while the reviewer scored the same work 87/100 and asked to
    ship. Three rollbacks, no progress.

    The narrow rule that replaced it: only an explicitly quoted fragment is treated
    as text to find. Everything else is a directive for the reviewers, which is the
    honest reading - "authentic idioms" is a judgement, not a string.
    """
    found: list[tuple[str, str]] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        match = _QUOTED_RE.search(text)
        if match:
            found.append((text, match.group(1).strip()))
    return found


def is_structural_requirement(text: str) -> bool:
    """Kept for callers that ask the old yes/no question.

    Now defined as "not checkable as literal text", which is the property that
    actually decides whether a gate may block on it.
    """
    return not bool(_QUOTED_RE.search(str(text or ""))) and bool(str(text or "").strip())


def partition_must_include(items: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``must_include`` into ``(literal, directive)`` entries.

    ``literal`` entries carry a quoted phrase to search for; ``directive`` entries
    are left to the reviewers.
    """
    literal: list[str] = []
    directive: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        (literal if _QUOTED_RE.search(text) else directive).append(text)
    return literal, directive




LYRIC_GATES: tuple[str, ...] = (
    "LG1_structure",
    "LG2_singability",
    "LG3_instrumental_consistency",
    "LG4_cliche",
    "LG5_hook",
    "LG6_vocal_spec",
    "LG7_user_constraints",
    "LG8_language",
)

MUSIC_GATES: tuple[str, ...] = (
    "MG1_form",
    "MG2_harmony",
    "MG3_energy_contrast",
    "MG4_tempo",
    "MG5_duration",
    "MG6_hook_and_motif",
    "MG7_user_constraints",
    "MG8_alignment",
)

PROMPT_GATES: tuple[str, ...] = (
    "PG1_limits",
    "PG2_style_hygiene",
    "PG3_lyric_tags",
    "PG4_descriptor_count",
    "PG5_no_negation_in_style",
    "PG6_exclude_sanity",
    "PG7_front_load",
    "PG8_vocal_consistency",
    "PG9_lossless_honesty",
    "PG10_instrumental_consistency",
)


def pass_criteria(stage: str) -> str:
    """Human-readable statement of exactly what makes ``stage`` pass.

    Lives beside the thresholds it describes, so the UI, the CLI and the prompts all
    read from one place. Added after a user asked "what do I need to pass?" while
    looking at a review panel that showed the verdict but never the requirement -
    the answer existed only in the code.
    """
    from ..domain.graph import STAGE_INFO

    title = STAGE_INFO[stage].title if stage in STAGE_INFO else stage
    mark = PASS_MARKS.get(stage)
    accept = ACCEPTANCE_SCORES.get(stage)
    if mark is None:
        return f"{title}: bukan tahap reviewer, tidak ada ambang lulus."
    lines = [
        f"SYARAT LULUS - {title}",
        f"Agregat skor (bobot: penilaian reviewer 62%, gate mekanis 38%) "
        f"harus >= {mark:.0f}/100.",
        f"Model penilai harus menyatakan verdict \"pass\"; skor bagus saja tidak cukup.",
        f"Tidak ada satu pun gate keras boleh gagal.",
        f"Tidak ada dimensi bernilai < {DIMENSION_FLOOR:.0f}/10.",
    ]
    if accept:
        lines.append(
            f"Jaring pengaman: bila semua gate bersih, tidak ada dimensi lemah, dan "
            f"skor >= {accept:.0f}/100 pada percobaan ke-{ACCEPTANCE_MIN_ATTEMPT + 1} atau "
            f"setelahnya, karya diterima walau reviewer masih minta revisi."
        )
    soft = sorted({code for code in _SOFT_CODES})
    if soft:
        lines.append(
            "Gate berikut bersifat lunak: kegagalannya mengurangi skor dan dicatat, "
            "tapi tidak memblokir: " + ", ".join(soft) + "."
        )
    return "\n".join(lines)


#: Gate codes that contain at least one advisory (non-blocking) branch. Derived
#: from the gate results themselves by the test suite, listed here for display.
_SOFT_CODES: frozenset[str] = frozenset(
    {"LG6_vocal_spec", "PG3_lyric_tags", "PG4_descriptor_count", "PG7_front_load"}
)


def run_lyric_gates(
    sheet: LyricSheet, brief: CreativeBrief | None, blueprint: MusicBlueprint | None
) -> list[GateResult]:
    """Hard checks on the lyric sheet."""
    results: list[GateResult] = []

    # LG1 - structural completeness.
    if not sheet.sections:
        results.append(
            _fail(
                "LG1_structure",
                "Lyric sheet has no sections.",
                restart_from=LYRICIST,
                fix="Write lyric sections matching the blueprint form.",
            )
        )
    else:
        _ok("LG1_structure")

    # LG2 - singability: line length spread and total content.
    sung = [l for s in sheet.sections for l in s.lines if l.text.strip()]
    if sheet.instrumental:
        results.append(_ok("LG2_singability"))
    elif len(sung) < 8:
        results.append(
            _fail(
                "LG2_singability",
                f"Only {len(sung)} lyric lines - not enough for a full song.",
                restart_from=LYRICIST,
                fix="Write complete verses, choruses and a bridge.",
            )
        )
    else:
        worst = ""
        for section in sheet.sections:
            counts = [l.syllables for l in section.lines if l.text.strip()]
            if len(counts) >= 3 and max(counts) - min(counts) > 7:
                worst = (
                    f"Section '{section.name}' swings {min(counts)}-{max(counts)} "
                    "syllables per line."
                )
                break
        if worst:
            results.append(
                _fail(
                    "LG2_singability",
                    worst,
                    restart_from=LYRICIST,
                    fix="Even out line lengths so one melody can carry the section.",
                    severity="major",
                )
            )
        else:
            results.append(_ok("LG2_singability"))

    # LG3 - instrumental brief must not carry words.
    if brief is not None and brief.instrumental:
        if sung:
            results.append(
                _fail(
                    "LG3_instrumental_consistency",
                    "Brief is instrumental but the sheet contains sung lines.",
                    restart_from=LYRICIST,
                    fix="Remove all lyric lines; describe the instrumental topline instead.",
                )
            )
        elif not sheet.instrumental_direction.strip():
            results.append(
                _fail(
                    "LG3_instrumental_consistency",
                    "Instrumental sheet has no instrumental_direction.",
                    restart_from=LYRICIST,
                    fix="Describe what carries the melody instead of a voice.",
                )
            )
        else:
            results.append(_ok("LG3_instrumental_consistency"))
    elif sheet.instrumental and not (brief and brief.instrumental):
        results.append(
            _fail(
                "LG3_instrumental_consistency",
                "Sheet is marked instrumental but the brief asks for vocals.",
                restart_from=INTAKE,
                fix="Confirm with the brief whether vocals are wanted.",
            )
        )
    else:
        results.append(_ok("LG3_instrumental_consistency"))

    # LG4 - cliche density.
    hits = T.find_cliches(sheet.full_text)
    if len(hits) >= 3:
        results.append(
            _fail(
                "LG4_cliche",
                f"{len(hits)} cliche phrases found: {', '.join(hits[:5])}.",
                restart_from=LYRICIST,
                fix="Replace each with a concrete, specific image from this story.",
            )
        )
    else:
        results.append(_ok("LG4_cliche"))

    # LG5 - the chorus must exist and not be pure repetition.
    if not sheet.instrumental:
        chorus = next(
            (s for s in sheet.sections if re.search(r"chorus|refrain|hook", s.name, re.I)),
            None,
        )
        if chorus is None:
            results.append(
                _fail(
                    "LG5_hook",
                    "No chorus/refrain section in the lyric sheet.",
                    restart_from=COMPOSER,
                    fix="Add a chorus to the form and write it.",
                )
            )
        else:
            texts = [line.text for line in chorus.lines if line.text.strip()]
            ratio = T.repetition_ratio(texts)
            unique_lines = {re.sub(r"[^\w\s]", "", t).strip().lower() for t in texts}
            # A chorus may legitimately repeat a couplet, but four identical lines is
            # a placeholder, not a hook. Both signals are cheap and together they
            # catch the real defect without punishing honest repetition.
            if len(texts) >= 4 and (len(unique_lines) <= 2 or ratio >= 0.6):
                results.append(
                    _fail(
                        "LG5_hook",
                        f"Chorus has only {len(unique_lines)} distinct line(s) "
                        f"({ratio * 100:.0f}% verbatim repetition).",
                        restart_from=LYRICIST,
                        fix="Keep the hook line but vary the surrounding lines.",
                        severity="major",
                    )
                )
            elif not sheet.hook_line.strip():
                results.append(
                    _fail(
                        "LG5_hook",
                        "No hook_line identified.",
                        restart_from=LYRICIST,
                        fix="Name the single line the listener should remember.",
                    )
                )
            else:
                results.append(_ok("LG5_hook"))
    else:
        results.append(_ok("LG5_hook"))

    # LG6 - vocal specification.
    if sheet.instrumental:
        results.append(_ok("LG6_vocal_spec"))
    else:
        # Only gender is blocking, and the split is deliberate. Suno steers hard on
        # gender - a wrong or missing gender flips the singer and ruins the render -
        # whereas voice_type/register/delivery refine fidelity without invalidating
        # the prompt. Blocking on all four produced a false blocker in a live run:
        # the reviewer scored 83.4 against a pass mark of 80 and wanted to ship, but
        # the gate refused, so lyricist and lyric reviewer ping-ponged until the
        # revision budget ran out. The missing refinements stay visible as a scored
        # deduction (`vocal_fit`) and as a note here, so nothing is hidden.
        missing = sheet.missing_vocal_fields()
        if "gender" in missing:
            results.append(
                _fail(
                    "LG6_vocal_spec",
                    "Vocal gender is missing, so the renderer may assign the wrong "
                    "singer entirely.",
                    restart_from=LYRICIST,
                    fix='Set vocal.gender to "male", "female" or "mixed".',
                )
            )
        elif missing:
            # A calibration complaint, not a fact: the prompt is renderable, the
            # renderer just gets less to steer on. Costs score, never blocks.
            results.append(
                _fail(
                    "LG6_vocal_spec",
                    "vocal spec is thin (missing "
                    + ", ".join(missing)
                    + "); renderable, but the renderer gets less to steer on",
                    restart_from="",
                    fix="Adding voice_type, register and delivery raises fidelity.",
                    severity="minor",
                    soft=True,
                )
            )
        else:
            results.append(_ok("LG6_vocal_spec"))

    # LG7 - user's must_include / must_avoid.
    #
    # Two very different things hide behind `must_include`. Some entries are
    # literal requirements a lyric should contain ("sebut nama kota asal"), and
    # some describe the shape or placement of the song ("pengakuan di bar terakhir
    # chorus terakhir") which a lyric can never satisfy verbatim. Demanding the
    # literal text for the second kind produced a gate no lyricist could ever
    # pass: every attempt failed, so the loop rolled back forever. Only literal
    # entries are checked as text; structural ones are judged by the reviewers.
    if brief is None:
        results.append(_ok("LG7_user_constraints"))
    else:
        violated = T.keyword_hits(sheet.full_text, brief.must_avoid)
        quoted = literal_requirements(brief.must_include)
        if violated:
            results.append(
                _fail(
                    "LG7_user_constraints",
                    f"Lyrics contain forbidden terms: {', '.join(violated)}.",
                    restart_from=LYRICIST,
                    fix="Remove every word the user asked to avoid.",
                )
            )
        elif quoted:
            blob = sheet.full_text + " " + sheet.notes
            missing = [phrase for _item, phrase in quoted if phrase.lower() not in blob.lower()]
            if missing:
                # A quoted line the user explicitly asked for is a real requirement,
                # so this is worth one revision - but only when NONE of them appear,
                # because partial delivery is the lyricist's call, not a gate's.
                if len(missing) == len(quoted):
                    results.append(
                        _fail(
                            "LG7_user_constraints",
                            "None of the quoted lines the user asked for appear in the "
                            "lyrics: " + ", ".join(f'"{m}"' for m in missing),
                            restart_from=LYRICIST,
                            fix="Weave each quoted line into the lyric as written.",
                            severity="major",
                        )
                    )
                else:
                    results.append(_ok("LG7_user_constraints"))
            else:
                results.append(_ok("LG7_user_constraints"))
        else:
            directives = partition_must_include(brief.must_include)[1]
            results.append(
                GateResult(
                    "LG7_user_constraints",
                    True,
                    message=(
                        "no quoted text required; directives judged by the reviewers: "
                        + "; ".join(directives[:4])[:200]
                        if directives
                        else "nothing to check"
                    ),
                )
            )

    # LG8 - language consistency with the brief.
    if brief is not None and brief.language and not sheet.instrumental:
        if sheet.language and sheet.language.split("-")[0] != brief.language.split("-")[0]:
            results.append(
                _fail(
                    "LG8_language",
                    f"Sheet language '{sheet.language}' differs from the brief "
                    f"'{brief.language}'.",
                    restart_from=LYRICIST,
                    fix="Rewrite the lyrics in the brief's language.",
                    severity="major",
                )
            )
        else:
            results.append(_ok("LG8_language"))
    else:
        results.append(_ok("LG8_language"))

    return results


def run_music_gates(
    blueprint: MusicBlueprint,
    arrangement: Arrangement,
    brief: CreativeBrief | None,
    sheet: LyricSheet | None,
) -> list[GateResult]:
    """Hard checks on the music blueprint and its lyric alignment."""
    results: list[GateResult] = []

    # MG1 - form sanity.
    if not blueprint.sections:
        results.append(
            _fail(
                "MG1_form",
                "Blueprint has no sections.",
                restart_from=COMPOSER,
                fix="Define intro/verse/chorus/bridge/outro with bar counts.",
            )
        )
    else:
        bars = sum(s.bars for s in blueprint.sections)
        thin = [s.name for s in blueprint.sections if s.bars < 2]
        chorus = [s for s in blueprint.sections if re.search(r"chorus|hook|refrain", s.name, re.I)]
        if thin:
            results.append(
                _fail(
                    "MG1_form",
                    f"Sections shorter than 2 bars: {', '.join(thin)}.",
                    restart_from=COMPOSER,
                    fix="Give every section enough bars to state its idea.",
                )
            )
        elif not chorus:
            results.append(
                _fail(
                    "MG1_form",
                    "No chorus/hook section in the form.",
                    restart_from=COMPOSER,
                    fix="Add a chorus; a song without a hook cannot be remembered.",
                )
            )
        elif bars < 32:
            results.append(
                _fail(
                    "MG1_form",
                    f"Total form is only {bars} bars.",
                    restart_from=COMPOSER,
                    fix="Expand to at least 48 bars for a complete song.",
                    severity="major",
                )
            )
        else:
            results.append(_ok("MG1_form"))

    # MG2 - harmony must be functional and complete.
    sections_with_chords = [s for s in blueprint.sections if s.chords]
    if not sections_with_chords:
        results.append(
            _fail(
                "MG2_harmony",
                "No chord data anywhere in the blueprint.",
                restart_from=COMPOSER,
                fix="Write a chord progression per section.",
            )
        )
    else:
        missing_fn = [
            s.name for s in sections_with_chords if any(not c.function for c in s.chords)
        ]
        empty_sections = [s.name for s in blueprint.sections if not s.chords]
        if missing_fn:
            results.append(
                _fail(
                    "MG2_harmony",
                    f"Chords without roman-numeral functions in: {', '.join(missing_fn)}.",
                    restart_from=COMPOSER,
                    fix="Give every chord its scale-degree function.",
                    severity="major",
                )
            )
        elif len(empty_sections) > 1:
            results.append(
                _fail(
                    "MG2_harmony",
                    f"Sections with no harmony at all: {', '.join(empty_sections)}.",
                    restart_from=COMPOSER,
                    fix="Even a drum break should state its harmonic context.",
                    severity="major",
                )
            )
        else:
            results.append(_ok("MG2_harmony"))

    # MG3 - the energy curve must actually move.
    energies = [s.energy for s in blueprint.sections]
    if len(energies) >= 3 and (max(energies) - min(energies)) < 2:
        results.append(
            _fail(
                "MG3_energy_contrast",
                f"Energy barely moves across the song (range {min(energies)}-{max(energies)}/10).",
                restart_from=COMPOSER,
                fix="Plan real contrast: intimate verse, full chorus, stripped bridge.",
                severity="major",
            )
        )
    else:
        results.append(_ok("MG3_energy_contrast"))

    # MG4 - tempo inside a musical window and consistent with the brief.
    tempo = blueprint.tempo_bpm
    if not tempo or not 40 <= tempo <= 220:
        results.append(
            _fail(
                "MG4_tempo",
                f"Tempo {tempo} BPM is outside 40-220.",
                restart_from=COMPOSER,
                fix="Choose a defensible tempo for this genre.",
            )
        )
    elif brief is not None and brief.tempo_bpm and abs(tempo - brief.tempo_bpm) > 3:
        results.append(
            _fail(
                "MG4_tempo",
                f"Blueprint tempo {tempo:.0f} BPM ignores the brief's {brief.tempo_bpm:.0f} BPM.",
                restart_from=INTAKE,
                fix="Either follow the brief's tempo or confirm the change with the user.",
            )
        )
    else:
        results.append(_ok("MG4_tempo"))

    # MG5 - duration in a usable range.
    seconds = blueprint.estimated_seconds
    if seconds and not 60 <= seconds <= 480:
        results.append(
            _fail(
                "MG5_duration",
                f"Song length {seconds:.0f}s is outside the 60-480s window.",
                restart_from=COMPOSER,
                fix="Rebalance bar counts so the song lands in a releasable length.",
                severity="major",
            )
        )
    else:
        results.append(_ok("MG5_duration"))

    # MG6 - hook and motif must be described, not implied.
    if not blueprint.hook_definition.strip():
        results.append(
            _fail(
                "MG6_hook_and_motif",
                "No hook_definition.",
                restart_from=COMPOSER,
                fix="Name the exact bar, interval and lyric that becomes the hook.",
            )
        )
    elif len(blueprint.melodic_motif.split()) < 3:
        results.append(
            _fail(
                "MG6_hook_and_motif",
                "Melodic motif is not described in playable terms.",
                restart_from=COMPOSER,
                fix="Give the motif as scale degrees plus its rhythm.",
                severity="major",
            )
        )
    else:
        results.append(_ok("MG6_hook_and_motif"))

    # MG7 - user constraints in the music layer.
    if brief is not None and brief.must_avoid:
        blob = " ".join(
            [blueprint.arrangement_notes, " ".join(blueprint.instrumentation)]
            + [f"{s.texture} {s.drums} {s.bass}" for s in blueprint.sections]
        )
        violated = [
            item
            for item in brief.must_avoid
            if item.strip() and item.strip().lower() in blob.lower()
        ]
        if violated:
            results.append(
                _fail(
                    "MG7_user_constraints",
                    f"Arrangement uses forbidden elements: {', '.join(violated)}.",
                    restart_from=COMPOSER,
                    fix="Remove them from every section, not just the chorus.",
                )
            )
        else:
            results.append(_ok("MG7_user_constraints"))
    else:
        results.append(_ok("MG7_user_constraints"))

    # MG8 - the arrangement must actually line the lyrics up.
    if sheet is not None and not sheet.instrumental:
        sung_lines = [l.text for s in sheet.sections for l in s.lines if l.text.strip()]
        mapped_lines = {m.line.strip().lower() for m in arrangement.mappings if m.line.strip()}
        if not mapped_lines:
            results.append(
                _fail(
                    "MG8_alignment",
                    "No lyric lines were mapped onto the grid.",
                    restart_from=ARRANGER,
                    fix="Map every line to a bar, beat position and melody shape.",
                )
            )
        else:
            missing = [line for line in sung_lines if line.lower() not in mapped_lines]
            if len(missing) > max(1, int(len(sung_lines) * 0.25)):
                results.append(
                    _fail(
                        "MG8_alignment",
                        f"{len(missing)} of {len(sung_lines)} lyric lines have no place on the grid.",
                        restart_from=ARRANGER,
                        fix="Every sung line needs a bar/beat position and melody contour.",
                    )
                )
            else:
                results.append(_ok("MG8_alignment"))
        stray = [
            m.line
            for m in arrangement.mappings
            if m.line.strip() and m.line.strip().lower() not in {l.lower() for l in sung_lines}
        ]
        if len(stray) > 3:
            results.append(
                _fail(
                    "MG8_alignment",
                    f"{len(stray)} mapped lines do not exist in the lyric sheet.",
                    restart_from=ARRANGER,
                    fix="Map only lines that are actually in the lyrics.",
                    severity="major",
                )
            )
    else:
        results.append(_ok("MG8_alignment"))

    return results


def run_prompt_gates(
    prompt: SunoPrompt,
    brief: CreativeBrief | None,
    sheet: LyricSheet | None,
    reference_lossless: str = "unknown",
) -> list[GateResult]:
    """Hard checks on the final Suno prompt, mirroring Suno's own validators."""
    results: list[GateResult] = []

    # PG1 - character limits. Truncation is silent in Suno, so this is fatal.
    overflows: list[str] = []
    if len(prompt.title) > LIMITS["title_chars"]:
        overflows.append(f"title {len(prompt.title)}/{LIMITS['title_chars']}")
    if len(prompt.style) > LIMITS["style_chars"]:
        overflows.append(f"style {len(prompt.style)}/{LIMITS['style_chars']}")
    if len(prompt.lyrics) > LIMITS["lyrics_chars"]:
        overflows.append(f"lyrics {len(prompt.lyrics)}/{LIMITS['lyrics_chars']}")
    if len(prompt.exclude_style_text) > LIMITS["exclude_chars"]:
        overflows.append(
            f"exclude {len(prompt.exclude_style_text)}/{LIMITS['exclude_chars']}"
        )
    if overflows:
        results.append(
            _fail(
                "PG1_limits",
                "Character limits exceeded: " + "; ".join(overflows) + ".",
                restart_from=PROMPT_ENGINEER,
                fix="Tighten the field; Suno truncates silently past the limit.",
            )
        )
    else:
        results.append(_ok("PG1_limits"))

    # PG2 - style field hygiene: no tags, no prose instructions, no mastering claims.
    hygiene: list[str] = []
    if T.has_brackets(prompt.style):
        hygiene.append("bracketed tags in the style field")
    promises = T.unsupported_promises(prompt.style)
    if promises:
        hygiene.append(f"technical mastering claims ({', '.join(promises)})")
    if len(T.split_descriptors(prompt.style)) < LIMITS["descriptors_min"]:
        hygiene.append("too few descriptors")
    if hygiene:
        results.append(
            _fail(
                "PG2_style_hygiene",
                "Style field problems: " + "; ".join(hygiene) + ".",
                restart_from=PROMPT_ENGINEER,
                fix="Descriptors only, comma separated, no brackets, no mastering talk.",
            )
        )
    else:
        results.append(_ok("PG2_style_hygiene"))

    # PG3 - structure tags in the lyrics field.
    tags = T.extract_section_tags(prompt.lyrics)
    if not tags:
        results.append(
            _fail(
                "PG3_lyric_tags",
                "Lyrics field has no [Section] tags.",
                restart_from=PROMPT_ENGINEER,
                fix="Tag every section on its own line, blank line between sections.",
            )
        )
    elif len(tags) < 4:
        results.append(
            _fail(
                "PG3_lyric_tags",
                f"Only {len(tags)} section tags found; Suno will improvise more of "
                "the form than you intend.",
                restart_from=PROMPT_ENGINEER,
                fix="Tag every section so Suno does not improvise the form.",
                severity="major",
                soft=True,
            )
        )
    else:
        results.append(_ok("PG3_lyric_tags"))

    # PG4 - descriptor count.
    #
    # The upper bound is a calibration band, not a hard limit, so it does not
    # block: plenty of working Suno prompts carry a long, well-ordered descriptor
    # list, and a reviewer that has read the whole prompt is better placed to judge
    # it than a count. Live evidence: three consecutive generations were refused
    # over 16-20 descriptors while the reviewer scored `style_clarity` 10/10 and
    # `descriptor_balance` 8/10 and wanted to ship.
    #
    # The lower bound *is* factual - a two-word style field reliably renders
    # generic - so that one stays hard.
    count = T.count_descriptors(prompt.style)
    if count > 12:
        results.append(
            _fail(
                "PG4_descriptor_count",
                f"{count} descriptors is a long list; check that none of them "
                "contradict each other and that the first 200 characters still "
                "carry the decisive ones.",
                restart_from="",
                fix="Consider trimming to the 5-8 that actually steer the render.",
                severity="minor",
                soft=True,
            )
        )
    elif count < LIMITS["descriptors_min"]:
        results.append(
            _fail(
                "PG4_descriptor_count",
                f"Only {count} descriptors; the render will come back generic.",
                restart_from=PROMPT_ENGINEER,
                fix="Add genre, mood, instrumentation, production and vocal descriptors.",
                severity="major",
            )
        )
    else:
        results.append(_ok("PG4_descriptor_count"))

    # PG5 - negation belongs in the exclude field, never the style field.
    style_negations = T.find_negations(prompt.style)
    if style_negations:
        results.append(
            _fail(
                "PG5_no_negation_in_style",
                f"Style field contains negations ({', '.join(style_negations[:3])}); "
                "Suno reads them unreliably.",
                restart_from=PROMPT_ENGINEER,
                fix="Move every exclusion to the exclude field.",
                severity="major",
            )
        )
    else:
        results.append(_ok("PG5_no_negation_in_style"))

    # PG6 - the two fields must not fight each other.
    conflicts: list[str] = []
    if sheet is not None and not sheet.instrumental and brief is not None:
        required: list[str] = []
        if brief.instrumentation:
            required.extend(brief.instrumentation)
        required.extend(brief.must_include)
        if required:
            conflicts.extend(T.negation_conflicts(prompt.style, required))
            conflicts.extend(
                [
                    f"Required element '{item}' is excluded"
                    for item in required
                    if any(item.lower() in ex.lower() for ex in prompt.exclude_styles)
                ]
            )
    conflicts.extend(T.contradicts_exclusions(prompt.style, prompt.exclude_styles))
    if brief is not None:
        forbidden_present = T.keyword_hits(
            prompt.style + " " + prompt.lyrics, brief.must_avoid
        )
        if forbidden_present:
            conflicts.append(
                f"User asked to avoid: {', '.join(forbidden_present)}"
            )
    if conflicts:
        results.append(
            _fail(
                "PG6_exclude_sanity",
                "Style/exclude conflict: " + "; ".join(conflicts[:3]) + ".",
                restart_from=PROMPT_ENGINEER,
                fix="Make the requirement and the exclusion agree, then reword.",
            )
        )
    else:
        results.append(_ok("PG6_exclude_sanity"))

    # PG7 - the first 200 characters carry the load.
    head = prompt.style[: LIMITS["style_critical_zone"]]
    coverage = T.descriptor_coverage(head)
    missing = [name for name, present in coverage.items() if not present]
    if "genre" in missing:
        results.append(
            _fail(
                "PG7_front_load",
                "The first 200 characters (the zone Suno weights hardest) contain no "
                "genre descriptor.",
                restart_from=PROMPT_ENGINEER,
                fix="Front-load genre, then mood, then lead instrument or voice.",
                severity="major",
            )
        )
    elif len(missing) >= 4:
        results.append(
            _fail(
                "PG7_front_load",
                f"Only {6 - len(missing)}/6 descriptor families in the critical zone.",
                restart_from="",
                fix="Front-load more of the decisive descriptors within 200 characters.",
                severity="minor",
                soft=True,
            )
        )
    else:
        results.append(_ok("PG7_front_load"))

    # PG8 - vocal gender must agree across style, lyrics and exclude.
    if sheet is not None and not sheet.instrumental:
        gender = sheet.vocal.gender
        exclude_blob = prompt.exclude_style_text.lower()
        if gender in {"male", "female"}:
            opposite = "female" if gender == "male" else "male"
            if re.search(rf"\b{gender}\b", exclude_blob):
                results.append(
                    _fail(
                        "PG8_vocal_consistency",
                        f"Exclude field removes '{gender}' while the vocal spec asks for it.",
                        restart_from=PROMPT_ENGINEER,
                        fix="Exclude only the opposite vocal gender.",
                    )
                )
            elif not re.search(rf"\b({gender}|{opposite})\b", exclude_blob):
                results.append(
                    _fail(
                        "PG8_vocal_consistency",
                        "Exclude field does not lock the vocal gender, so Suno may flip it.",
                        restart_from=PROMPT_ENGINEER,
                        fix=f"Add '{opposite} vocals' to the exclude field.",
                        severity="major",
                    )
                )
            else:
                results.append(_ok("PG8_vocal_consistency"))
        elif gender == "mixed":
            if not re.search(r"\b(male|female)\b", prompt.style.lower()):
                results.append(
                    _fail(
                        "PG8_vocal_consistency",
                        "Mixed/duet vocal was specified but the style field never says "
                        "male or female.",
                        restart_from=PROMPT_ENGINEER,
                        fix="Name both voices explicitly in the style field.",
                        severity="major",
                    )
                )
            else:
                results.append(_ok("PG8_vocal_consistency"))
        else:
            results.append(_ok("PG8_vocal_consistency"))
    else:
        results.append(_ok("PG8_vocal_consistency"))

    # PG9 - do not promise clarity the reference cannot support.
    if reference_lossless in {"lossy", "lossy-transcoded"}:
        promises = [
            phrase
            for phrase in ("airy", "crisp hi-hat", "hi-fi", "hifi", "sparkling highs", "crystal clear")
            if phrase in prompt.style.lower()
        ]
        if promises and not prompt.provider_notes:
            results.append(
                _fail(
                    "PG9_lossless_honesty",
                    f"Style promises {', '.join(promises)} but the reference is {reference_lossless}.",
                    restart_from=AUDIO_ANALYST,
                    fix="Either drop the promise or document the risk for the user.",
                    severity="major",
                )
            )
        else:
            results.append(_ok("PG9_lossless_honesty"))
    else:
        results.append(_ok("PG9_lossless_honesty"))

    # PG10 - instrumental prompts must exclude vocals outright.
    if (sheet is not None and sheet.instrumental) or (brief is not None and brief.instrumental):
        exclude_blob = prompt.exclude_style_text.lower()
        if not re.search(r"\b(vocal|vocals|voice|singing)\b", exclude_blob):
            results.append(
                _fail(
                    "PG10_instrumental_consistency",
                    "Instrumental request but the exclude field does not forbid vocals.",
                    restart_from=PROMPT_ENGINEER,
                    fix="Add 'vocals, lead vocals, singing' to the exclude field.",
                )
            )
        else:
            results.append(_ok("PG10_instrumental_consistency"))
    else:
        results.append(_ok("PG10_instrumental_consistency"))

    return results


def summarise(results: Sequence[GateResult]) -> tuple[bool, list[GateResult]]:
    """Return ``(all_ok, failures)``."""
    failures = [r for r in results if not r.ok]
    return (not failures, failures)


def gate_report(results: Sequence[GateResult]) -> str:
    """One-line-per-gate report for logs and the UI."""
    lines = []
    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        lines.append(f"[{mark}] {result.code}" + (f" - {result.message}" if result.message else ""))
    return "\n".join(lines)


__all__ = [
    "GateResult",
    "PASS_MARKS",
    "pass_criteria",
    "ACCEPTANCE_SCORES",
    "ACCEPTANCE_MIN_ATTEMPT",
    "DIMENSION_FLOOR",
    "is_soft",
    "LYRIC_GATES",
    "MUSIC_GATES",
    "PROMPT_GATES",
    "run_lyric_gates",
    "run_music_gates",
    "run_prompt_gates",
    "summarise",
    "gate_report",
]

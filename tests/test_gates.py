"""Hard gates: the deterministic half of every review.

Each gate gets a payload that must pass and a payload that must fail, because a
gate that never fires is dead weight and a gate that always fires is a false
blocker that destroys trust in the whole loop.
"""

from __future__ import annotations

import pytest

from suno_agents.domain.models import (
    Arrangement,
    CreativeBrief,
    LineMapping,
    LyricLine,
    LyricSection,
    LyricSheet,
    MusicBlueprint,
    SectionPlan,
    SunoPrompt,
    VocalSpec,
)
from suno_agents.quality.gates import (
    run_lyric_gates,
    run_music_gates,
    run_prompt_gates,
    summarise,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def good_brief(**overrides) -> CreativeBrief:
    data = {
        "working_title": "Tikungan yang Salah",
        "concept": "seseorang menunggu di tempat yang sama setelah perpisahan",
        "genre": ["indie folk"],
        "mood": ["melancholic"],
        "language": "id",
        "vocal_gender": "female",
        "instrumentation": ["acoustic guitar", "brushed drums"],
        "must_include": [],
        "must_avoid": ["EDM drop"],
        "tempo_bpm": 92,
    }
    data.update(overrides)
    return CreativeBrief.from_dict(data)


def good_lyrics(**overrides) -> LyricSheet:
    sections = [
        LyricSection(
            name="Verse 1",
            lines=[
                LyricLine(text="kopi dingin di meja kursi yang kamu tinggali", syllables=13),
                LyricLine(text="hujan sudah berhenti aku masih di sini", syllables=12),
                LyricLine(text="radio tua menyala lagu yang tidak selesai", syllables=13),
                LyricLine(text="aku hafal setiap jeda yang kamu tinggalkan", syllables=13),
            ],
            performance_cue="close-miked",
        ),
        LyricSection(
            name="Chorus",
            lines=[
                LyricLine(text="aku tidak minta kembali aku cuma ingin tahu", syllables=13),
                LyricLine(text="apakah kamu juga menyimpan nama ini", syllables=12),
                LyricLine(text="biar waktu yang menjawab pelan pelan", syllables=11),
                LyricLine(text="kalau kamu dengar lagu ini itu aku", syllables=12),
            ],
        ),
        LyricSection(
            name="Bridge",
            lines=[
                LyricLine(text="kalau memang sudah selesai bilang saja", syllables=12),
                LyricLine(text="jangan tinggalkan aku di tengah kalimat", syllables=12),
            ],
        ),
    ]
    data = {
        "title_options": ["Tikungan yang Salah", "Kopi Dingin"],
        "working_title": "Tikungan yang Salah",
        "language": "id",
        "sections": sections,
        "vocal": VocalSpec(
            present=True,
            gender="female",
            voice_type="husky female alto",
            register="chest in verses, mixed on the chorus",
            timbre="smoky",
            delivery="close-miked, behind the beat",
        ),
        "hook_line": "kalau kamu dengar lagu ini itu aku",
        "emotional_arc": "settled grief to acceptance",
    }
    data.update(overrides)
    return LyricSheet.from_dict(data)


def good_blueprint(**overrides) -> MusicBlueprint:
    # 36 bars at 92 BPM is roughly 94 seconds: a complete short song rather than
    # the fragment a minimal fixture would otherwise produce.
    sections = [
        SectionPlan(
            name="Verse 1",
            bars=8,
            energy=5,
            drums="brushed snare on 2 and 4",
            bass="root on 1",
            melodic_device="motif 5-6-5-3",
            purpose="establish the situation",
        ),
        SectionPlan(name="Pre-Chorus", bars=4, energy=6, purpose="raise tension"),
        SectionPlan(name="Chorus", bars=8, energy=9, purpose="state the thesis"),
        SectionPlan(name="Verse 2", bars=8, energy=5, purpose="complicate the story"),
        SectionPlan(name="Bridge", bars=4, energy=4, purpose="change the angle"),
        SectionPlan(name="Outro", bars=4, energy=3, purpose="fade"),
    ]
    from suno_agents.domain.models import ChordEvent

    for section in sections:
        section.chords = [
            ChordEvent(bar=1, beats=4, symbol="C", function="I"),
            ChordEvent(bar=3, beats=4, symbol="G", function="V"),
        ]
    data = {
        "key": "C",
        "mode": "major",
        "tempo_bpm": 92,
        "time_signature": "4/4",
        "groove": "fingerpicked guitar in 8ths, brushed snare on 2 and 4",
        "harmonic_language": "diatonic major",
        "chord_progression_summary": "I-V-vi-IV",
        "sections": sections,
        "total_bars": 36,
        "estimated_seconds": 93.9,
        "melodic_motif": "scale degrees 5-6-5-3 (dotted 8th, 16th, 8th, quarter)",
        "hook_definition": "bar 5 of the chorus, inverted motif, drums drop out",
        "tension_curve": "3-5-9-4-9-3",
        "instrumentation": ["acoustic guitar", "brushed drums"],
        "arrangement_notes": "no synth, no auto-tune",
    }
    data.update(overrides)
    return MusicBlueprint.from_dict(data)


def good_arrangement(sheet: LyricSheet | None = None) -> Arrangement:
    sheet = sheet or good_lyrics()
    mappings = []
    for index, section in enumerate(sheet.sections):
        for line_index, line in enumerate(section.lines):
            mappings.append(
                LineMapping(
                    section=section.name,
                    line=line.text,
                    start_bar=1 + index * 8 + line_index * 2,
                    start_beat=1.0,
                    bars=2.0,
                    syllables=line.syllables,
                    syllables_per_beat=1.5,
                    contour="arch",
                    breath_mark="end of line",
                )
            )
    return Arrangement(
        song_form="V1-PC-C-V2-B-O",
        total_bars=36,
        total_seconds=93.9,
        vocal_entry_bar=1,
        hook_bar=9,
        mappings=mappings,
        prosody_notes="two bars per line at 92 BPM",
        syllable_report="12-13 syllables per line",
        stress_notes="stress on strong beats",
        breath_plan="full bar of breath at each couplet end",
        dynamic_plan="verse -20 LUFS, chorus -14",
        transition_plan="last bar of each section drops the drums",
        hook_placement="bar 9, drums stop for one beat",
        instrumental_breaks=["bars 1-4 guitar only"],
        conflicts=[],
        resolved_by=[],
    )


def good_prompt(**overrides) -> SunoPrompt:
    style = (
        "indie folk ballad, Indonesian, warm intimate, 92 BPM, sparse fingerpicked "
        "acoustic guitar, brushed drums, husky female alto with conversational delivery, "
        "tape saturation, analog warmth"
    )
    lyrics = (
        "[Intro]\n\n[Verse 1]\naku hafal setiap jeda yang kamu tinggalkan\n\n"
        "[Chorus]\nkalau kamu dengar lagu ini itu aku\n\n"
        "[Verse 2]\nradio tua menyala lagu yang tidak selesai\n\n"
        "[Outro]"
    )
    data = {
        "title": "Tikungan yang Salah",
        "style": style,
        "exclude_styles": ["male vocals", "male lead", "heavy drums", "edm drop"],
        "lyrics": lyrics,
        "vocal_gender_tag": "female",
        "lossless_checklist": ["Render as WAV", "Keep -1 dBTP", "High-pass vocal at 90 Hz", "Check 200-400 Hz"],
        "model_target": "v5",
    }
    data.update(overrides)
    return SunoPrompt.from_dict(data)


# --------------------------------------------------------------------------- #
# lyric gates
# --------------------------------------------------------------------------- #


def test_clean_lyrics_pass_every_gate():
    ok, failures = summarise(run_lyric_gates(good_lyrics(), good_brief(), good_blueprint()))
    assert ok, [f"{f.code}: {f.message}" for f in failures]


def test_instrumental_brief_with_sung_lines_fails_gate_three():
    brief = good_brief(instrumental=True, vocal_gender="instrumental")
    ok, failures = summarise(run_lyric_gates(good_lyrics(), brief, good_blueprint()))
    assert not ok
    codes = {f.code for f in failures}
    assert "LG3_instrumental_consistency" in codes
    assert next(f for f in failures if f.code == "LG3_instrumental_consistency").restart_from == "lyricist"


def test_thin_lyric_sheet_fails_singability_gate():
    sheet = good_lyrics(
        sections=[LyricSection(name="Verse 1", lines=[LyricLine(text="aku menunggu", syllables=5)])]
    )
    ok, failures = summarise(run_lyric_gates(sheet, good_brief(), good_blueprint()))
    assert not ok
    assert "LG2_singability" in {f.code for f in failures}


def test_cliche_lyrics_fail_gate_four():
    sheet = good_lyrics(
        sections=[
            LyricSection(
                name="Chorus",
                lines=[
                    LyricLine(text="dancing in the dark with broken glass"),
                    LyricLine(text="fire and ice, endless rain"),
                    LyricLine(text="set me free, hold on tight"),
                    LyricLine(text="tears falling on empty streets"),
                ],
            )
        ]
    )
    ok, failures = summarise(run_lyric_gates(sheet, good_brief(), good_blueprint()))
    assert not ok
    assert "LG4_cliche" in {f.code for f in failures}


def test_chorus_that_only_repeats_itself_fails_hook_gate():
    repeated = LyricLine(text="kalau kamu dengar lagu ini itu aku", syllables=12)
    sheet = good_lyrics(
        sections=[
            LyricSection(name="Chorus", lines=[repeated, repeated, repeated, repeated]),
        ],
        hook_line="kalau kamu dengar lagu ini itu aku",
    )
    ok, failures = summarise(run_lyric_gates(sheet, good_brief(), good_blueprint()))
    assert not ok
    assert "LG5_hook" in {f.code for f in failures}


def test_missing_vocal_spec_fails_gate_six():
    sheet = good_lyrics(vocal={})
    ok, failures = summarise(run_lyric_gates(sheet, good_brief(), good_blueprint()))
    assert not ok
    assert "LG6_vocal_spec" in {f.code for f in failures}


def test_forbidden_word_in_lyrics_fails_constraint_gate():
    sheet = good_lyrics(
        sections=[
            LyricSection(
                name="Chorus",
                lines=[
                    LyricLine(text="kita terbang ke EDM drop malam ini", syllables=11),
                    *good_lyrics().sections[1].lines[1:],
                ],
            )
        ]
    )
    ok, failures = summarise(run_lyric_gates(sheet, good_brief(), good_blueprint()))
    assert not ok
    assert "LG7_user_constraints" in {f.code for f in failures}


def test_wrong_language_fails_gate_eight():
    sheet = good_lyrics(language="en")
    ok, failures = summarise(run_lyric_gates(sheet, good_brief(), good_blueprint()))
    assert not ok
    assert "LG8_language" in {f.code for f in failures}


def test_instrumental_brief_requires_direction():
    brief = good_brief(instrumental=True, vocal_gender="instrumental")
    sheet = LyricSheet(
        title_options=["Ruang Kosong"],
        language="id",
        instrumental=True,
        instrumental_direction="",
        sections=[LyricSection(name="Chorus", lines=[])],
        vocal=VocalSpec(present=False, gender="instrumental"),
    )
    ok, failures = summarise(run_lyric_gates(sheet, brief, good_blueprint()))
    assert not ok
    assert "LG3_instrumental_consistency" in {f.code for f in failures}

    sheet.instrumental_direction = "lead melodica carries the topline, no human syllables"
    ok, failures = summarise(run_lyric_gates(sheet, brief, good_blueprint()))
    assert "LG3_instrumental_consistency" not in {f.code for f in failures}


# --------------------------------------------------------------------------- #
# music gates
# --------------------------------------------------------------------------- #


def test_clean_song_passes_music_gates():
    sheet = good_lyrics()
    ok, failures = summarise(
        run_music_gates(good_blueprint(), good_arrangement(sheet), good_brief(), sheet)
    )
    assert ok, [f"{f.code}: {f.message}" for f in failures]


def test_missing_chorus_fails_form_gate():
    blueprint = good_blueprint(
        sections=[SectionPlan(name="Verse 1", bars=8, energy=5), SectionPlan(name="Outro", bars=4, energy=3)]
    )
    ok, failures = summarise(
        run_music_gates(blueprint, good_arrangement(), good_brief(), good_lyrics())
    )
    assert not ok
    assert "MG1_form" in {f.code for f in failures}


def test_chords_without_functions_fail_harmony_gate():
    blueprint = good_blueprint()
    for section in blueprint.sections:
        for chord in section.chords:
            chord.function = ""
    ok, failures = summarise(
        run_music_gates(blueprint, good_arrangement(), good_brief(), good_lyrics())
    )
    assert not ok
    assert "MG2_harmony" in {f.code for f in failures}


def test_flat_energy_curve_fails_contrast_gate():
    blueprint = good_blueprint()
    for section in blueprint.sections:
        section.energy = 7
    blueprint.sections[0].energy = 7
    ok, failures = summarise(
        run_music_gates(blueprint, good_arrangement(), good_brief(), good_lyrics())
    )
    assert not ok
    assert "MG3_energy_contrast" in {f.code for f in failures}


def test_tempo_drift_from_brief_rolls_back_to_intake():
    blueprint = good_blueprint(tempo_bpm=140)
    ok, failures = summarise(
        run_music_gates(blueprint, good_arrangement(), good_brief(), good_lyrics())
    )
    assert not ok
    gate = next(f for f in failures if f.code == "MG4_tempo")
    assert gate.restart_from == "intake"


def test_unmapped_lyric_lines_fail_alignment_gate():
    arrangement = good_arrangement()
    arrangement.mappings = arrangement.mappings[:2]
    ok, failures = summarise(
        run_music_gates(good_blueprint(), arrangement, good_brief(), good_lyrics())
    )
    assert not ok
    assert "MG8_alignment" in {f.code for f in failures}
    assert next(f for f in failures if f.code == "MG8_alignment").restart_from == "arranger"


def test_forbidden_instrument_in_arrangement_fails_constraint_gate():
    blueprint = good_blueprint(arrangement_notes="includes a huge EDM drop in the chorus")
    ok, failures = summarise(
        run_music_gates(blueprint, good_arrangement(), good_brief(), good_lyrics())
    )
    assert not ok
    assert "MG7_user_constraints" in {f.code for f in failures}


# --------------------------------------------------------------------------- #
# prompt gates
# --------------------------------------------------------------------------- #


def test_clean_prompt_passes_every_gate():
    ok, failures = summarise(
        run_prompt_gates(good_prompt(), good_brief(), good_lyrics(), "lossless")
    )
    assert ok, [f"{f.code}: {f.message}" for f in failures]


def test_style_over_limit_fails_limits_gate():
    prompt = good_prompt(style="indie folk ballad, " * 80)
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG1_limits" in {f.code for f in failures}


def test_bracket_tags_in_style_fail_hygiene_gate():
    prompt = good_prompt(style="indie folk [Chorus] ballad, warm, female vocal, guitar, 92 BPM")
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG2_style_hygiene" in {f.code for f in failures}


def test_mastering_claim_in_style_fails_hygiene_gate():
    prompt = good_prompt(
        style=(
            "indie folk ballad, warm, lossless 24-bit studio master, female alto vocal, "
            "acoustic guitar, brushed drums, 92 BPM"
        )
    )
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG2_style_hygiene" in {f.code for f in failures}


def test_untagged_lyrics_fail_format_gate():
    prompt = good_prompt(lyrics="aku menunggu di tikungan yang salah\nkau pergi tanpa suara")
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG3_lyric_tags" in {f.code for f in failures}


def test_negation_in_style_field_fails_gate_five():
    prompt = good_prompt(
        style=(
            "indie folk ballad, warm, no drums, female alto vocal, acoustic guitar, "
            "brushed textures, 92 BPM"
        )
    )
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG5_no_negation_in_style" in {f.code for f in failures}


def test_unlocked_vocal_gender_fails_exclusion_gate():
    prompt = good_prompt(exclude_styles=["heavy drums", "edm drop"])
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG8_vocal_consistency" in {f.code for f in failures}


def test_exclude_that_removes_the_requested_vocal_fails():
    prompt = good_prompt(exclude_styles=["female vocals", "heavy drums"])
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG8_vocal_consistency" in {f.code for f in failures}


def test_front_loading_genre_outside_critical_zone_fails_gate_seven():
    filler = ", ".join(["warm", "intimate", "tape saturation", "analog warmth"] * 8)
    prompt = good_prompt(
        style=f"{filler}, indie folk ballad, 92 BPM, husky female alto, acoustic guitar"
    )
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG7_front_load" in {f.code for f in failures}


def test_instrumental_prompt_without_vocal_exclusion_fails():
    brief = good_brief(instrumental=True, vocal_gender="instrumental")
    sheet = LyricSheet(
        title_options=["Ruang"],
        language="id",
        instrumental=True,
        instrumental_direction="melodica topline",
        sections=[LyricSection(name="Chorus", lines=[])],
        vocal=VocalSpec(present=False, gender="instrumental"),
    )
    prompt = good_prompt(lyrics="[Intro]\n\n[Theme]\nmelodica lead\n\n[Outro]", exclude_styles=["heavy drums"])
    ok, failures = summarise(run_prompt_gates(prompt, brief, sheet, "lossless"))
    assert not ok
    assert "PG10_instrumental_consistency" in {f.code for f in failures}


def test_airy_promise_over_lossy_reference_fails_honesty_gate():
    prompt = good_prompt(
        style=(
            "indie folk ballad, warm, airy highs, crisp hi-hat, female alto vocal, "
            "acoustic guitar, brushed drums, 92 BPM"
        )
    )
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossy-transcoded"))
    assert not ok
    assert "PG9_lossless_honesty" in {f.code for f in failures}


def test_descriptor_count_band_is_enforced():
    prompt = good_prompt(style="indie folk")
    ok, failures = summarise(run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless"))
    assert not ok
    assert "PG4_descriptor_count" in {f.code for f in failures}


@pytest.mark.parametrize("severity", ["blocker", "major", "minor", "polish"])
def test_gate_failures_carry_a_restart_target_and_severity(severity):
    from suno_agents.quality.gates import _fail

    gate = _fail("TEST", "message", restart_from="composer", severity=severity)
    issue = gate.to_issue()
    assert issue.restart_from == "composer"
    assert issue.severity == severity
    assert issue.code == "TEST"

# --------------------------------------------------------------------------- #
# must_include: literal text vs structural direction
#
# Found live: a brief asking for "the confession in the last bar of the final
# chorus" made LG7 unsatisfiable, so every attempt failed and the pipeline rolled
# back forever. No lyric can contain that phrase verbatim.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "requirement,checkable",
    [
        # Only an explicitly quoted fragment is text a lyric must contain.
        ('sertakan kalimat "aku suka kamu" dua kali', True),
        ("nyanyikan 'dari dulu' di akhir", True),
        ('sebut kata "candela" di chorus', True),
        # Everything below is a directive for the reviewers, not a string to find.
        ("authentic Caribbean Spanish idioms and cadence", False),
        ("empowering affirmation phrases", False),
        ("fast vocal intro within the first 4 bars", False),
        ("syncopated 108 BPM afrobeats rhythm", False),
        ("pengakuan di bar terakhir chorus terakhir", False),
        ("menyebut nama kota asal", False),
        ("sebut warna jaket", False),
    ],
)
def test_only_quoted_requirements_are_checkable_as_text(requirement, checkable):
    from suno_agents.quality.gates import literal_requirements

    assert bool(literal_requirements([requirement])) is checkable


def test_unquoted_directives_do_not_block_the_gate():
    """The exact live failure: a brief asked for "authentic Caribbean Spanish idioms
    and cadence" and "empowering affirmation phrases". Both were treated as literal
    requirements, so the gate demanded those sentences inside a lyric, failed forever
    with an identical message, and burned three full regenerations while the reviewer
    scored the same work 87/100 and asked to ship."""
    from suno_agents.quality.gates import run_lyric_gates

    brief = good_brief(
        must_include=[
            "fast vocal intro within the first 4 bars",
            "authentic Caribbean Spanish idioms and cadence",
            "empowering affirmation phrases",
            "syncopated 108 BPM afrobeats rhythm",
        ]
    )
    ok, failures = summarise(run_lyric_gates(good_lyrics(), brief, good_blueprint()))
    assert "LG7_user_constraints" not in {f.code for f in failures}
    assert ok is True, [f"{f.code}: {f.message}" for f in failures]
    gate = next(
        g for g in run_lyric_gates(good_lyrics(), brief, good_blueprint())
        if g.code == "LG7_user_constraints"
    )
    assert gate.ok and "reviewers" in gate.message
    # And the directives are still named, so nothing is silently dropped.
    assert "empowering affirmation" in gate.message


def test_a_quoted_requirement_is_still_enforced():
    from suno_agents.quality.gates import run_lyric_gates

    brief = good_brief(must_include=['sertakan kalimat "tiga bulan aku simpan sendiri"'])
    ok, failures = summarise(run_lyric_gates(good_lyrics(), brief, good_blueprint()))
    assert "LG7_user_constraints" in {f.code for f in failures}


def test_a_quoted_requirement_passes_when_present():
    from suno_agents.quality.gates import run_lyric_gates

    brief = good_brief(must_include=['sebut "kopi dingin" di verse'])
    ok, failures = summarise(run_lyric_gates(good_lyrics(), brief, good_blueprint()))
    assert "LG7_user_constraints" not in {f.code for f in failures}


def test_partition_separates_quoted_from_directives():
    from suno_agents.quality.gates import partition_must_include

    literal, directive = partition_must_include(
        ['sertakan "aku suka kamu"', "sebut kopi dingin", "outro fades"]
    )
    assert literal == ['sertakan "aku suka kamu"']
    assert len(directive) == 2


def test_incomplete_vocal_spec_is_caught_by_the_artifact_validator_too():
    """The agent-level repair must see the same defects as the reviewer's gate.

    Found live: the model omitted voice_type/register/delivery twice, and only the
    reviewer gate noticed, so each catch cost a full generation plus a rollback.
    """
    sheet = good_lyrics(vocal={"present": True, "gender": "female"})
    assert sheet.missing_vocal_fields() == ["voice_type", "register", "delivery"]
    problems = sheet.validate()
    assert any("Vocal specification is incomplete" in p for p in problems)
    assert any("voice_type" in p for p in problems)

    complete = good_lyrics()
    assert complete.missing_vocal_fields() == []
    assert not any("Vocal specification" in p for p in complete.validate())

    # instrumental sheets have no vocal spec to complete
    instrumental = good_lyrics(
        instrumental=True, instrumental_direction="melodica topline", vocal={"present": False}
    )
    assert instrumental.missing_vocal_fields() == []


def test_thin_vocal_spec_is_a_soft_gate_not_a_blocker():
    """Missing refinements must not veto a reviewer that wants to ship.

    Found live: the lyric reviewer scored 83.4 against a pass mark of 80 and the
    gate refused it purely because voice_type/register/delivery were empty, so the
    run ping-ponged between lyricist and reviewer until the budget ran out. The
    gap is real and must cost score, but it refines fidelity rather than
    invalidating the prompt.
    """
    from suno_agents.quality.gates import run_lyric_gates

    sheet = good_lyrics(vocal={"present": True, "gender": "female"})
    gate = next(
        g for g in run_lyric_gates(sheet, good_brief(), good_blueprint())
        if g.code == "LG6_vocal_spec"
    )
    assert gate.soft is True, "the result itself must be marked soft"
    assert gate.restart_from == "", "a soft gate must never request a rollback"
    assert gate.severity == "minor"
    # Every refinement gap is still named, and the validator agrees on what is missing.
    for field in sheet.missing_vocal_fields():
        assert field in gate.message
    assert set(sheet.missing_vocal_fields()) == {"voice_type", "register", "delivery"}


def test_missing_vocal_gender_remains_blocking():
    """Suno flips the singer on gender, so this one is not negotiable."""
    from suno_agents.quality.gates import run_lyric_gates

    sheet = good_lyrics(vocal={"present": True})
    gate = next(
        g for g in run_lyric_gates(sheet, good_brief(), good_blueprint())
        if g.code == "LG6_vocal_spec"
    )
    assert gate.ok is False
    assert gate.restart_from == "lyricist"
    assert "gender" in gate.message.lower()


def test_several_soft_gate_failures_deduct_score_without_blocking():
    from suno_agents.quality.gates import GateResult
    from suno_agents.quality.verdict import finalise_review
    from suno_agents.domain.models import ReviewVerdict
    from suno_agents.domain.graph import LYRIC_REVIEWER

    verdict = ReviewVerdict.from_dict(
        {"stage": "lyric_reviewer", "verdict": "pass", "passed": True, "overall_score": 86,
         "dimensions": {k: 9.0 for k in ("originality", "imagery", "singability",
                                          "emotional_truth", "hook_strength",
                                          "language_quality", "vocal_fit", "brief_fidelity")},
         "issues": [], "restart_from": "", "reasoning": "Ships."}
    )
    gates = [GateResult("LG1_structure", True)]
    clean = finalise_review(LYRIC_REVIEWER, verdict, gates, pass_mark=80.0)
    soft = [GateResult("LG6_vocal_spec", False, "thin spec", severity="minor",
                       restart_from="", soft=True)]
    with_soft = finalise_review(LYRIC_REVIEWER, verdict, gates + soft, pass_mark=80.0)
    assert with_soft.passed is True
    assert with_soft.overall_score < clean.overall_score
    assert "Soft gates" in with_soft.reasoning

# --------------------------------------------------------------------------- #
# soft gates belong on the result, not on a global list of codes
#
# Found live: a global code list forced whole gates to be either blocking or not,
# so PG4's *calibration* complaint about descriptor count blocked three
# consecutive generations while the reviewer scored style_clarity 10/10 and asked
# to ship. One gate legitimately contains both a factual check and a taste call.
# --------------------------------------------------------------------------- #


def test_softness_is_per_result_not_per_code():
    from suno_agents.quality.gates import GateResult, is_soft

    assert is_soft(GateResult("PG4_descriptor_count", False)) is False
    assert is_soft(GateResult("PG4_descriptor_count", False, soft=True)) is True
    assert is_soft(GateResult("PG4_descriptor_count", True)) is False


def test_descriptor_count_upper_band_is_soft_but_the_lower_band_is_hard():
    """Too many descriptors is taste; too few is a measured failure to steer."""
    from suno_agents.quality.gates import run_prompt_gates

    many = good_prompt(style=", ".join(
        ["indie folk ballad", "Indonesian", "warm", "92 BPM", "female alto vocal",
         "acoustic guitar", "brushed drums", "tape saturation", "intimate", "dry vocal",
         "mono bass", "restrained dynamics", "close-miked", "analog warmth"])
    )
    gate = next(g for g in run_prompt_gates(many, good_brief(), good_lyrics(), "lossless")
                if g.code == "PG4_descriptor_count")
    assert gate.ok is False and gate.soft is True
    assert gate.restart_from == ""

    thin = good_prompt(style="indie folk, warm, female vocal")
    gate = next(g for g in run_prompt_gates(thin, good_brief(), good_lyrics(), "lossless")
                if g.code == "PG4_descriptor_count")
    assert gate.ok is False and gate.soft is False
    assert gate.restart_from == "prompt_engineer"


def test_a_long_descriptor_list_does_not_block_a_passing_review():
    """The exact live scenario: reviewer wants to ship, only PG4 objects."""
    from suno_agents.domain.models import ReviewVerdict
    from suno_agents.quality.gates import GateResult, run_prompt_gates
    from suno_agents.quality.verdict import finalise_review

    prompt = good_prompt(style=", ".join(
        ["indie folk ballad", "Indonesian", "warm", "92 BPM", "female alto vocal",
         "acoustic guitar", "brushed drums", "tape saturation", "intimate", "dry vocal",
         "mono bass", "restrained dynamics", "close-miked", "analog warmth"])
    )
    gates = run_prompt_gates(prompt, good_brief(), good_lyrics(), "lossless")
    verdict = ReviewVerdict.from_dict(
        {"stage": "prompt_reviewer", "verdict": "pass", "passed": True, "overall_score": 85.5,
         "dimensions": {k: 8.5 for k in ("prompt_fidelity", "style_clarity", "descriptor_balance",
                                          "lyrics_format", "exclusion_effectiveness",
                                          "structural_clarity", "vocal_direction", "title_hook",
                                          "render_risk", "lossless_readiness")},
         "issues": [], "restart_from": "", "reasoning": "Ship it."}
    )
    merged = finalise_review("prompt_reviewer", verdict, gates, pass_mark=85.0, attempt=1)
    assert merged.passed is True, merged.gates_failed
    # The weighted merge runs first (model 62% + gates 38%), so compare against the
    # same merge without the soft gate: the soft gap must cost score from that basis.
    no_soft = finalise_review("prompt_reviewer", verdict, [g for g in gates if g.ok],
                              pass_mark=85.0, attempt=1)
    assert merged.overall_score < no_soft.overall_score, "the soft gap must still cost score"
    assert merged.overall_score >= 85.0, "but it must not push a shippable prompt under the mark"
    assert "PG4_descriptor_count" in merged.reasoning
    assert "Soft gates" in merged.reasoning

# --------------------------------------------------------------------------- #
# the requirement must be stated, and stated once
#
# A user looking at a review panel could see the verdict but not the requirement,
# because the answer only existed in the code. Now it is a value the UI renders,
# and this pins it to the thresholds it describes.
# --------------------------------------------------------------------------- #


def test_pass_criteria_names_every_threshold_it_enforces():
    from suno_agents.quality.gates import (
        ACCEPTANCE_SCORES,
        DIMENSION_FLOOR,
        PASS_MARKS,
        pass_criteria,
    )

    for stage in PASS_MARKS:
        text = pass_criteria(stage)
        assert f">= {PASS_MARKS[stage]:.0f}/100" in text, stage
        assert str(int(DIMENSION_FLOOR)) in text, stage
        assert "gate keras" in text, stage
        assert "verdict" in text, stage
        if stage in ACCEPTANCE_SCORES:
            assert f"{ACCEPTANCE_SCORES[stage]:.0f}/100" in text, stage


def test_pass_criteria_says_which_gates_are_advisory():
    from suno_agents.quality.gates import pass_criteria

    text = pass_criteria("lyric_reviewer")
    assert "LG6_vocal_spec" in text
    # And a non-reviewer stage gets a clear "not applicable" rather than an empty line.
    assert "bukan tahap reviewer" in pass_criteria("composer")


def test_every_advisory_gate_code_is_reported_as_advisory():
    """The displayed list must match the gates that actually can be non-blocking."""
    from suno_agents.quality.gates import _SOFT_CODES, pass_criteria
    from suno_agents.quality import gates as gates_module

    referenced = {
        code
        for code in (name for name in dir(gates_module) if name.isupper())
        for _ in [0]
    }
    # Each listed code must exist as a gate the module raises somewhere.
    source = __import__("inspect").getsource(gates_module)
    for code in _SOFT_CODES:
        assert f'"{code}"' in source, code
    text = pass_criteria("prompt_reviewer")
    for code in _SOFT_CODES:
        assert code in text, code
    assert referenced is not None

"""The review loop: gates beat opinions, and rollbacks land where they should.

These tests drive the real orchestrator with scripted models, so the thing being
verified is the control flow itself: a failed gate must send work back even when
the model says "pass", a legal reviewer request must be honoured, an illegal one
must be clamped, and the revision budget must stop a runaway loop.
"""

from __future__ import annotations

import json

import pytest

from suno_agents.domain.graph import (
    ARRANGER,
    AUDIO_ANALYST,
    COMPOSER,
    INTAKE,
    LYRICIST,
    LYRIC_REVIEWER,
    MUSIC_REVIEWER,
    PROMPT_ENGINEER,
    PROMPT_REVIEWER,
)
from suno_agents.domain.models import PipelineState, ReviewIssue, ReviewVerdict
from suno_agents.llm.mock import MockRunner, json_responder, sequence
from suno_agents.llm.trace import TraceRecorder
from suno_agents.pipeline.orchestrator import Orchestrator
from suno_agents.quality.gates import GateResult
from suno_agents.quality.verdict import finalise_review, gate_score, revision_notes_from
from suno_agents.testing.demo import (
    DEMO_EXCLUDE_BAD,
    DEMO_EXCLUDE_GOOD,
    _arrangement_payload,
    _blueprint_payload,
    _intake_payload,
    _lyric_payload,
    _prompt_payload,
    _reference_payload,
    build_demo_runner,
)


# --------------------------------------------------------------------------- #
# verdict merging
# --------------------------------------------------------------------------- #


def _verdict(**overrides) -> ReviewVerdict:
    data = {
        "stage": "prompt_reviewer",
        "verdict": "pass",
        "passed": True,
        "overall_score": 95.0,
        "dimensions": {
            "prompt_fidelity": 9.5,
            "style_clarity": 9.5,
            "descriptor_balance": 9.5,
            "lyrics_format": 9.5,
            "exclusion_effectiveness": 9.5,
            "structural_clarity": 9.5,
            "vocal_direction": 9.5,
            "title_hook": 9.5,
            "render_risk": 9.5,
            "lossless_readiness": 9.5,
        },
        "issues": [],
        "restart_from": "",
        "reasoning": "Looks good.",
    }
    data.update(overrides)
    return ReviewVerdict.from_dict(data)


def test_failed_gate_overrides_a_passing_model_verdict():
    gates = [
        GateResult("PG1_limits", True),
        GateResult("PG8_vocal_consistency", False, "gender unlocked", restart_from=PROMPT_ENGINEER),
    ]
    verdict = finalise_review(PROMPT_REVIEWER, _verdict(), gates, pass_mark=85.0)
    assert verdict.passed is False
    assert verdict.verdict == "revise"
    assert verdict.restart_from == PROMPT_ENGINEER
    assert "PG8_vocal_consistency" in verdict.gates_failed
    assert "Hard gates failed" in verdict.reasoning


def test_clean_gates_and_strong_scores_pass():
    gates = [GateResult("PG1_limits", True), GateResult("PG2_style_hygiene", True)]
    verdict = finalise_review(PROMPT_REVIEWER, _verdict(), gates, pass_mark=85.0)
    assert verdict.passed is True
    assert verdict.restart_from == ""


def test_low_dimension_blocks_a_pass_even_with_clean_gates():
    verdict = finalise_review(
        PROMPT_REVIEWER,
        _verdict(dimensions={"exclusion_effectiveness": 3.0, "style_clarity": 9.0}),
        [GateResult("PG1_limits", True)],
        pass_mark=85.0,
    )
    assert verdict.passed is False
    assert any(issue.code == "LOW_EXCLUSION_EFFECTIVENESS" for issue in verdict.issues)


def test_score_below_pass_mark_fails_even_when_model_says_pass():
    verdict = finalise_review(
        PROMPT_REVIEWER,
        _verdict(overall_score=70.0),
        [GateResult("PG1_limits", True)],
        pass_mark=85.0,
    )
    assert verdict.passed is False
    assert verdict.overall_score < 85.0


def test_reject_requires_a_few_failed_attempts():
    gates = [GateResult("PG1_limits", False, "over", restart_from=PROMPT_ENGINEER, severity="blocker")]
    early = finalise_review(PROMPT_REVIEWER, _verdict(overall_score=10.0), gates, pass_mark=85.0, attempt=0)
    late = finalise_review(PROMPT_REVIEWER, _verdict(overall_score=10.0), gates, pass_mark=85.0, attempt=3)
    assert early.verdict == "revise"
    assert late.verdict == "reject"


def test_gate_score_weights_blockers_heaviest():
    soft = [GateResult("A", False, severity="polish"), GateResult("B", True)]
    hard = [GateResult("A", False, severity="blocker"), GateResult("B", True)]
    assert gate_score(hard) < gate_score(soft) < 100.0
    assert gate_score([GateResult("B", True)]) == 100.0


def test_rollback_target_uses_the_earliest_legal_root_cause():
    gates = [
        GateResult("MG8_alignment", False, "unmapped", restart_from=ARRANGER, severity="blocker"),
        GateResult("MG2_harmony", False, "no functions", restart_from=COMPOSER, severity="major"),
    ]
    verdict = finalise_review(MUSIC_REVIEWER, _verdict(verdict="revise", passed=False, overall_score=60.0), gates, pass_mark=80.0)
    # composer runs before arranger, so a composer-rooted defect wins.
    assert verdict.restart_from == COMPOSER


def test_illegal_restart_request_is_clamped_into_the_legal_set():
    llm = _verdict(verdict="revise", passed=False, overall_score=60.0, restart_from="totally_invented")
    gates = [GateResult("MG8_alignment", False, "unmapped", restart_from="", severity="major")]
    verdict = finalise_review(MUSIC_REVIEWER, llm, gates, pass_mark=80.0)
    from suno_agents.domain.graph import legal_restarts

    assert verdict.restart_from in legal_restarts(MUSIC_REVIEWER)


def test_revision_notes_carry_the_fix_list_and_strengths():
    gates = [GateResult("PG8_vocal_consistency", False, "gender unlocked", restart_from=PROMPT_ENGINEER)]
    verdict = finalise_review(
        PROMPT_REVIEWER,
        _verdict(verdict="revise", passed=False, overall_score=70.0, strengths=["tight imagery"]),
        gates,
        pass_mark=85.0,
    )
    notes = revision_notes_from(verdict, for_stage=PROMPT_ENGINEER)
    blob = "\n".join(notes)
    assert "PG8_vocal_consistency" in blob
    assert "gender unlocked" in blob
    assert "tight imagery" in blob


# --------------------------------------------------------------------------- #
# the loop, end to end
# --------------------------------------------------------------------------- #


def test_full_demo_run_rolls_back_twice_and_passes():
    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    result = orchestrator.run(None)
    assert result.passed is True
    assert result.state.revisions == 2
    assert result.prompt is not None
    assert result.state.prompt_review is not None and result.state.prompt_review.passed

    # The two scripted failures are deterministic gate failures, recorded in the
    # review log; the per-stage fields hold only the final (passing) verdict.
    failed_gates = [
        gate for verdict in result.state.review_log for gate in verdict.gates_failed
    ]
    assert "MG8_alignment" in failed_gates
    assert result.state.music_review is not None and result.state.music_review.passed
    assert result.state.prompt_review is not None and result.state.prompt_review.passed
    assert len(result.state.review_log) == 5  # 3 reviewers, 2 of them twice


def test_rollback_events_point_at_the_right_stages():
    events: list = []
    orchestrator = Orchestrator(build_demo_runner(), settings=None, on_event=events.append)
    orchestrator.run(None)
    rollbacks = [e for e in events if e.kind == "rollback"]
    assert [r.data["target"] for r in rollbacks] == [ARRANGER, PROMPT_ENGINEER]
    for event in rollbacks:
        assert event.data["gates_failed"]
        assert event.data["issues"]


def test_arranger_is_actually_rerun_after_the_rollback():
    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    orchestrator.run(None)
    stages = [record.stage for record in orchestrator.state.history]
    assert stages.count(ARRANGER) == 2
    assert stages.count(PROMPT_ENGINEER) == 2
    # and the loop advanced past the reviewers again in order
    assert stages.index(MUSIC_REVIEWER) < len(stages) - 1


def test_run_report_contains_the_prompt_and_the_trace():
    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    result = orchestrator.run(None)
    report = result.report
    assert "PROMPT SUNO" in report
    assert "STYLE OF MUSIC" in report
    assert "TITLE" in report
    assert "JEJAK STAGE" in report
    assert "REVIEW PROMPT" in report


def test_output_files_are_complete_and_parseable():
    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    orchestrator.run(None)
    files = orchestrator.output_files()
    for required in (
        "report.md",
        "state.json",
        "title.txt",
        "style.txt",
        "exclude.txt",
        "lyrics.txt",
        "suno_prompt.txt",
        "lossless_checklist.md",
        "lyrics_full.md",
        "music_blueprint.md",
        "arrangement.md",
        "brief.md",
    ):
        assert required in files, required
    state = json.loads(files["state.json"])
    assert state["prompt"]["title"]
    assert state["success"] is True
    assert files["exclude.txt"]
    # the prompt must be re-loadable, since runs are resumable
    from suno_agents.domain.models import PipelineState

    reloaded = PipelineState.from_dict(state)
    assert reloaded.prompt is not None
    assert reloaded.prompt.style == orchestrator.state.prompt.style


def test_revision_budget_stops_a_runaway_loop():
    runner = MockRunner(_always_failing_script())
    orchestrator = Orchestrator(runner, settings=_settings(max_revisions=2))
    result = orchestrator.run(None)
    assert result.passed is False
    assert result.state.revisions <= 2
    assert "Batas revisi" in result.stopped_reason or "tidak ada target mundur" in result.stopped_reason


def test_stop_request_ends_the_loop_before_the_next_stage():
    orchestrator = Orchestrator(build_demo_runner(), settings=None)

    def stopper(event) -> None:
        if event.kind == "stage_done" and event.stage == COMPOSER:
            orchestrator.stop()

    orchestrator.on_event = stopper
    result = orchestrator.run(None)
    assert "Dihentikan" in result.stopped_reason
    assert result.state.prompt is None


def test_loop_escalates_when_reviewer_gives_no_usable_target():
    """A reviewer that neither passes nor names a target must not spin forever.

    Without a usable request the loop escalates one legal stage per attempt, so
    the run terminates inside its revision budget instead of re-running the same
    stage until the heat death of the universe.
    """
    from suno_agents.domain.graph import legal_restarts

    events: list = []
    runner = _vague_music_reviewer_runner()
    orchestrator = Orchestrator(runner, settings=_settings(max_revisions=3), on_event=events.append)
    result = orchestrator.run(None)

    rollbacks = [e for e in events if e.kind == "rollback"]
    assert rollbacks, "the vague reviewer should still have triggered rollbacks"
    assert len(rollbacks) <= 3
    targets = [e.data["target"] for e in rollbacks]
    for target in targets:
        assert target in legal_restarts(MUSIC_REVIEWER)
    # Escalation must move upstream over successive failures, not repeat one target.
    assert len(set(targets)) > 1
    assert result.state.revisions <= orchestrator.max_revisions


# --------------------------------------------------------------------------- #
# scripts
# --------------------------------------------------------------------------- #


def _reference_stage() -> list:
    return [json_responder(_reference_payload())]


def _settings(max_revisions: int):
    from suno_agents.config.models import AppSettings

    settings = AppSettings()
    settings.ensure_defaults()
    settings.max_revisions = max_revisions
    settings.stage_attempts = 1
    return settings


def _always_failing_script():
    """Every stage works; the music reviewer never accepts anything.

    Fed to a flat :class:`MockRunner` deliberately: it proves the loop is bounded
    even when a single cursor keeps answering every stage.
    """
    bad_music_review = {
        "stage": MUSIC_REVIEWER,
        "verdict": "revise",
        "passed": False,
        "overall_score": 40.0,
        "dimensions": {k: 4.0 for k in (
            "structure", "harmony", "melody_contour", "groove_and_rhythm", "energy_arc",
            "hook_strength", "lyrics_integration", "character_and_identity",
            "producibility", "lossless_readiness",
        )},
        "issues": [
            {
                "code": "BAD",
                "severity": "blocker",
                "dimension": "structure",
                "message": "The form does not work.",
                "fix": "Rewrite the form.",
                "restart_from": COMPOSER,
            }
        ],
        "restart_from": COMPOSER,
        "reasoning": "Not a song yet.",
        "strengths": [],
    }
    return [
        json_responder(_intake_payload()),
        json_responder(_reference_payload()),
        json_responder(_blueprint_payload()),
        json_responder(_lyric_payload()),
        json_responder(
            {
                "stage": LYRIC_REVIEWER,
                "verdict": "pass",
                "passed": True,
                "overall_score": 90.0,
                "dimensions": {
                    "originality": 9.0, "imagery": 9.0, "singability": 9.0,
                    "emotional_truth": 9.0, "hook_strength": 9.0, "language_quality": 9.0,
                    "vocal_fit": 9.0, "brief_fidelity": 9.0,
                },
                "issues": [],
                "restart_from": "",
                "reasoning": "Good.",
                "strengths": [],
            }
        ),
        json_responder(_arrangement_payload(True)),
        json_responder(bad_music_review),
        json_responder(_prompt_payload(DEMO_EXCLUDE_GOOD)),
        json_responder(
            {
                "stage": PROMPT_REVIEWER,
                "verdict": "pass",
                "passed": True,
                "overall_score": 92.0,
                "dimensions": {
                    "prompt_fidelity": 9.0, "style_clarity": 9.0, "descriptor_balance": 9.0,
                    "lyrics_format": 9.0, "exclusion_effectiveness": 9.0,
                    "structural_clarity": 9.0, "vocal_direction": 9.0, "title_hook": 9.0,
                    "render_risk": 9.0, "lossless_readiness": 9.0,
                },
                "issues": [],
                "restart_from": "",
                "reasoning": "Good.",
                "strengths": [],
            }
        ),
    ]


class _StageRoutedRunner:
    """Hands each pipeline stage its own script, like the app's DemoRunner does."""

    def __init__(self, scripts: dict[str, list]) -> None:
        self.model = "routed-script"
        self._scripts = scripts
        self._runners: dict[str, MockRunner] = {}

    def for_stage(self, stage: str) -> MockRunner:
        runner = self._runners.get(stage)
        if runner is None:
            runner = MockRunner(self._scripts.get(stage, ["{}"]), model=f"routed:{stage}")
            self._runners[stage] = runner
        return runner

    def complete(self, *args, **kwargs):
        return self.for_stage(COMPOSER).complete(*args, **kwargs)

    def complete_json(self, *args, **kwargs):
        return self.for_stage(COMPOSER).complete_json(*args, **kwargs)


def _vague_music_reviewer_runner():
    """Every stage behaves; the music reviewer perpetually fails with no target.

    The reviewer is *stage-routed* rather than fed from a flat script, because a
    flat script would drift onto the next stage's responses and the test would
    then be measuring the mock instead of the loop.
    """
    vague = {
        "stage": MUSIC_REVIEWER,
        "verdict": "revise",
        "passed": False,
        "overall_score": 55.0,
        "dimensions": {k: 5.0 for k in (
            "structure", "harmony", "melody_contour", "groove_and_rhythm", "energy_arc",
            "hook_strength", "lyrics_integration", "character_and_identity",
            "producibility", "lossless_readiness",
        )},
        "issues": [],
        "restart_from": "",
        "reasoning": "Vague dissatisfaction.",
        "strengths": [],
    }
    return _StageRoutedRunner(
        {
            INTAKE: [json_responder(_intake_payload())],
            AUDIO_ANALYST: [json_responder(_reference_payload())],
            COMPOSER: [json_responder(_blueprint_payload())],
            LYRICIST: [json_responder(_lyric_payload())],
            LYRIC_REVIEWER: [_passing_lyric_review()],
            ARRANGER: [json_responder(_arrangement_payload(True))],
            MUSIC_REVIEWER: [json_responder(vague)],
            PROMPT_ENGINEER: [json_responder(_prompt_payload(DEMO_EXCLUDE_GOOD))],
            PROMPT_REVIEWER: [_passing_prompt_review()],
        }
    )


def _passing_lyric_review():
    return json_responder(
        {
            "stage": LYRIC_REVIEWER,
            "verdict": "pass",
            "passed": True,
            "overall_score": 90.0,
            "dimensions": {
                "originality": 9.0, "imagery": 9.0, "singability": 9.0,
                "emotional_truth": 9.0, "hook_strength": 9.0, "language_quality": 9.0,
                "vocal_fit": 9.0, "brief_fidelity": 9.0,
            },
            "issues": [],
            "restart_from": "",
            "reasoning": "Good.",
            "strengths": [],
        }
    )


def _passing_prompt_review():
    return json_responder(
        {
            "stage": PROMPT_REVIEWER,
            "verdict": "pass",
            "passed": True,
            "overall_score": 92.0,
            "dimensions": {
                "prompt_fidelity": 9.0, "style_clarity": 9.0, "descriptor_balance": 9.0,
                "lyrics_format": 9.0, "exclusion_effectiveness": 9.0,
                "structural_clarity": 9.0, "vocal_direction": 9.0, "title_hook": 9.0,
                "render_risk": 9.0, "lossless_readiness": 9.0,
            },
            "issues": [],
            "restart_from": "",
            "reasoning": "Good.",
            "strengths": [],
        }
    )


@pytest.mark.parametrize(
    "stage",
    [LYRIC_REVIEWER, MUSIC_REVIEWER, PROMPT_REVIEWER],
)
def test_every_reviewer_stage_has_gates_and_a_pass_mark(stage):
    from suno_agents.quality.gates import PASS_MARKS

    assert stage in PASS_MARKS
    assert PASS_MARKS[stage] >= 80.0


def test_intake_stage_is_skipped_when_a_brief_is_supplied():
    from suno_agents.testing.demo import demo_brief

    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    orchestrator.run(demo_brief())
    assert all(record.stage != INTAKE for record in orchestrator.state.history)


def test_agent_failure_is_reported_instead_of_crashing():
    class ExplodingRunner:
        model = "boom"

        def complete_json(self, *_args, **_kwargs):
            raise RuntimeError("provider exploded")

    orchestrator = Orchestrator(ExplodingRunner(), settings=_settings(max_revisions=1))
    result = orchestrator.run(None)
    assert result.passed is False
    assert "gagal" in result.stopped_reason.lower()
    assert result.state.finished is True


def test_audio_caveats_reach_the_final_prompt():
    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    result = orchestrator.run(
        None, reference_lossless="lossy-transcoded", reference_quality=40
    )
    prompt = result.prompt
    assert prompt is not None
    assert "lossy-transcoded" in prompt.provider_notes
    assert any("WAV" in item for item in prompt.lossless_checklist)


def test_missing_reference_summary_is_stated_honestly():
    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    result = orchestrator.run(None)
    assert result.state.reference_summary
    assert "Tidak ada file referensi" in result.state.reference_summary

# --------------------------------------------------------------------------- #
# routing authority and tolerant parsing
#
# Both of these were found by running the pipeline against a real provider, and
# both were expensive: the first discarded the composer's work on a lyricist
# problem, the second faked eight dimension scores the model never sent.
# --------------------------------------------------------------------------- #


def test_one_contrarian_minor_issue_cannot_drag_the_rollback_upstream():
    """A reviewer's consolidated target beats a single dissenting issue."""
    raw = {
        "stage": "review",
        "verdict": "revise",
        "passed": False,
        "overall_score": 71,
        "restart_from": "lyricist",
        "dimensions": {},
        "issues": [
            {"severity": "major", "message": "line too dense", "restart_from": "lyricist"},
            {"severity": "major", "message": "idiom reused", "restart_from": "lyricist"},
            {"severity": "major", "message": "chair staging broke", "restart_from": "lyricist"},
            {"severity": "major", "message": "brief fields unconfirmed", "restart_from": "intake"},
        ],
        "reasoning": "Strong premise, not shippable yet.",
        "strengths": ["object work"],
    }
    verdict = ReviewVerdict.from_dict(raw, stage=LYRIC_REVIEWER)
    gates = [GateResult(code, True) for code in
             ("LG1_structure", "LG2_singability", "LG4_cliche", "LG5_hook")]
    merged = finalise_review(LYRIC_REVIEWER, verdict, gates, pass_mark=80.0, attempt=1)
    assert merged.restart_from == LYRICIST, (
        "the reviewer named lyricist; one advisory issue must not escalate to intake"
    )
    assert merged.passed is False


def test_a_measured_gate_failure_still_outranks_the_reviewers_own_target():
    """Gates are facts: a real upstream defect must win over the model's opinion."""
    raw = {
        "stage": "review",
        "verdict": "revise",
        "passed": False,
        "overall_score": 88,
        "restart_from": "arranger",
        "issues": [],
        "reasoning": "Alignment looks fine to me.",
    }
    verdict = ReviewVerdict.from_dict(raw, stage=MUSIC_REVIEWER)
    gates = [
        GateResult("MG1_form", True),
        GateResult("MG2_harmony", False, "no functions", restart_from=COMPOSER, severity="blocker"),
    ]
    merged = finalise_review(MUSIC_REVIEWER, verdict, gates, pass_mark=80.0, attempt=0)
    assert merged.restart_from == COMPOSER
    assert "MG2_harmony" in merged.gates_failed


def test_blockers_may_escalate_one_level_past_the_consolidated_target():
    raw = {
        "stage": "review",
        "verdict": "revise",
        "passed": False,
        "overall_score": 70,
        "restart_from": "",  # the model forgot to consolidate
        "issues": [
            {"severity": "blocker", "message": "form cannot work", "restart_from": "composer"},
            {"severity": "minor", "message": "nit", "restart_from": "intake"},
        ],
        "reasoning": "Form is wrong at the root.",
    }
    verdict = ReviewVerdict.from_dict(raw, stage=MUSIC_REVIEWER)
    merged = finalise_review(MUSIC_REVIEWER, verdict, [], pass_mark=80.0, attempt=0)
    assert merged.restart_from == COMPOSER


def test_missing_dimensions_do_not_manufacture_floor_failures():
    """A reviewer that only gives a headline score must not look like all zeros."""
    raw = {
        "stage": "lyric_reviewer",
        "verdict": "pass",
        "passed": True,
        "overall_score": 84,
        "restart_from": "",
        "issues": [],
        "reasoning": "Ships as is.",
        "strengths": ["tight imagery"],
    }
    verdict = ReviewVerdict.from_dict(raw, stage=LYRIC_REVIEWER)
    assert verdict.dimensions == {}
    merged = finalise_review(
        LYRIC_REVIEWER, verdict, [GateResult("LG1_structure", True)], pass_mark=80.0
    )
    assert not any(issue.code.startswith("LOW_") for issue in merged.issues), merged.issues
    assert merged.passed is True
    assert "overall only" in merged.reasoning


def test_dimension_parsing_accepts_the_shapes_models_actually_send():
    from suno_agents.domain.models import parse_dimensions

    assert parse_dimensions({"a": 8}) == {"a": 8.0}
    assert parse_dimensions({"a": {"score": 7.5}}) == {"a": 7.5}
    assert parse_dimensions([{"name": "x", "score": 9}]) == {"x": 9.0}
    assert parse_dimensions({"a": None, "b": "n/a"}) == {}
    assert parse_dimensions(None) == {}
    assert parse_dimensions("nonsense") == {}


def test_reviewer_stage_and_restart_fields_are_normalised():
    from suno_agents.domain.models import normalise_stage_name

    assert normalise_stage_name("Lyrics") == LYRICIST
    assert normalise_stage_name("review") == ""
    assert normalise_stage_name("") == ""
    verdict = ReviewVerdict.from_dict(
        {"stage": "review", "verdict": "revise", "passed": False, "overall_score": 70,
         "restart_from": "Lyrics", "issues": [{"message": "x"}]},
        stage=LYRIC_REVIEWER,
    )
    assert verdict.restart_from == LYRICIST
    assert verdict.stage == "review"


def test_reviewers_do_not_reprompt_and_other_agents_reprompt_once():
    """Live runs cost minutes per generation, so the retry budget is explicit."""
    from suno_agents.agents import build_agent
    from suno_agents.domain.graph import COMPOSER, LYRIC_REVIEWER, MUSIC_REVIEWER, PROMPT_REVIEWER

    runner = MockRunner(["{}"])
    for stage in (LYRIC_REVIEWER, MUSIC_REVIEWER, PROMPT_REVIEWER):
        assert build_agent(stage, runner).parse_retries == 0, stage
    composer = build_agent(COMPOSER, runner)
    assert composer.parse_retries >= 1
    assert composer.parse_retries == 2

# --------------------------------------------------------------------------- #
# convergence: the loop must be able to terminate
#
# Found live: a reviewer produced 85.0 and then 90.7 with zero hard gate failures
# and kept returning "revise", because its prompt said "a generous pass is a
# failure of your job" and never stated a threshold. The pipeline would have spun
# until the revision budget ran out without ever delivering.
# --------------------------------------------------------------------------- #


def _clean_dimensions(**overrides) -> dict[str, float]:
    dims = {
        "originality": 8.0, "imagery": 8.0, "singability": 8.0, "emotional_truth": 9.0,
        "hook_strength": 8.0, "language_quality": 8.0, "vocal_fit": 8.0, "brief_fidelity": 9.0,
    }
    dims.update(overrides)
    return dims


def test_clean_gates_and_a_high_score_are_accepted_even_if_the_model_says_revise():
    verdict = ReviewVerdict.from_dict(
        {
            "stage": "lyric_reviewer", "verdict": "revise", "passed": False,
            "overall_score": 85.0, "dimensions": _clean_dimensions(),
            "issues": [], "restart_from": "lyricist", "reasoning": "still improvable",
        }
    )
    merged = finalise_review(
        LYRIC_REVIEWER, verdict, [GateResult("LG1_structure", True)],
        pass_mark=80.0, attempt=1,
    )
    assert merged.passed is True
    assert merged.verdict == "pass"
    assert "Accepted on measured quality" in merged.reasoning


def test_the_first_pass_is_never_auto_accepted():
    """The reviewer must get one full repair round to act on its own findings."""
    verdict = ReviewVerdict.from_dict(
        {
            "stage": "lyric_reviewer", "verdict": "revise", "passed": False,
            "overall_score": 92.0, "dimensions": _clean_dimensions(),
            "issues": [], "restart_from": "lyricist", "reasoning": "draft",
        }
    )
    first = finalise_review(
        LYRIC_REVIEWER, verdict, [GateResult("LG1_structure", True)],
        pass_mark=80.0, attempt=0,
    )
    assert first.passed is False
    assert first.restart_from == LYRICIST


def test_a_failed_hard_gate_defeats_the_acceptance_rule():
    verdict = ReviewVerdict.from_dict(
        {
            "stage": "lyric_reviewer", "verdict": "revise", "passed": False,
            "overall_score": 95.0, "dimensions": _clean_dimensions(),
            "issues": [], "restart_from": "intake", "reasoning": "high score",
        }
    )
    merged = finalise_review(
        LYRIC_REVIEWER,
        verdict,
        [GateResult("LG3_instrumental_consistency", False, "words in an instrumental",
                    restart_from=LYRICIST)],
        pass_mark=80.0, attempt=3,
    )
    assert merged.passed is False


def test_a_low_dimension_defeats_the_acceptance_rule():
    verdict = ReviewVerdict.from_dict(
        {
            "stage": "lyric_reviewer", "verdict": "revise", "passed": False,
            "overall_score": 90.0, "dimensions": _clean_dimensions(imagery=3.0),
            "issues": [], "restart_from": "lyricist", "reasoning": "weak imagery",
        }
    )
    merged = finalise_review(
        LYRIC_REVIEWER, verdict, [GateResult("LG1_structure", True)],
        pass_mark=80.0, attempt=3,
    )
    assert merged.passed is False
    assert any(i.code == "LOW_IMAGERY" for i in merged.issues)


def test_a_score_below_the_acceptance_mark_is_still_a_revision():
    verdict = ReviewVerdict.from_dict(
        {
            "stage": "lyric_reviewer", "verdict": "revise", "passed": False,
            "overall_score": 74.0, "dimensions": _clean_dimensions(singability=6.0),
            "issues": [{"message": "one line is un-singable", "severity": "major"}],
            "restart_from": "lyricist", "reasoning": "real defect",
        }
    )
    merged = finalise_review(
        LYRIC_REVIEWER, verdict, [GateResult("LG1_structure", True)],
        pass_mark=80.0, attempt=3,
    )
    assert merged.passed is False
    assert merged.restart_from == LYRICIST


def test_acceptance_marks_are_above_their_pass_marks():
    from suno_agents.quality.gates import ACCEPTANCE_SCORES, PASS_MARKS

    for stage, mark in ACCEPTANCE_SCORES.items():
        assert mark > PASS_MARKS[stage], stage
        assert mark - PASS_MARKS[stage] <= 6.0, (
            f"{stage}: a gap larger than a few points would make acceptance unreachable"
        )


def test_reviewer_prompts_contain_a_threshold_and_no_one_sided_bias():
    """The prompt is the actual cause of the non-terminating loop, so test it.

    Both halves matter: a numeric threshold the model can aim at, and removal of
    the instruction that framed any pass as a failure of duty.
    """
    from suno_agents.agents import build_agent
    from suno_agents.domain.graph import LYRIC_REVIEWER, MUSIC_REVIEWER, PROMPT_REVIEWER
    from suno_agents.domain.models import PipelineState as _State
    from suno_agents.quality.gates import PASS_MARKS

    PipelineState = _State
    for stage in (LYRIC_REVIEWER, MUSIC_REVIEWER, PROMPT_REVIEWER):
        agent = build_agent(stage, MockRunner(["{}"]))
        system = agent.system_prompt(PipelineState(), ())
        request = agent.user_message(PipelineState(), ())
        assert "DECIDING pass vs revise" in system, stage
        assert "do not use \"revise\" for".lower() in system.lower(), stage
        # The threshold the model must clear appears in the request it answers.
        assert f"{PASS_MARKS[stage]:.0f}/100" in request, stage
        # And the phrasing that caused the failure is gone.
        assert "generous pass is a failure" not in system.lower(), stage

# --------------------------------------------------------------------------- #
# feedback reaches the stage that must act on it, and repeats are named
#
# Motivated by a reported worry: "the reviewer does not tell the rerun agent why it
# failed, so the same mistake repeats". Two of the three claims turned out false on
# inspection - the notes do reach the target stage, and downstream stages are
# intentionally not told - but the audit found two real defects on the same path.
# --------------------------------------------------------------------------- #


def test_revision_notes_reach_the_stage_being_rerun():
    """The target stage must receive the actual complaints, not just a verdict."""
    from suno_agents.domain.models import ReviewVerdict
    from suno_agents.quality.verdict import revision_notes_from

    verdict = ReviewVerdict.from_dict(
        {
            "stage": "music_reviewer", "verdict": "revise", "passed": False,
            "overall_score": 60.0, "gates_failed": ["MG8_alignment"], "restart_from": "arranger",
            "issues": [
                {"code": "X", "severity": "blocker", "dimension": "lyrics_integration",
                 "message": "11 dari 16 baris lirik tidak punya tempat di grid",
                 "fix": "Petakan setiap baris ke bar dan beat"},
            ],
            "reasoning": "r", "strengths": ["kurva energi bagus"],
        }
    )
    notes = revision_notes_from(verdict, for_stage="arranger")
    blob = "\n".join(notes)
    assert "MG8_alignment" in blob, "the failed gate must be named"
    assert "tidak punya tempat di grid" in blob, "the reviewer's own words must survive"
    assert "Petakan setiap baris" in blob, "the suggested fix must be included"
    assert "kurva energi bagus" in blob, "strengths must be protected"


def test_the_agent_prompt_actually_contains_the_revision_notes():
    """End to end: the note must appear in the system prompt the model receives."""
    trace = TraceRecorder()
    seen: list[str] = []

    class Capturing(MockRunner):
        def complete_json(self, messages, **kwargs):
            seen.append(str(kwargs.get("system", "")))
            return super().complete_json(messages, **kwargs)

    runner = Capturing(
        sequence(
            _music_review_payload("revise", 60.0, "composer",
                                  [{"code": "X", "severity": "blocker",
                                    "message": "Form terlalu tipis", "fix": "Tambah 8 bar",
                                    "restart_from": "composer"}]),
            _music_review_payload("pass", 95.0, ""),
        )
    )
    orchestrator = Orchestrator(runner, settings=None, trace=trace)
    orchestrator.state = PipelineState(brief=None)
    orchestrator.trace = trace
    # Drive just the composer twice through the public surface.
    from suno_agents.agents import build_agent

    agent = build_agent(COMPOSER, runner, trace=trace)
    notes = ["Reviewer verdict: revise. Restart target: composer.",
             "[blocker] Form terlalu tipis -> Tambah 8 bar"]
    agent.run(PipelineState(), notes)
    assert seen, "the agent must have called the model"
    assert "REVISION PASS" in seen[-1]
    assert "Form terlalu tipis" in seen[-1], "the complaint must be in the prompt"
    assert "Tambah 8 bar" in seen[-1]


def test_a_repeated_defect_is_announced_to_the_stage():
    """The real gap the audit found: nothing said "you already failed on this".

    Without it the rewrite tends to arrive with the same defect, because the model
    has no way to connect its previous attempt to the current complaint.
    """
    from suno_agents.domain.models import ReviewVerdict
    from suno_agents.quality.verdict import failure_keys, revision_notes_from

    first = ReviewVerdict.from_dict(
        {"stage": "lyric_reviewer", "verdict": "revise", "passed": False, "overall_score": 70.0,
         "gates_failed": ["LG2_singability"], "restart_from": "lyricist",
         "issues": [{"code": "verse_too_dense", "severity": "major", "message": "padat"}]}
    )
    second = ReviewVerdict.from_dict(
        {"stage": "lyric_reviewer", "verdict": "revise", "passed": False, "overall_score": 72.0,
         "gates_failed": ["LG2_singability"], "restart_from": "lyricist",
         "issues": [{"code": "verse_too_dense", "severity": "major", "message": "masih padat"}]}
    )
    assert failure_keys(first) & failure_keys(second)

    # No history: nothing to repeat.
    plain = "\n".join(revision_notes_from(second, for_stage="lyricist"))
    assert "THIS IS A REPEAT" not in plain

    # With the earlier attempt: it must be called out, with the keys named.
    loud = "\n".join(revision_notes_from(second, for_stage="lyricist", history=[first]))
    assert "THIS IS A REPEAT" in loud
    assert "LG2_singability" in loud
    assert "cosmetic" in loud, "the instruction must demand a different approach"


def test_a_brand_new_defect_is_not_labelled_a_repeat():
    from suno_agents.domain.models import ReviewVerdict
    from suno_agents.quality.verdict import revision_notes_from

    earlier = ReviewVerdict.from_dict(
        {"stage": "lyric_reviewer", "verdict": "revise", "passed": False, "overall_score": 70.0,
         "gates_failed": ["LG2_singability"], "restart_from": "lyricist",
         "issues": [{"code": "verse_too_dense", "severity": "major", "message": "padat"}]}
    )
    fresh = ReviewVerdict.from_dict(
        {"stage": "lyric_reviewer", "verdict": "revise", "passed": False, "overall_score": 70.0,
         "gates_failed": ["LG4_cliche"], "restart_from": "lyricist",
         "issues": [{"code": "cliche_hit", "severity": "major", "message": "klise"}]}
    )
    notes = "\n".join(revision_notes_from(fresh, for_stage="lyricist", history=[earlier]))
    assert "THIS IS A REPEAT" not in notes, (
        "a different problem after an upstream change must not be mislabelled"
    )


def test_an_upstream_problem_is_not_mislabelled_as_a_repeat():
    """A defect reported by a different stage is a new context, not a repeat."""
    from suno_agents.domain.models import ReviewVerdict
    from suno_agents.quality.verdict import revision_notes_from

    earlier = ReviewVerdict.from_dict(
        {"stage": "music_reviewer", "verdict": "revise", "passed": False, "overall_score": 60.0,
         "gates_failed": ["MG2_harmony"], "restart_from": "composer",
         "issues": [{"code": "no_functions", "severity": "blocker", "message": "tanpa fungsi"}]}
    )
    now = ReviewVerdict.from_dict(
        {"stage": "lyric_reviewer", "verdict": "revise", "passed": False, "overall_score": 70.0,
         "gates_failed": ["LG2_singability"], "restart_from": "lyricist",
         "issues": [{"code": "verse_too_dense", "severity": "major", "message": "padat"}]}
    )
    notes = "\n".join(revision_notes_from(now, for_stage="lyricist", history=[earlier]))
    assert "THIS IS A REPEAT" not in notes


def test_the_repeat_is_visible_in_the_events_and_the_report():
    """The loop must say it is repeating, in the UI and in the written report."""
    def review(stage, verdict, restart, issues):
        return {
            "stage": stage, "verdict": verdict, "passed": verdict == "pass",
            "overall_score": 70.0, "gates_failed": ["LG2_singability"],
            "restart_from": restart, "issues": issues, "reasoning": "r", "strengths": [],
        }

    same_issue = [{"code": "verse_too_dense", "severity": "major",
                   "dimension": "singability", "message": "padat", "restart_from": LYRICIST}]
    script = [
        json_responder(_demo("intake")), json_responder(_demo("audio_analyst")),
        json_responder(_demo("composer")), json_responder(_demo("lyricist")),
        json_responder(review("lyric_reviewer", "revise", LYRICIST, same_issue)),
        json_responder(_demo("lyricist")),
        json_responder(review("lyric_reviewer", "revise", LYRICIST, same_issue)),
        json_responder(_demo("lyricist")),
        json_responder({
            "stage": "lyric_reviewer", "verdict": "pass", "passed": True, "overall_score": 90.0,
            "dimensions": {"originality": 9.0, "imagery": 9.0, "singability": 9.0,
                           "emotional_truth": 9.0, "hook_strength": 9.0,
                           "language_quality": 9.0, "vocal_fit": 9.0, "brief_fidelity": 9.0},
            "gates_failed": [], "issues": [], "restart_from": "", "reasoning": "ok",
            "strengths": [],
        }),
        json_responder(_demo("arranger")),
        json_responder({
            "stage": "music_reviewer", "verdict": "pass", "passed": True, "overall_score": 91.0,
            "dimensions": {k: 9.0 for k in (
                "structure", "harmony", "melody_contour", "groove_and_rhythm", "energy_arc",
                "hook_strength", "lyrics_integration", "character_and_identity",
                "producibility", "lossless_readiness")},
            "gates_failed": [], "issues": [], "restart_from": "", "reasoning": "ok",
            "strengths": [],
        }),
        json_responder(_demo("prompt_engineer")),
        json_responder({
            "stage": "prompt_reviewer", "verdict": "pass", "passed": True, "overall_score": 92.0,
            "dimensions": {k: 9.0 for k in (
                "prompt_fidelity", "style_clarity", "descriptor_balance", "lyrics_format",
                "exclusion_effectiveness", "structural_clarity", "vocal_direction",
                "title_hook", "render_risk", "lossless_readiness")},
            "gates_failed": [], "issues": [], "restart_from": "", "reasoning": "ok",
            "strengths": [],
        }),
    ]
    events: list = []
    orchestrator = Orchestrator(MockRunner(script), settings=None, on_event=events.append)
    result = orchestrator.run(None)
    assert result.passed is True

    rollbacks = [e for e in events if e.kind == "rollback"]
    assert len(rollbacks) == 2
    assert rollbacks[0].data.get("repeat") is False
    assert rollbacks[1].data.get("repeat") is True, "the second pass repeats the same defect"
    assert "MENGULANG" in rollbacks[1].message
    assert rollbacks[1].data.get("attempts_at_target") == 2

    report = result.report
    assert "PENGULANGAN MASALAH" in report
    section = report[report.find("PENGULANGAN MASALAH"):]
    assert "dilaporkan 2x" in section, section[:300]
    assert "verse_too_dense" in section

def _demo(stage: str):
    """The demo payload for a production stage, by stage id."""
    from suno_agents.testing.demo import (
        DEMO_EXCLUDE_GOOD, _arrangement_payload, _blueprint_payload, _intake_payload,
        _lyric_payload, _prompt_payload, _reference_payload,
    )

    return {
        "intake": _intake_payload,
        "audio_analyst": _reference_payload,
        "composer": _blueprint_payload,
        "lyricist": _lyric_payload,
        "arranger": lambda: _arrangement_payload(True),
        "prompt_engineer": lambda: _prompt_payload(DEMO_EXCLUDE_GOOD),
    }[stage]()


def _music_review_payload(verdict: str, score: float, restart: str, issues: list | None = None):
    return {
        "stage": "music_reviewer", "verdict": verdict, "passed": verdict == "pass",
        "overall_score": score, "restart_from": restart, "issues": issues or [],
        "reasoning": "r", "strengths": [],
        "dimensions": {k: 9.0 for k in (
            "structure", "harmony", "melody_contour", "groove_and_rhythm", "energy_arc",
            "hook_strength", "lyrics_integration", "character_and_identity",
            "producibility", "lossless_readiness")},
    }

def test_the_loop_stops_when_a_defect_survives_three_attempts():
    """A defect that outlives two rewrites of the same stage will not be fixed by a
    third. A live run spent three full regenerations on one gate that could never be
    satisfied; stopping with the best artifact is the honest outcome, and the user
    asked explicitly for "not so strict that one song takes forever"."""
    def lyric_review(score: float):
        return {
            "stage": "lyric_reviewer", "verdict": "revise", "passed": False,
            "overall_score": score, "gates_failed": ["LG7_user_constraints"],
            "restart_from": LYRICIST,
            "issues": [{"code": "missing_directive", "severity": "major",
                        "dimension": "brief_fidelity", "message": "sama",
                        "restart_from": LYRICIST}],
            "reasoning": "r", "strengths": [],
        }

    passing_music = {
        "stage": "music_reviewer", "verdict": "pass", "passed": True, "overall_score": 91.0,
        "dimensions": {k: 9.0 for k in (
            "structure", "harmony", "melody_contour", "groove_and_rhythm", "energy_arc",
            "hook_strength", "lyrics_integration", "character_and_identity",
            "producibility", "lossless_readiness")},
        "gates_failed": [], "issues": [], "restart_from": "", "reasoning": "ok", "strengths": [],
    }
    # The gate must genuinely fail each time, otherwise the acceptance rule fires at
    # the second attempt (correctly) and the loop never reaches three repeats. The
    # brief asks for a quoted line the demo lyric sheet does not contain, which is
    # exactly the shape of the live run this rule was written for.
    from suno_agents.testing.demo import demo_brief

    brief = demo_brief()
    brief.must_include = ['sertakan kalimat "baris yang tidak pernah ada di lirik ini"']

    runner = StageScripts(
        {
            "intake": [json_responder(_demo("intake"))],
            "audio_analyst": [json_responder(_demo("audio_analyst"))],
            "composer": [json_responder(_demo("composer"))],
            "lyricist": [json_responder(_demo("lyricist"))],
            "lyric_reviewer": [
                json_responder(lyric_review(86.0)),
                json_responder(lyric_review(87.0)),
                json_responder(lyric_review(87.1)),
                json_responder(lyric_review(87.2)),
            ],
            "arranger": [json_responder(_demo("arranger"))],
            "music_reviewer": [json_responder(passing_music)],
            "prompt_engineer": [json_responder(_demo("prompt_engineer"))],
        }
    )
    events: list = []
    orchestrator = Orchestrator(runner, settings=_settings(max_revisions=24), on_event=events.append)
    result = orchestrator.run(brief)

    # Three rejections, two rollbacks: the loop stops on the third failure rather
    # than scheduling a fourth attempt that would fail the same way.
    rejections = [v for v in result.state.review_log if not v.passed]
    assert len(rejections) == 3, [v.overall_score for v in rejections]
    assert result.state.revisions == 2, result.state.revisions
    assert "tiga kali" in result.stopped_reason or "tidak membuang waktu" in result.stopped_reason
    warnings = [e for e in events if e.kind == "warn" and e.data.get("repeated")]
    assert warnings, "the stop must name which defect repeated"
    repeated = warnings[0].data["repeated"]
    # Keys are prefixed by kind (gate:/issue:/dimension:) so a reader can tell whether
    # a machine check or a reviewer's finding is the one that keeps coming back.
    assert any("missing_directive" in key for key in repeated), repeated
    assert any("LG7_user_constraints" in key for key in repeated), repeated

class StageScripts:
    """A runner that gives each stage its own script.

    A flat MockRunner shares one cursor across every stage, so with two reviewers in
    the mix the responses drift onto the wrong stage and the test silently measures
    the mock instead of the loop.
    """

    def __init__(self, scripts: dict[str, list], *, trace=None) -> None:
        self.model = "routed-script"
        self.trace = trace
        self._scripts = scripts
        self._runners: dict[str, MockRunner] = {}

    def for_stage(self, stage: str) -> MockRunner:
        runner = self._runners.get(stage)
        if runner is None:
            runner = MockRunner(self._scripts.get(stage, ["{}"]), model=f"routed:{stage}", trace=self.trace)
            runner.label = f"{stage}#1"
            self._runners[stage] = runner
        return runner

    def complete(self, *args, **kwargs):
        return self.for_stage("intake").complete(*args, **kwargs)

    def complete_json(self, *args, **kwargs):
        return self.for_stage("intake").complete_json(*args, **kwargs)

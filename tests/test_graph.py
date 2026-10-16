"""The pipeline graph is the contract that keeps rollbacks sane.

These tests pin the rules that make "the reviewer decides where we restart"
predictable: only legal ancestors, model-invented names mapped onto real stages,
and escalation when a stage keeps failing.
"""

from __future__ import annotations

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
    STAGE_ORDER,
    earlier_stages,
    escalate,
    is_reviewer,
    legal_restarts,
    normalise_restart_name,
    resolve_restart,
    restart_depth,
    stage_index,
    stages_between,
)


def test_stage_order_is_the_nine_specialists():
    assert STAGE_ORDER == (
        INTAKE,
        AUDIO_ANALYST,
        COMPOSER,
        LYRICIST,
        LYRIC_REVIEWER,
        ARRANGER,
        MUSIC_REVIEWER,
        PROMPT_ENGINEER,
        PROMPT_REVIEWER,
    )
    assert stage_index(INTAKE) == 0
    assert stage_index(PROMPT_REVIEWER) == 8


def test_reviewer_can_only_reach_legal_ancestors():
    for reviewer in (LYRIC_REVIEWER, MUSIC_REVIEWER, PROMPT_REVIEWER):
        assert is_reviewer(reviewer)
        for target in legal_restarts(reviewer):
            assert stage_index(target) < stage_index(reviewer)


def test_lyric_reviewer_cannot_roll_back_to_itself_or_forward():
    assert resolve_restart(LYRIC_REVIEWER, LYRIC_REVIEWER) in legal_restarts(LYRIC_REVIEWER)
    assert resolve_restart(LYRIC_REVIEWER, ARRANGER) == LYRICIST


def test_restart_alias_mapping():
    assert normalise_restart_name("Lyrics") == LYRICIST
    assert normalise_restart_name("lirik") == LYRICIST
    assert normalise_restart_name("Nada") == COMPOSER
    assert normalise_restart_name("audio analysis") == AUDIO_ANALYST
    assert normalise_restart_name("suno") == PROMPT_ENGINEER
    assert normalise_restart_name("brief") == INTAKE
    assert normalise_restart_name("totally-made-up") == ""
    assert normalise_restart_name("") == ""


def test_resolve_restart_honours_first_legal_choice_in_a_priority_list():
    target = resolve_restart(PROMPT_REVIEWER, ["nonsense", "arranger", "composer"])
    assert target == ARRANGER


def test_resolve_restart_falls_back_to_closest_legal_ancestor():
    # The model asked for something impossible; we must still return something sane.
    assert resolve_restart(MUSIC_REVIEWER, "imaginary_stage") == ARRANGER
    assert resolve_restart(INTAKE, "anything") == ""


def test_escalation_walks_upstream_and_stops_at_the_brief():
    # Escalation skips reviewers so it never re-approves work that already passed.
    assert escalate(ARRANGER) == LYRICIST
    assert escalate(ARRANGER, 3) == AUDIO_ANALYST
    assert escalate(INTAKE) == ""
    assert escalate(COMPOSER, 99) == INTAKE
    assert escalate(PROMPT_REVIEWER) == PROMPT_ENGINEER
    assert escalate(PROMPT_REVIEWER, 2) == ARRANGER


def test_restart_depth_measures_discarded_work():
    assert restart_depth(PROMPT_REVIEWER, PROMPT_ENGINEER) == 1
    assert restart_depth(PROMPT_REVIEWER, INTAKE) == 8
    # `composer` IS reachable from the prompt reviewer (its lyrics depend on it).
    assert restart_depth(PROMPT_REVIEWER, COMPOSER) == 6
    # A forward or self target is not a rollback at all.
    assert restart_depth(INTAKE, COMPOSER) == 0
    assert restart_depth(PROMPT_REVIEWER, PROMPT_REVIEWER) == 0


def test_earlier_stages_and_ranges():
    assert earlier_stages(INTAKE) == ()
    assert earlier_stages(COMPOSER) == (INTAKE, AUDIO_ANALYST)
    assert stages_between(LYRICIST, ARRANGER) == (LYRICIST, LYRIC_REVIEWER, ARRANGER)
    assert stages_between(ARRANGER, COMPOSER) == ()

def test_escalation_ladder_is_monotonic():
    ladder = [escalate(PROMPT_REVIEWER, steps) for steps in range(1, 9)]
    indexes = [stage_index(name) for name in ladder if name]
    assert indexes == sorted(indexes, reverse=True)
    assert ladder[-1] == INTAKE

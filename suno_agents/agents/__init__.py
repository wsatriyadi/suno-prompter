"""Agent registry: stage id -> agent class.

The orchestrator builds agents from here, and the UI uses it to describe the
team. Adding a tenth specialist means one entry in this table plus its module.
"""

from __future__ import annotations

from typing import Callable

from ..domain.graph import (
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
)
from .analyst import AudioAnalystAgent
from .arranger import ArrangerAgent
from .base import Agent, AgentError, AgentRun
from .composer import ComposerAgent
from .intake import IntakeAgent
from .lyric_reviewer import LyricReviewerAgent
from .lyricist import LyricistAgent
from .music_reviewer import MusicReviewerAgent
from .prompt_engineer import PromptEngineerAgent
from .prompt_reviewer import PromptReviewerAgent

AGENT_CLASSES: dict[str, Callable[..., Agent]] = {
    INTAKE: IntakeAgent,
    AUDIO_ANALYST: AudioAnalystAgent,
    COMPOSER: ComposerAgent,
    LYRICIST: LyricistAgent,
    LYRIC_REVIEWER: LyricReviewerAgent,
    ARRANGER: ArrangerAgent,
    MUSIC_REVIEWER: MusicReviewerAgent,
    PROMPT_ENGINEER: PromptEngineerAgent,
    PROMPT_REVIEWER: PromptReviewerAgent,
}


def build_agent(stage: str, runner, **kwargs) -> Agent:
    """Instantiate the specialist for ``stage``."""
    try:
        factory = AGENT_CLASSES[stage]
    except KeyError as exc:  # pragma: no cover - programming error
        raise AgentError(f"No agent registered for stage '{stage}'.") from exc
    return factory(runner, **kwargs)


def all_stages() -> tuple[str, ...]:
    return STAGE_ORDER


__all__ = [
    "AGENT_CLASSES",
    "build_agent",
    "all_stages",
    "Agent",
    "AgentError",
    "AgentRun",
]

"""Lirik Studio - a specialist-agent pipeline that turns a music idea into a
production-ready Suno prompt.

Public surface is intentionally small: the UI and CLI talk to
:class:`suno_agents.pipeline.orchestrator.Orchestrator` and
:class:`suno_agents.config.providers.ProviderStore`.
"""

__version__ = "1.0.0"
APP_NAME = "Lirik Studio"

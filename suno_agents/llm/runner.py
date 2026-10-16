"""Runner abstraction: what an agent needs from an LLM.

Agents never touch HTTP. They receive a :class:`Runner`, which is either a
:class:`~suno_agents.llm.client.LLMClient` wrapper (live) or a scripted
:class:`~suno_agents.llm.mock.MockRunner` (tests, offline demo).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .wire import ChatMessage, ChatResult


@runtime_checkable
class Runner(Protocol):
    """Minimal surface an agent may rely on."""

    model: str

    def complete(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        *,
        system: str | None = ...,
        temperature: float | None = ...,
        max_tokens: int | None = ...,
        json_mode: str | None = ...,
        schema: Mapping[str, Any] | None = ...,
        on_delta: Callable[[str], None] | None = ...,
    ) -> ChatResult: ...

    def complete_json(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        *,
        system: str | None = ...,
        schema: Mapping[str, Any] | None = ...,
        temperature: float | None = ...,
        max_tokens: int | None = ...,
        on_delta: Callable[[str], None] | None = ...,
        retries: int = ...,
    ) -> tuple[Any, ChatResult]: ...


class ClientRunner:
    """Adapts a client to the Runner protocol, and routes per stage.

    ``settings`` is optional but strongly recommended: without it every stage uses
    the active provider's model and the per-stage routing in the settings dialog
    does nothing. That was a real defect - the orchestrator asked for
    ``for_stage()``, only the test runners had it, and the UI promised routing it
    never delivered.
    """

    def __init__(self, client, *, model: str | None = None, settings=None) -> None:
        self._client = client
        self.settings = settings
        self.model = model or getattr(client.provider, "model", "")
        self._routed: dict[str, Any] = {}

    @property
    def client(self):
        return self._client

    # ------------------------------------------------------------------ routing
    def for_stage(self, stage: str):
        """Return the runner for ``stage``, honouring its routing override.

        Cached per stage: the wrapper is cheap, but rebuilding it on every call
        would discard the learned parameter-compatibility state of the endpoint.
        """
        if self.settings is None:
            return None
        if stage in self._routed:
            return self._routed[stage]
        target = self.settings.routing_for(stage)
        if target.provider is None or not target.model:
            return None
        base = self._client
        # A different provider means a different endpoint, hence a different client.
        if target.provider.id != getattr(base.provider, "id", None):
            from .client import LLMClient, StageRoutedClient

            base = LLMClient(target.provider, on_event=getattr(base, "on_event", None))
        runner = ClientRunner(
            _pinned(base, target.model, target.max_tokens),
            model=target.model,
        )
        self._routed[stage] = runner
        return runner

    def describe_routing(self) -> dict[str, str]:
        """What every stage will actually use, for the UI and for bug reports."""
        if self.settings is None:
            return {}
        out: dict[str, str] = {}
        for stage in self.settings.stage_overrides:
            target = self.settings.routing_for(stage)
            out[stage] = target.describe()
        return out

    # ------------------------------------------------------------------ runner
    def complete(self, messages, **kwargs) -> ChatResult:
        kwargs.pop("extra", None)
        return self._client.chat(messages, **kwargs)

    def complete_json(self, messages, **kwargs):
        kwargs.pop("extra", None)
        return self._client.chat_json(messages, **kwargs)


def _pinned(client, model: str, max_tokens: int | None):
    """Pin a model onto a client, via the wrapper when one is needed."""
    if model == getattr(client.provider, "model", None) and not max_tokens:
        return client
    from .client import StageRoutedClient

    return StageRoutedClient(client, model=model, max_tokens=max_tokens)


class NullRunner:
    """Runner that fails loudly. Used when no provider is configured."""

    model = "none"

    def complete(self, *_args, **_kwargs) -> ChatResult:
        raise RuntimeError(
            "No LLM provider is configured. Open Settings -> LLM Providers, "
            "add an OpenAI-compatible endpoint and press Test."
        )

    def complete_json(self, *_args, **_kwargs):
        self.complete()


__all__ = ["Runner", "ClientRunner", "NullRunner"]

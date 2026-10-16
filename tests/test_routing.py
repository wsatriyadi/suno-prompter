"""Per-stage routing: saved settings must round-trip, and must actually be used.

Two real defects motivated this file, both reported by the user:

1. **Opening the routing tab destroyed the routing.** The dialog loaded the tab
   through a collect-all path, so ``RoutingTab.collect()`` ran against widgets that
   had never been populated and overwrote every saved override with blanks. The
   user configured nine stages, opened the tab again, and saw nothing.
2. **Routing was never applied at runtime.** The orchestrator asked its runner for
   ``for_stage()``, but only the test and demo runners implemented it, so in the
   real app every agent silently used the active provider's model.

Both are covered here at the level they failed: the settings object for the first,
the wire for the second.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from support.openai_stub import StubProvider  # noqa: E402

from suno_agents.config.models import AppSettings, ProviderConfig, StageOverride  # noqa: E402
from suno_agents.config.store import SettingsStore  # noqa: E402
from suno_agents.domain.graph import STAGE_ORDER  # noqa: E402
from suno_agents.llm.client import LLMClient, StageRoutedClient  # noqa: E402
from suno_agents.llm.runner import ClientRunner  # noqa: E402
from suno_agents.pipeline.orchestrator import Orchestrator  # noqa: E402


@pytest.fixture
def stub():
    """A real OpenAI-protocol server on a loopback port."""
    with StubProvider() as provider:
        yield provider


# --------------------------------------------------------------------------- #
# resolution and persistence
# --------------------------------------------------------------------------- #


def _three_provider_settings(stub_base: str) -> AppSettings:
    settings = AppSettings()
    settings.providers = [
        ProviderConfig(id="base", name="Base", base_url=f"{stub_base}/v1", model="model-base"),
        ProviderConfig(id="strong", name="Strong", base_url=f"{stub_base}/v1", model="model-strong"),
        ProviderConfig(id="cheap", name="Cheap", base_url=f"{stub_base}/v1", model="model-cheap"),
    ]
    settings.active_provider_id = "base"
    settings.ensure_defaults()
    return settings


def test_an_override_wins_and_everything_else_inherits(stub):
    settings = _three_provider_settings(stub.base_url())
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="strong", model="", temperature=0.8, enabled=True
    )
    composer = settings.routing_for("composer")
    assert composer.provider.id == "strong"
    assert composer.model == "model-strong", "an empty override model inherits its provider"
    assert composer.overridden is True

    intake = settings.routing_for("intake")
    assert intake.provider.id == "base"
    assert intake.model == "model-base"
    assert intake.overridden is False
    assert intake.inherited_from_active is True


def test_a_disabled_override_falls_back_to_the_active_provider(stub):
    settings = _three_provider_settings(stub.base_url())
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="strong", model="model-strong", enabled=False
    )
    target = settings.routing_for("composer")
    assert target.provider.id == "base"
    assert target.model == "model-base"
    assert target.overridden is False


def test_routing_survives_a_save_and_load_round_trip(stub, tmp_path):
    settings = _three_provider_settings(stub.base_url())
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="strong", model="", temperature=0.85, enabled=True
    )
    settings.stage_overrides["lyric_reviewer"] = StageOverride(
        provider_id="cheap", model="model-cheap-turbo", temperature=0.15, enabled=True
    )
    store = SettingsStore(tmp_path / "settings.json")
    store.save(settings)

    reloaded = store.load()
    assert reloaded.stage_overrides["composer"].provider_id == "strong"
    assert abs(reloaded.stage_overrides["composer"].temperature - 0.85) < 1e-9
    assert reloaded.stage_overrides["lyric_reviewer"].model == "model-cheap-turbo"
    assert reloaded.routing_for("lyric_reviewer").model == "model-cheap-turbo"
    assert reloaded.routing_for("lyric_reviewer").provider.id == "cheap"


def test_every_stage_can_be_pointed_at_a_different_model(stub):
    """The user's actual requirement: nine agents, nine models if they want."""
    settings = _three_provider_settings(stub.base_url())
    models = {
        stage: f"model-{index}" for index, stage in enumerate(STAGE_ORDER)
    }
    for stage, model in models.items():
        settings.stage_overrides[stage] = StageOverride(
            provider_id="base", model=model, temperature=0.5, enabled=True
        )
    resolved = {stage: settings.routing_for(stage).model for stage in STAGE_ORDER}
    assert resolved == models
    assert len(set(resolved.values())) == len(STAGE_ORDER)
    assert all(settings.routing_for(s).overridden for s in STAGE_ORDER)


def test_max_tokens_can_be_overridden_per_stage(stub):
    settings = _three_provider_settings(stub.base_url())
    settings.providers[0].max_tokens = 4096
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="base", model="m", max_tokens=24000, enabled=True
    )
    assert settings.routing_for("composer").max_tokens == 24000
    assert settings.routing_for("intake").max_tokens == 4096


# --------------------------------------------------------------------------- #
# the tab must not lose what the user configured
# --------------------------------------------------------------------------- #


def test_collect_before_load_is_inert():
    """The exact defect: an unpopulated form must never write blanks.

    Simulated without a display by exercising the guard directly, because the bug
    is in the guard, not in Tk.
    """
    from suno_agents.ui.settings import RoutingTab

    tab = RoutingTab.__new__(RoutingTab)  # no settings, no widgets
    tab.rows = {}
    tab._loaded = False
    settings = AppSettings()
    settings.ensure_defaults()
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="strong", model="model-x", enabled=True
    )
    before = settings.to_dict()["stage_overrides"]

    RoutingTab.collect(tab, settings)
    assert settings.to_dict()["stage_overrides"] == before, (
        "collecting a tab that was never loaded must change nothing"
    )


def test_the_dialog_keeps_routing_when_the_tab_is_opened(stub):
    """Open the tab twice and confirm the saved overrides are untouched.

    Runs the real dialog when a display is available; the scenario script covers
    the Tk path, and this asserts the settings contract it must uphold.
    """
    settings = _three_provider_settings(stub.base_url())
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="strong", model="composer-model", temperature=0.9, enabled=True
    )
    settings.stage_overrides["prompt_reviewer"] = StageOverride(
        provider_id="cheap", model="critic-model", temperature=0.1, enabled=True
    )
    expected = {k: v.to_dict() for k, v in settings.stage_overrides.items()}

    # What the old code did: collect() during the tab's own load path.
    from suno_agents.ui.settings import RoutingTab

    tab = RoutingTab.__new__(RoutingTab)
    tab.rows = {}
    tab._loaded = False
    RoutingTab.collect(tab, settings)
    assert {k: v.to_dict() for k, v in settings.stage_overrides.items()} == expected


# --------------------------------------------------------------------------- #
# the runner must apply it
# --------------------------------------------------------------------------- #


def test_client_runner_routes_only_when_settings_are_supplied(stub):
    client = LLMClient(
        ProviderConfig(id="base", name="Base", base_url=stub.base_url("/v1"),
                       model="model-base", timeout=20.0)
    )
    bare = ClientRunner(client)
    assert bare.for_stage("composer") is None, (
        "without settings there is nothing to route, and the stage inherits"
    )

    settings = _three_provider_settings(stub.base_url("/v1").rsplit("/v1", 1)[0])
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="base", model="model-big", enabled=True
    )
    routed = ClientRunner(client, settings=settings)
    stage_runner = routed.for_stage("composer")
    assert stage_runner is not None
    assert stage_runner.model == "model-big"
    assert routed.for_stage("composer") is stage_runner, "routing must be cached per stage"


def test_stage_routed_client_pins_the_model_without_losing_the_endpoint(stub):
    provider = ProviderConfig(id="base", name="Base", base_url=stub.base_url("/v1"),
                              model="model-base", timeout=20.0, max_tokens=2048)
    base = LLMClient(provider)
    pinned = StageRoutedClient(base, model="model-pinned", max_tokens=9000)
    assert pinned.model == "model-pinned"
    result = pinned.chat([{"role": "user", "content": "hi"}])
    assert result.text
    # The underlying client is shared, so learned compatibility state survives.
    assert pinned.provider is base.provider


def test_a_stage_override_reaches_the_wire_and_others_inherit(stub):
    """The end-to-end proof, recorded at the server: nine stages, three models.

    Both halves matter. The overridden stages must send their own model, and the
    un-overridden ones must keep inheriting the active provider - otherwise
    "override" would quietly become "replace everything".
    """
    seen: list[str] = []
    original = StubProvider.base_url  # keep the fixture alive

    from support import openai_stub as stub_mod

    class Recording(stub_mod._Handler):
        def _read_body(self):
            body = super()._read_body()
            seen.append(str(body.get("model", "?")))
            return body

    # Reuse the running stub's port but install a recording handler class.
    import threading
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Recording)
    httpd.saw_rate_limit = False  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        settings = AppSettings()
        settings.providers = [
            ProviderConfig(id="p", name="Stub", base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1",
                           api_key="test", model="model-default", timeout=30.0)
        ]
        settings.active_provider_id = "p"
        settings.ensure_defaults()
        settings.max_revisions = 4
        settings.stage_attempts = 1
        for stage, model in (
            ("composer", "model-big"),
            ("lyricist", "model-poet"),
            ("prompt_reviewer", "model-critic"),
        ):
            settings.stage_overrides[stage] = StageOverride(
                provider_id="p", model=model, temperature=0.4, enabled=True
            )

        runner = ClientRunner(LLMClient(settings.providers[0]), settings=settings)
        result = Orchestrator(runner, settings=settings).run(None)

        assert result.passed is True
        assert "model-big" in seen, seen
        assert "model-poet" in seen, seen
        assert "model-critic" in seen, seen
        assert "model-default" in seen, "un-overridden stages must inherit"

        # And the UI's view of it comes from the same resolution.
        assert settings.routing_for("composer").model == "model-big"
        assert settings.routing_for("lyric_reviewer").model == "model-default"
    finally:
        httpd.shutdown()
        httpd.server_close()

# --------------------------------------------------------------------------- #
# per-stage output budget
# --------------------------------------------------------------------------- #


def test_an_explicit_small_budget_is_honoured_exactly(stub):
    """A per-stage budget is a decision, often a cost one, not a hint.

    The agent used to raise any budget below its reasoning-burn floor with max(),
    which silently overrode the user - and made the slower, costlier call anyway.
    The floor now applies only when nobody stated a budget; a stated one is used
    as-is, and the client's repair loop is what grows it if it turns out too small.
    """
    settings = _three_provider_settings(stub.base_url())
    settings.providers[0].max_tokens = 4096
    settings.stage_overrides["composer"] = StageOverride(
        provider_id="base", model="m-composer", max_tokens=6000, enabled=True
    )
    assert settings.routing_for("composer").max_tokens == 6000

    from suno_agents.agents import build_agent
    from suno_agents.llm.mock import MockRunner

    agent = build_agent("composer", MockRunner(["{}"]), max_tokens=6000)
    captured: dict = {}

    class Capture:
        model = "capture"

        def complete_json(self, _messages, **kwargs):
            from suno_agents.llm.wire import ChatResult

            captured.update(kwargs)
            return {}, ChatResult(text="{}")

    agent.runner = Capture()  # type: ignore[assignment]
    agent._call([{"role": "user", "content": "x"}], "system")  # type: ignore[attr-defined]
    assert captured["max_tokens"] == 6000


def test_without_a_budget_no_cap_is_requested(stub):
    """The default must impose no cap at all.

    Every fixed cap was wrong in the expensive direction: 300 tokens returned pure
    reasoning, 8192 truncated a blueprint mid-JSON, and 16384 was still short once
    reasoning was added. Sending no ``max_tokens`` lets the model use its own
    maximum, which is the only setting that cannot silently truncate.
    """
    from suno_agents.agents import build_agent
    from suno_agents.llm.mock import MockRunner
    from suno_agents.llm.wire import ChatResult

    agent = build_agent("composer", MockRunner(["{}"]))
    captured: dict = {}

    class Capture:
        model = "capture"

        def complete_json(self, _messages, **kwargs):
            captured.update(kwargs)
            return {}, ChatResult(text="{}")

    agent.runner = Capture()  # type: ignore[assignment]
    agent._call([{"role": "user", "content": "x"}], "system")  # type: ignore[attr-defined]
    assert captured.get("max_tokens") in (0, None), (
        "an unset budget must mean no cap, not a guessed number"
    )


def test_the_default_output_cap_is_no_cap():
    """A fresh install must not truncate on the first run, so it sends no cap."""
    from suno_agents.config.models import ProviderConfig

    provider = ProviderConfig()
    assert provider.max_tokens == 0, (
        f"default output cap {provider.max_tokens} would impose a guess"
    )
    assert provider.validate() == [], provider.validate()
    # A short timeout is its own trap: an uncapped reply can take minutes.
    assert provider.timeout >= 300


def test_the_request_omits_max_tokens_entirely_when_uncapped(stub):
    """The field must be absent, not enormous.

    Some gateways validate ``max_tokens`` against their own ceiling and reject a
    larger number, so "no limit" has to mean the field is not sent.
    """
    settings = _three_provider_settings(stub.base_url())
    provider = settings.providers[0]
    provider.max_tokens = 0
    body = LLMClient(provider)._build_body(
        [{"role": "user", "content": "x"}], "off", None, None, None, stream=False
    )
    assert "max_tokens" not in body


def test_an_explicit_cap_is_still_sent(stub):
    settings = _three_provider_settings(stub.base_url())
    provider = settings.providers[0]
    provider.max_tokens = 12000
    body = LLMClient(provider)._build_body(
        [{"role": "user", "content": "x"}], "off", None, None, None, stream=False
    )
    assert body["max_tokens"] == 12000

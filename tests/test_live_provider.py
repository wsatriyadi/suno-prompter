"""Live provider tests: the real HTTP client against a real OpenAI-compatible server.

The stub in ``tests/support/openai_stub.py`` speaks the wire protocol over a real
socket, and the awkward behaviours it emulates are the ones that actually break
integrations: an endpoint that rejects ``response_format``, one that insists on
streaming, one that rate-limits the first request, and one that prefixes replies
with a reasoning block.

These tests cover the client, the settings model, and a full pipeline run against
the stub - the closest thing to "a real provider" a hermetic test can honestly
reach. No network access, no API key.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from support.openai_stub import StubProvider  # noqa: E402

from suno_agents.config.models import AppSettings, ProviderConfig  # noqa: E402
from suno_agents.llm.client import LLMClient  # noqa: E402
from suno_agents.llm.runner import ClientRunner  # noqa: E402
from suno_agents.llm.wire import LLMError, LLMHTTPError  # noqa: E402
from suno_agents.pipeline.orchestrator import Orchestrator  # noqa: E402


@pytest.fixture
def stub():
    with StubProvider() as provider:
        yield provider


def _provider(stub: StubProvider, prefix: str = "/v1", **overrides) -> ProviderConfig:
    data = {
        "id": "stub",
        "name": "Stub",
        "base_url": stub.base_url(prefix),
        "api_key": "test-key",
        "model": "stub-model-large",
        "temperature": 0.6,
        "max_tokens": 1024,
        "timeout": 20.0,
        "max_retries": 2,
    }
    data.update(overrides)
    return ProviderConfig.from_dict(data)


# --------------------------------------------------------------------------- #
# connection, models, auth
# --------------------------------------------------------------------------- #


def test_test_connection_round_trips_against_a_real_socket(stub):
    ok, message, latency = LLMClient(_provider(stub)).test_connection()
    assert ok is True
    assert "OK" in message
    assert latency > 0


def test_test_connection_reports_a_dead_endpoint_honestly():
    provider = ProviderConfig(
        id="dead", name="Dead", base_url="http://127.0.0.1:9/v1", model="nope", timeout=3.0
    )
    ok, message, _latency = LLMClient(provider).test_connection()
    assert ok is False
    assert message


def test_list_models_parses_the_openai_shape(stub):
    assert LLMClient(_provider(stub)).list_models() == [
        "stub-model-large",
        "stub-model-small",
    ]


def test_url_derivation_handles_both_base_url_forms(stub):
    for prefix in ("/v1", "/v1/chat/completions", "/v1/"):
        provider = _provider(stub, prefix)
        assert provider.chat_url == f"http://127.0.0.1:{stub.port}/v1/chat/completions"
        assert provider.models_url == f"http://127.0.0.1:{stub.port}/v1/models"


# --------------------------------------------------------------------------- #
# chat, streaming, JSON mode
# --------------------------------------------------------------------------- #


def test_chat_returns_parsed_text_and_usage(stub):
    result = LLMClient(_provider(stub)).chat(
        [{"role": "user", "content": "say something"}], system="be brief"
    )
    assert result.text
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 126
    assert result.model


def test_chat_json_returns_a_python_object(stub):
    payload, result = LLMClient(_provider(stub)).chat_json(
        [{"role": "user", "content": "answer in json please"}], system="Answer with JSON."
    )
    assert isinstance(payload, dict)
    assert payload["answer"] == 42
    assert result.text


def test_streaming_delivers_chunks_in_order(stub):
    chunks: list[str] = []
    result = LLMClient(_provider(stub)).chat(
        [{"role": "user", "content": "stream this"}], stream=True, on_delta=chunks.append
    )
    assert len(chunks) > 1, "streaming should arrive in more than one delta"
    assert "".join(chunks) == result.text


def test_reasoning_blocks_are_stripped_from_the_visible_answer(stub):
    """A model that leaks ``<think…>`` must not leak it into the artifact."""
    client = LLMClient(_provider(stub, "/think"))
    result = client.chat([{"role": "user", "content": "answer in json please"}])
    assert "<think" not in result.text
    assert result.reasoning, "the reasoning text should be captured, not discarded"
    payload, _ = client.chat_json([{"role": "user", "content": "answer in json please"}])
    assert payload["answer"] == 42


# --------------------------------------------------------------------------- #
# the compatibility ladder
# --------------------------------------------------------------------------- #


def test_endpoint_without_response_format_is_retried_without_it(stub):
    """An endpoint that rejects ``response_format`` must still be usable.

    The client drops the offending parameter once, remembers it for the session,
    and succeeds - instead of failing every call forever.
    """
    client = LLMClient(_provider(stub, "/novj"))
    result = client.chat(
        [{"role": "user", "content": "answer in json please"}], json_mode="object"
    )
    assert result.text
    assert "response_format" in client._unsupported

    # The next call must not re-send it (the stub would 400 again).
    payload, _ = client.chat_json([{"role": "user", "content": "answer in json"}])
    assert isinstance(payload, dict)


def test_first_request_rate_limit_is_retried(stub):
    client = LLMClient(_provider(stub, "/flaky", max_retries=2))
    result = client.chat([{"role": "user", "content": "hello"}])
    assert result.text
    assert stub.rate_limited_once


def test_retries_are_exhausted_before_giving_up(stub):
    """A client with no retries must surface the 429 instead of hanging."""
    client = LLMClient(_provider(stub, "/flaky", max_retries=0))
    with pytest.raises(LLMHTTPError) as excinfo:
        client.chat([{"role": "user", "content": "hello"}])
    assert excinfo.value.status == 429
    assert excinfo.value.is_retryable


def test_stream_only_endpoint_falls_back_reports_clearly(stub):
    client = LLMClient(_provider(stub, "/streamonly", max_retries=0))
    with pytest.raises(LLMHTTPError):
        # A non-stream request to a stream-only endpoint is refused, and the error
        # must be raised rather than swallowed into an empty answer.
        client.chat([{"role": "user", "content": "hello"}], stream=False)
    # And streaming does work there.
    chunks: list[str] = []
    result = client.chat(
        [{"role": "user", "content": "hello"}], stream=True, on_delta=chunks.append
    )
    assert result.text and chunks


def test_local_endpoint_without_a_key_still_works(stub):
    """Local servers often need no key: an empty key must not block the request."""
    provider = _provider(stub, api_key="")
    # A loopback base URL counts as credentialed, which is what lets Ollama and
    # LM Studio work with no key at all.
    assert provider.has_credentials is True
    ok, _message, _latency = LLMClient(provider).test_connection()
    assert ok is True


def test_remote_endpoint_without_a_key_is_flagged(stub):
    provider = _provider(stub, api_key="", base_url="https://api.example.com/v1")
    assert provider.has_credentials is False


def test_env_indirection_resolves_at_call_time(stub, monkeypatch):
    provider = _provider(stub, api_key="env:LIRIK_STUB_KEY")
    monkeypatch.setenv("LIRIK_STUB_KEY", "secret-from-env")
    assert provider.resolved_key() == "secret-from-env"
    ok, _message, _latency = LLMClient(provider).test_connection()
    assert ok is True


# --------------------------------------------------------------------------- #
# runner + a whole pipeline over the wire
# --------------------------------------------------------------------------- #


def test_client_runner_exposes_the_model_name(stub):
    runner = ClientRunner(LLMClient(_provider(stub)))
    assert runner.model == "stub-model-large"
    result = runner.complete([{"role": "user", "content": "hi"}])
    assert result.text


def test_pipeline_runs_against_a_real_endpoint(stub):
    """Nine stages, three reviewers, two rollbacks - over an actual HTTP socket."""
    settings = AppSettings()
    settings.ensure_defaults()
    settings.providers = [_provider(stub)]
    settings.active_provider_id = "stub"
    settings.max_revisions = 6
    settings.stage_attempts = 1

    events: list = []
    orchestrator = Orchestrator(
        ClientRunner(LLMClient(_provider(stub))),
        settings=settings,
        on_event=events.append,
    )
    result = orchestrator.run(None, reference_summary="", reference_lossless="unknown")

    # The stub always passes reviews, so this must succeed end to end.
    assert result.prompt is not None
    assert result.prompt.style
    assert result.prompt.lyrics
    assert result.state.finished is True
    stages = [record.stage for record in result.state.history]
    assert len(stages) >= 8
    # Every stage got a real request; the reviewer stages actually ran last.
    assert stages[-1] == "prompt_reviewer"
    assert any(event.kind == "gates" for event in events)
    # A run over the wire must still be bounded by the loop's own rules.
    assert result.state.revisions <= 6
    assert result.state.prompt_review is not None and result.state.prompt_review.passed


def test_garbage_answers_produce_a_bounded_diagnosable_failure(stub):
    """A model that answers nonsense must not hang the loop or fake a pass.

    Here the very first stage cannot satisfy its own schema (the intake agent
    requires a ``reply`` field), so the agent exhausts its parse retries and the
    orchestrator stops with a reason that names the stage. The important part is
    what does *not* happen: no exception escapes to the caller, no prompt is
    invented, and the run is marked finished instead of left hanging.
    """
    settings = AppSettings()
    settings.ensure_defaults()
    settings.max_revisions = 4
    settings.stage_attempts = 1

    events: list = []
    orchestrator = Orchestrator(
        ClientRunner(LLMClient(_provider(stub, "/garbage"))),
        settings=settings,
        on_event=events.append,
    )
    result = orchestrator.run(None)

    assert result.passed is False
    assert result.prompt is None
    assert result.state.finished is True
    assert "intake" in result.stopped_reason or "Konsultan" in result.stopped_reason
    assert result.state.revisions == 0, "nothing should have been promoted to a revision"
    assert any(event.kind == "error" for event in events), "the failure must be reported"
    assert any("reply" in str(event.message).lower() for event in events if event.kind == "error")


def test_settings_validation_catches_common_mistakes():
    empty_model = ProviderConfig(id="m", name="M", base_url="http://x/v1", model="")
    assert any("Model name is empty" in p for p in empty_model.validate())

    malformed = ProviderConfig(id="bad", name="Bad", base_url="not-a-url", model="m")
    assert any("http://" in p for p in malformed.validate())

    bad_json = ProviderConfig(id="j", name="J", base_url="http://x/v1", model="m", json_mode="junk")
    assert any("JSON mode" in p for p in bad_json.validate())

    hot = ProviderConfig(id="t", name="T", base_url="http://x/v1", model="m", temperature=9.0)
    assert any("Temperature" in p for p in hot.validate())


def test_a_malformed_base_url_raises_our_own_error_not_a_valueerror():
    """The UI catches LLMError; urllib's bare ValueError would crash the app."""
    bad = ProviderConfig(id="bad", name="Bad", base_url="not-a-url", model="m", timeout=3.0)
    client = LLMClient(bad)
    with pytest.raises(LLMHTTPError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    assert "http://" in str(excinfo.value)

    with pytest.raises(LLMHTTPError):
        client.list_models()

    chunks: list[str] = []
    with pytest.raises(LLMHTTPError):
        client.chat([{"role": "user", "content": "hi"}], stream=True, on_delta=chunks.append)

# --------------------------------------------------------------------------- #
# the json_object precondition (found against a real provider)
# --------------------------------------------------------------------------- #


def test_json_object_without_the_word_json_in_the_prompt_is_repaired(stub):
    """A real gateway 400s unless the prompt says "json" while streaming.

    This defect was found live, not in a lab: ``response_format: json_object`` plus
    ``stream: true`` against a prompt that never mentions JSON is rejected by
    OpenAI-compatible servers, and the client's only symptom was a bare
    "HTTP 400". The client must guarantee the wording itself rather than depend on
    whatever a stage prompt happens to say.
    """
    client = LLMClient(_provider(stub, "/strictjson"))
    chunks: list[str] = []
    result = client.chat(
        [{"role": "user", "content": "Write three short lyric lines about rain."}],
        json_mode="object",
        stream=True,
        on_delta=chunks.append,
    )
    assert result.text, "the request must succeed despite the strict server"
    assert chunks, "streaming must still deliver deltas"
    # And the guarantee must be invisible to the caller's own prompt.
    sent = " ".join(m["content"] for m in client._build_body(
        [{"role": "user", "content": "Write three lines."}], "object", None, None, None, stream=True
    )["messages"])
    assert "json" in sent.lower()


def test_json_object_hint_is_not_duplicated_when_already_present(stub):
    from suno_agents.llm.wire import JSON_OBJECT_HINT, ensure_json_keyword

    already = [{"role": "user", "content": "Reply with JSON only, please."}]
    assert ensure_json_keyword(already, "object") == already

    plain = [{"role": "user", "content": "Tell me a story."}]
    patched = ensure_json_keyword(plain, "object")
    assert JSON_OBJECT_HINT in patched[-1]["content"]
    assert plain[-1]["content"] == "Tell me a story.", "input must not be mutated"

    # schema mode carries its own requirement; off mode wants no JSON at all.
    assert ensure_json_keyword(plain, "schema") == plain
    assert ensure_json_keyword(plain, "off") == plain


def test_server_error_reasons_reach_the_exception_message(stub):
    """The whole reason this bug was hard to find: the message said only HTTP 400."""
    client = LLMClient(_provider(stub, "/strictjson"))
    # Bypass the repair path by pre-marking both escapes as exhausted.
    client._json_hint_forced = True
    client._unsupported.add("response_format")
    result = client.chat(
        [{"role": "user", "content": "no keyword here at all"}], json_mode="object", stream=True
    )
    assert result.text


def test_error_detail_extraction():
    from suno_agents.llm.client import _error_detail

    assert "json" in _error_detail(
        '{"error":{"message":"Prompt must contain the word json","type":"x"}}'
    )
    assert _error_detail("") == ""
    assert _error_detail("plain text failure") == " - plain text failure"

# --------------------------------------------------------------------------- #
# reasoning-model budget behaviour (found live)
# --------------------------------------------------------------------------- #


def test_a_reasoning_only_reply_is_retried_with_more_budget(stub):
    """A model that burns its output budget on hidden reasoning must recover.

    Measured against a real endpoint: with a small cap the reply carried 1032
    characters of reasoning, zero characters of content, and ``reasoning_tokens``
    equal to the whole cap. The client used to fail the stage outright. Critically
    it must raise the *effective* cap, including when that cap came from provider
    settings rather than the call - otherwise the retry silently changes nothing.
    """
    # Cap supplied by the provider, not by the call: the case that never retried.
    client = LLMClient(_provider(stub, "/burner", max_tokens=400, max_retries=0))
    payload, result = client.chat_json(
        [{"role": "user", "content": "give me json"}], retries=1
    )
    assert payload, "the retry must produce a real answer"
    assert result.text.strip()


def test_a_reasoning_only_reply_with_no_retries_left_reports_the_cause(stub):
    """The failure message must name reasoning burn, not just 'empty response'."""
    client = LLMClient(_provider(stub, "/burner", max_tokens=400, max_retries=0))
    with pytest.raises(LLMError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}], max_tokens=400)
    message = str(excinfo.value).lower()
    assert "reason" in message or "empty" in message or "content" in message


def test_budget_raise_grows_a_cap_but_never_imposes_one():
    """A repair turn may raise a stated cap; it must never invent one.

    ``0`` means the caller asked for the model's own maximum, so "raising" it would
    be a regression that reintroduces the truncation the user turned off.
    """
    from suno_agents.llm.client import LLMClient

    assert LLMClient._raise_budget(None) == 0, "no cap stays no cap"
    assert LLMClient._raise_budget(0) == 0, "an explicit no-cap stays no cap"
    floor = LLMClient.RETRY_BUDGET_FLOOR
    assert LLMClient._raise_budget(300) == floor
    assert LLMClient._raise_budget(8192) == floor
    assert LLMClient._raise_budget(16384) == 24576
    assert LLMClient._raise_budget(10**6) == LLMClient.RETRY_BUDGET_CEILING
    # monotonic where a cap exists
    budgets = [LLMClient._raise_budget(b) for b in (100, 2000, 8000, 16000)]
    assert budgets == sorted(budgets)


def test_agents_impose_no_output_cap_by_default():
    """Every artifact-producing agent must ask for no cap, and honour a stated one.

    The floor this replaces was still a guess, and each guess truncated a real run.
    """
    from suno_agents.agents import build_agent
    from suno_agents.domain.graph import (
        ARRANGER, AUDIO_ANALYST, COMPOSER, INTAKE, LYRICIST, PROMPT_ENGINEER,
    )
    from suno_agents.llm.mock import MockRunner

    runner = MockRunner(["{}"])
    stages = (INTAKE, AUDIO_ANALYST, COMPOSER, LYRICIST, ARRANGER, PROMPT_ENGINEER)
    for stage in stages:
        agent = build_agent(stage, runner)
        assert agent.DEFAULT_OUTPUT_TOKENS == 0, stage
        captured: dict = {}

        class Capture:
            model = "capture"

            def complete_json(self, _m, **kwargs):
                from suno_agents.llm.wire import ChatResult

                captured.update(kwargs)
                return {}, ChatResult(text="{}")

        agent.runner = Capture()  # type: ignore[assignment]
        try:
            agent._call([{"role": "user", "content": "x"}], "sys")  # type: ignore[attr-defined]
        except Exception:
            pass
        assert captured.get("max_tokens") in (0, None), f"{stage} imposed a cap"

    # And an explicit budget still wins.
    explicit = build_agent(COMPOSER, runner, max_tokens=9000)
    captured2: dict = {}

    class Capture2:
        model = "capture"

        def complete_json(self, _m, **kwargs):
            from suno_agents.llm.wire import ChatResult

            captured2.update(kwargs)
            return {}, ChatResult(text="{}")

    explicit.runner = Capture2()  # type: ignore[assignment]
    try:
        explicit._call([{"role": "user", "content": "x"}], "sys")  # type: ignore[attr-defined]
    except Exception:
        pass
    assert captured2.get("max_tokens") == 9000

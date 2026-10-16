"""The agent log: recording, filtering, export, and what actually reaches it.

The feature exists to answer "what did we send, what came back, and why did the
agent decide that?", so the tests are written against those three questions rather
than against the data structures:

* a real HTTP call - recorded from the wire, including a failure and a retry;
* agent decisions - parsed, repair, accepted-with-problems, failed;
* the log must survive its own bugs (a logging failure must not fail a run);
* the exports must contain what they claim (JSONL keeps every character).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from support.openai_stub import StubProvider  # noqa: E402

from suno_agents.agents import build_agent  # noqa: E402
from suno_agents.config.models import AppSettings, ProviderConfig  # noqa: E402
from suno_agents.domain.graph import COMPOSER, INTAKE, PROMPT_REVIEWER  # noqa: E402
from suno_agents.domain.models import PipelineState  # noqa: E402
from suno_agents.llm import LLMClient  # noqa: E402
from suno_agents.llm.mock import MockRunner  # noqa: E402
from suno_agents.llm.runner import ClientRunner  # noqa: E402
from suno_agents.llm.trace import (  # noqa: E402
    AgentDecision,
    LLMExchange,
    MAX_ENTRIES,
    TraceRecorder,
    filter_entries,
    summarise,
)
from suno_agents.llm.wire import LLMParseError  # noqa: E402
from suno_agents.pipeline.orchestrator import Orchestrator  # noqa: E402
from suno_agents.testing.demo import build_demo_runner  # noqa: E402


@pytest.fixture
def stub():
    with StubProvider() as provider:
        yield provider


def _provider(stub: StubProvider, prefix: str = "/v1", **overrides) -> ProviderConfig:
    data = {
        "id": "stub", "name": "Stub", "base_url": stub.base_url(prefix),
        "api_key": "test", "model": "stub-model-large", "timeout": 20.0, "max_retries": 0,
    }
    data.update(overrides)
    return ProviderConfig.from_dict(data)


# --------------------------------------------------------------------------- #
# recording: from the wire
# --------------------------------------------------------------------------- #


def test_a_real_call_is_recorded_with_prompt_and_reply(stub):
    trace = TraceRecorder()
    client = LLMClient(_provider(stub), trace=trace, label="composer#1")
    result = client.chat(
        [{"role": "user", "content": "balas JSON"}], system="kamu composer", json_mode="object"
    )
    assert result.text

    entries = trace.entries()
    assert len(entries) == 1
    exchange = entries[0]
    assert isinstance(exchange, LLMExchange)
    assert exchange.stage == "composer", "the stage must be derived from the label"
    assert exchange.label == "composer#1"
    assert exchange.model == result.model
    assert exchange.status == "ok"
    assert exchange.total_tokens == 126
    # The whole request is kept, system prompt first - that is the point of the log.
    assert [m["role"] for m in exchange.messages][0] == "system"
    assert "kamu composer" in exchange.system_prompt
    assert "balas JSON" in exchange.user_message
    assert exchange.response
    assert exchange.request_chars > 0


def test_a_failed_call_is_recorded_with_its_reason(stub):
    trace = TraceRecorder()
    client = LLMClient(_provider(stub, "/burner", max_tokens=400), trace=trace, label="intake#2")
    from suno_agents.llm.wire import LLMError

    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}], max_tokens=400)

    entry = trace.entries()[0]
    assert entry.status in {"empty", "error"}
    assert entry.error, "a failed exchange must say why"
    assert "reasoning" in entry.error or "token" in entry.error or "Balasan" in entry.error
    assert entry.is_problem is True


def test_every_retry_is_recorded_separately(stub):
    """Retries are where the interesting failures live, so each attempt is an entry."""
    trace = TraceRecorder()
    client = LLMClient(_provider(stub, "/flaky", max_retries=2), trace=trace, label="intake#1")
    client.chat([{"role": "user", "content": "hello"}])
    statuses = [e.status for e in trace.entries() if e.kind == "exchange"]
    assert "error" in statuses, f"the 429 attempt must be logged: {statuses}"
    assert "ok" in statuses, f"the successful retry must be logged: {statuses}"
    assert len(statuses) >= 2


def test_streaming_is_recorded(stub):
    trace = TraceRecorder()
    client = LLMClient(_provider(stub), trace=trace, label="lyricist#1")
    chunks: list[str] = []
    client.chat([{"role": "user", "content": "stream"}], stream=True, on_delta=chunks.append)
    entry = trace.entries()[0]
    assert entry.stream is True
    assert entry.response, "a streamed reply must be stored in full"
    assert "".join(chunks) == entry.response


def test_a_client_without_a_recorder_behaves_exactly_as_before(stub):
    client = LLMClient(_provider(stub))
    assert client.trace is None
    assert client.chat([{"role": "user", "content": "hi"}]).text


def test_the_recorder_cannot_break_a_call(stub):
    """Logging is best-effort: a broken recorder must not fail an otherwise fine call."""

    class Exploding(TraceRecorder):
        def record(self, entry):  # type: ignore[override]
            raise RuntimeError("disk on fire")

    client = LLMClient(_provider(stub), trace=Exploding(), label="x#1")
    assert client.chat([{"role": "user", "content": "hi"}]).text


# --------------------------------------------------------------------------- #
# recording: agent decisions
# --------------------------------------------------------------------------- #


def test_a_clean_agent_run_records_reply_and_parsed(stub):
    trace = TraceRecorder()
    settings = AppSettings()
    settings.providers = [_provider(stub)]
    settings.active_provider_id = "stub"
    settings.ensure_defaults()
    runner = ClientRunner(LLMClient(settings.providers[0], trace=trace), settings=settings)
    agent = build_agent(COMPOSER, runner, trace=trace)
    agent.run(PipelineState(), [])

    outcomes = [e.outcome for e in trace.entries() if e.kind == "decision"]
    assert outcomes[0] == "reply"
    assert "parsed" in outcomes
    # And the traffic it produced is attributed to the same stage.
    stages = {e.stage for e in trace.entries() if e.kind == "exchange"}
    assert stages == {"composer"}


def test_a_repair_round_is_recorded_with_the_problems():
    """The repair loop is a decision worth seeing: it explains an extra generation."""
    from suno_agents.agents.composer import ComposerAgent
    from suno_agents.domain.models import MusicBlueprint

    class PickyComposer(ComposerAgent):
        """A composer whose validator is deliberately unsatisfiable once."""

        parse_retries = 1

        def validate(self, artifact: MusicBlueprint) -> list[str]:
            return ["sections missing harmony"]

    trace = TraceRecorder()
    runner = MockRunner(
        ['{"key":"C","mode":"major","sections":[],"melodic_motif":"5-6-5-3","hook_definition":"bar 5"}']
    )
    PickyComposer(runner, trace=trace).run(PipelineState(), [])

    decisions = [e for e in trace.entries() if e.kind == "decision"]
    repairs = [d for d in decisions if d.outcome == "repair"]
    assert repairs, [d.outcome for d in decisions]
    assert repairs[0].problems == ["sections missing harmony"]
    assert repairs[0].attempt == 1
    # The second pass accepts it with the problem still on record, rather than
    # silently pretending the warning never happened.
    assert any(d.outcome == "accepted_with_problems" for d in decisions)


def test_a_total_failure_is_recorded():
    from suno_agents.agents.composer import ComposerAgent
    from suno_agents.agents.base import AgentError

    trace = TraceRecorder()
    runner = MockRunner(["tidak ada json di sini"])

    class Strict(ComposerAgent):
        parse_retries = 1

    # Either error type is a legitimate end state; what matters is that the log
    # records why the agent gave up rather than going quiet.
    with pytest.raises((AgentError, LLMParseError)):
        Strict(runner, trace=trace).run(PipelineState(), [])
    decisions = [e for e in trace.entries() if e.kind == "decision"]
    assert decisions, "a failure must leave a decision behind"
    assert decisions[-1].outcome == "failed"
    # The detail must carry the cause, not just the fact of failure.
    assert "JSON" in decisions[-1].detail or "json" in decisions[-1].detail
    assert decisions[-1].problems, "the raw error text must be kept"
    assert decisions[-1].attempt == 2, "two attempts were made"


def test_gate_verdicts_appear_in_the_log():
    trace = TraceRecorder()
    Orchestrator(build_demo_runner(trace=trace), settings=None, trace=trace).run(None)
    gates = [
        e for e in trace.entries()
        if e.kind == "decision" and e.outcome == "gate"
    ]
    # The demo script fails twice on purpose, so five gate verdicts are produced:
    # one per reviewer pass, with the two failures rerunning their reviewer.
    assert len(gates) == 5, [g.detail for g in gates]
    assert all(g.stage in {"lyric_reviewer", "music_reviewer", "prompt_reviewer"} for g in gates)
    assert any("mundur ke" in g.detail for g in gates), "a rollback must be visible"
    assert any("gate keras gagal" in g.detail for g in gates)


# --------------------------------------------------------------------------- #
# the log as a whole
# --------------------------------------------------------------------------- #


def test_an_offline_run_produces_a_log_with_traffic_and_decisions():
    """Offline demo mode must populate the tab too, not just decisions."""
    trace = TraceRecorder()
    result = Orchestrator(build_demo_runner(trace=trace), settings=None, trace=trace).run(None)
    assert result.passed
    exchanges = [e for e in trace.entries() if e.kind == "exchange"]
    decisions = [e for e in trace.entries() if e.kind == "decision"]
    assert len(exchanges) >= 9, f"one per stage at least: {len(exchanges)}"
    assert len(decisions) >= 9
    # Every stage that ran must be attributable.
    stages = {e.stage for e in exchanges}
    assert {"intake", "composer", "lyricist", "prompt_reviewer"} <= stages
    assert "permintaan" in summarise(trace.entries())


def test_the_log_is_bounded():
    trace = TraceRecorder(max_entries=5)
    for index in range(20):
        trace.note(f"catatan {index}")
    assert len(trace) == 5
    assert trace.entries()[-1].detail == "catatan 19", "the newest survive, not the oldest"


def test_incremental_reads_never_repeat_entries():
    trace = TraceRecorder()
    trace.note("satu")
    first, total = trace.since(0)
    assert len(first) == 1 and total == 1
    trace.note("dua")
    second, total = trace.since(total)
    assert [e.detail for e in second] == ["dua"], "a poll only sees what is new"
    assert total == 2
    assert trace.since(total)[0] == []


def test_filters_select_the_right_entries():
    trace = TraceRecorder()
    trace.record_exchange(stage="composer", model="m", status="ok")
    trace.record_exchange(stage="lyricist", model="m", status="empty", error="kosong")
    trace.note("catatan", stage="composer")
    entries = trace.entries()

    assert len(filter_entries(entries, "all")) == 3
    assert len(filter_entries(entries, "exchanges")) == 2
    assert len(filter_entries(entries, "decisions")) == 1
    problems = filter_entries(entries, "problems")
    assert len(problems) == 1 and problems[0].stage == "lyricist"
    assert len(filter_entries(entries, "all", "composer")) == 2
    assert filter_entries(entries, "exchanges", "lyricist")[0].status == "empty"


def test_exports_carry_what_they_promise(tmp_path):
    trace = TraceRecorder()
    long_reply = "x" * 50_000
    trace.record_exchange(
        stage="composer", model="m", provider="p", url="u", label="composer#1",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
        response=long_reply, status="ok", total_tokens=12,
    )
    trace.decide(stage="composer", attempt=1, outcome="repair", detail="perbaiki", problems=["p1"])

    jsonl = trace.to_jsonl()
    lines = [json.loads(line) for line in jsonl.strip().split("\n")]
    assert len(lines) == 2
    exchange = lines[0]
    assert exchange["kind"] == "exchange"
    assert exchange["status"] == "ok"
    # JSONL is the machine copy: the full reply is preserved, not clipped.
    assert exchange["response"] == long_reply
    assert lines[1]["problems"] == ["p1"]

    markdown = trace.to_markdown()
    assert "# Log agen" in markdown
    assert "System prompt" in markdown
    assert "Pesan user" in markdown
    assert "Balasan" in markdown
    assert "perbaiki" in markdown


def test_a_huge_field_is_clipped_and_says_so():
    trace = TraceRecorder()
    trace.record_exchange(
        stage="x", model="m", messages=[], response="y" * 300_000, status="ok"
    )
    from suno_agents.llm.trace import MAX_FIELD_CHARS

    entry = trace.entries()[0]
    # The recorder stores what it is given; the markdown renderer is what clips.
    assert len(entry.response) == 300_000
    rendered = trace.to_markdown()
    assert "chars omitted" in rendered
    assert len(rendered) < 300_000
    assert MAX_FIELD_CHARS == 200_000


def test_clearing_empties_the_log():
    trace = TraceRecorder()
    trace.note("x")
    assert len(trace) == 1
    trace.clear()
    assert len(trace) == 0
    assert trace.counts() == {"exchange": 0, "decision": 0, "problems": 0}


def test_run_outputs_include_the_agent_log():
    trace = TraceRecorder()
    orchestrator = Orchestrator(build_demo_runner(trace=trace), settings=None, trace=trace)
    orchestrator.run(None)
    files = orchestrator.output_files()
    assert "agent_log.jsonl" in files
    assert "agent_log.md" in files
    rows = [json.loads(line) for line in files["agent_log.jsonl"].strip().split("\n")]
    assert any(row["kind"] == "exchange" for row in rows)
    assert any(row["kind"] == "decision" for row in rows)


def test_the_session_owns_one_recorder_shared_by_clients_and_agents(stub):
    """A single recorder is what lets the tab show traffic and decisions together."""
    home = {"LIRIK_HOME": os.environ.get("LIRIK_HOME")}
    try:
        import tempfile

        os.environ["LIRIK_HOME"] = tempfile.mkdtemp(prefix="lirik-trace-session-")
        from suno_agents.config.store import SettingsStore
        from suno_agents.pipeline.session import StudioSession

        session = StudioSession()
        settings = session.settings
        settings.providers = [_provider(stub)]
        settings.active_provider_id = "stub"
        settings.ensure_defaults()
        SettingsStore().save(settings)
        session.reload_settings()

        runner = session.build_runner()
        assert getattr(runner, "_client", None) is not None
        assert runner._client.trace is session.trace, "the client must log to the session"

        agent = build_agent(INTAKE, runner, trace=session.trace)
        assert agent.trace is session.trace
    finally:
        if home["LIRIK_HOME"] is None:
            os.environ.pop("LIRIK_HOME", None)
        else:
            os.environ["LIRIK_HOME"] = home["LIRIK_HOME"]


def test_client_label_survives_per_stage_routing(stub):
    """A routed client must still write into the shared log with its own label."""
    trace = TraceRecorder()
    settings = AppSettings()
    settings.providers = [_provider(stub)]
    settings.active_provider_id = "stub"
    settings.ensure_defaults()
    settings.stage_overrides["composer"] = type(
        settings.stage_overrides["composer"]
    )(provider_id="stub", model="stub-model-small", enabled=True)

    runner = ClientRunner(LLMClient(settings.providers[0], trace=trace), settings=settings)
    stage_runner = runner.for_stage("composer")
    assert stage_runner is not None
    agent = build_agent(COMPOSER, stage_runner, trace=trace)
    agent.run(PipelineState(), [])

    exchanges = [e for e in trace.entries() if e.kind == "exchange"]
    assert exchanges, "a routed client must still record its traffic"
    assert exchanges[0].model == "stub-model-small"
    assert exchanges[0].stage == "composer"

# --------------------------------------------------------------------------- #
# a rejected gate verdict is a problem worth surfacing
# --------------------------------------------------------------------------- #


def test_a_rejected_gate_verdict_counts_as_a_problem():
    """The entry someone opens the log to find: a reviewer sent the work back."""
    trace = TraceRecorder()
    trace.decide(stage="music_reviewer", attempt=1, outcome="gate",
                 detail="pass 92.9/100 - 0 gate keras gagal, 0 gate lunak")
    trace.decide(stage="music_reviewer", attempt=2, outcome="gate",
                 detail="revise 76.6/100 - 1 gate keras gagal, 0 gate lunak; mundur ke arranger")
    entries = trace.entries()
    assert entries[0].is_problem is False, "a pass is not a problem"
    assert entries[1].is_problem is True, "a rejection must be findable"
    problems = filter_entries(entries, "problems")
    assert [p.detail.split()[0] for p in problems] == ["revise"]
    assert trace.counts()["problems"] == 1


def test_panel_clear_does_not_resurrect_history():
    """Clearing a view must not be undone by the next poll.

    The panel reads new records by index, so resetting that index to zero made the
    following refresh re-read every entry the recorder still held: "clear" visibly
    undid itself a moment later.
    """
    from suno_agents.ui.agentlog import AgentLogPanel

    trace = TraceRecorder()
    for index in range(3):
        trace.note(f"catatan {index}")

    panel = AgentLogPanel.__new__(AgentLogPanel)
    panel._source = lambda cursor: trace.since(cursor)
    panel._all = []
    panel._cursor = 0
    panel._known_total = 0
    panel._mode = "all"
    panel._stage = ""

    class FakeList:
        def __init__(self):
            self.entries = []

        def set_entries(self, entries):
            self.entries = list(entries)

        def reset(self):
            self.entries = []

        def select(self, _index):
            pass

    class FakeLabel:
        def configure(self, **_kwargs):
            pass

    panel.list = FakeList()
    panel.summary = FakeLabel()
    panel.detail_show = lambda _entry=None: None

    AgentLogPanel.refresh(panel, force=True)
    assert len(panel._all) == 3

    AgentLogPanel.clear(panel)
    assert panel.list.entries == []
    # The next poll must not bring them back.
    AgentLogPanel.refresh(panel)
    assert panel.list.entries == [], "clear() was undone by the next refresh"
    assert panel._all == []

    # And a genuinely new record still appears.
    trace.note("catatan baru")
    AgentLogPanel.refresh(panel)
    assert [e.detail for e in panel.list.entries] == ["catatan baru"]

    # forget() is for when the recorder itself was emptied.
    trace.clear()
    AgentLogPanel.clear(panel)
    AgentLogPanel.refresh(panel, force=False)
    assert panel._all == []


def test_clearing_the_recorder_and_forgetting_the_panel_stays_empty():
    from suno_agents.ui.agentlog import AgentLogPanel

    trace = TraceRecorder()
    trace.note("x")
    panel = AgentLogPanel.__new__(AgentLogPanel)
    panel._source = lambda cursor: trace.since(cursor)
    panel._all = []
    panel._cursor = 0
    panel._known_total = 0
    panel._mode = "all"
    panel._stage = ""

    class FakeList:
        def __init__(self):
            self.entries = []

        def set_entries(self, entries):
            self.entries = list(entries)

        def reset(self):
            self.entries = []

        def select(self, _index):
            pass

    class FakeLabel:
        def configure(self, **_kwargs):
            pass

    panel.list = FakeList()
    panel.summary = FakeLabel()
    panel.detail_show = lambda _entry=None: None

    AgentLogPanel.refresh(panel, force=True)
    trace.clear()
    AgentLogPanel.forget(panel)
    AgentLogPanel.refresh(panel)
    assert panel._all == []
    assert len(trace) == 0

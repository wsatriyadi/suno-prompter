"""Drive a real offline run and inspect the rendered Log Agen tab.

Run as a subprocess (see ``tests/test_ui.py``). Asserts the tab is not merely
present but useful: that request/reply pairs and agent decisions both appear, that
the filters select the right rows, that the detail pane shows the actual prompt
text and reply, and that exporting writes the two files it promises.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)


def fail(message: str) -> int:
    print(f"ERROR: {message}")
    return 1


def walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from walk(child)


def texts_in(widget) -> list[str]:
    return [
        child.get("1.0", "end").strip()
        for child in walk(widget)
        if type(child).__name__ == "Text"
    ]


def main() -> int:
    home = tempfile.mkdtemp(prefix="lirik-log-ui-")
    os.environ["LIRIK_HOME"] = home
    try:
        from suno_agents.ui.agentlog import LogDetail, AgentLogPanel
        from suno_agents.ui.app import StudioApp

        app = StudioApp(offline=True)
        app.withdraw()
        app.update()

        if "log" not in app.view_buttons:
            return fail(f"no Log Agen tab: {list(app.view_buttons)}")
        app._show_right("log")
        app.update()
        panel = app.agent_log
        if not isinstance(panel, AgentLogPanel):
            return fail("the log tab is not the agent log panel")

        # Run the pipeline and let the UI pump deliver the records.
        app.session.settings.max_revisions = 12
        app._start_run()
        deadline = time.time() + 120
        while app.session.result is None and time.time() < deadline:
            app.update()
            app._drain_once()
            time.sleep(0.01)
        app.update()
        app._drain_once()
        app.update()

        if app.session.result is None:
            return fail("the offline pipeline never finished")

        recorder = app.session.trace
        if len(recorder) < 10:
            return fail(f"only {len(recorder)} log entries recorded")

        # 1. Both kinds are present.
        counts = recorder.counts()
        if counts["exchange"] < 9:
            return fail(f"expected one request per stage at least: {counts}")
        if counts["decision"] < 9:
            return fail(f"expected decisions to be logged too: {counts}")

        # 2. Filters select the right rows.
        expectations = {
            "all": counts["exchange"] + counts["decision"],
            "exchanges": counts["exchange"],
            "decisions": counts["decision"],
            "problems": counts["problems"],
        }
        for mode, expected in expectations.items():
            panel.set_filter(mode)
            app.update()
            rows = len(panel.list._rows)
            if rows != expected:
                return fail(f"filter {mode}: {rows} rows, expected {expected}")

        # 3. A stage filter narrows further.
        panel.set_filter("all")
        panel.stage_menu["menu"].invoke(2)  # second stage in the menu
        app.update()
        if len(panel.list._rows) >= expectations["all"]:
            return fail("the stage filter did not narrow the list")

        # 4. The detail pane shows the real request and the real reply.
        panel.stage_var.set("semua tahap")
        panel.set_stage("semua tahap")
        panel.set_filter("exchanges")
        app.update()
        if not panel.list._rows:
            return fail("no request rows to inspect")
        panel.list.select(0)
        app.update()
        detail = next((c for c in walk(panel) if isinstance(c, LogDetail)), None)
        if detail is None:
            return fail("the detail pane is missing")
        blocks = texts_in(detail)
        if len(blocks) < 3:
            return fail(f"detail shows only {len(blocks)} blocks: {[b[:40] for b in blocks]}")
        joined = "\n".join(blocks)
        if "You are one specialist inside LIRIK STUDIO" not in joined:
            return fail("the system prompt that was sent is not shown")
        if '"reply"' not in joined and '"key"' not in joined and '"title"' not in joined:
            return fail("the model reply is not shown")
        status = detail.status_badge._text
        if status not in {"ok", "truncated", "empty", "error"}:
            return fail(f"unexpected status badge for a request entry: {status}")

        # 5. A decision entry explains itself.
        panel.set_filter("decisions")
        app.update()
        gate_index = next(
            (
                index
                for index, entry in enumerate(panel.list._entries)
                if getattr(entry, "outcome", "") == "gate"
            ),
            None,
        )
        if gate_index is None:
            return fail("no gate decision in the log")
        panel.list.select(gate_index)
        app.update()
        gate_blocks = texts_in(detail)
        if not gate_blocks or not any("gate" in b.lower() for b in gate_blocks):
            return fail(f"the gate decision does not explain itself: {gate_blocks[:1]}")

        # 6. Meaningful runs put problems in the log (the demo rollback is one).
        panel.set_filter("problems")
        app.update()
        if not panel.list._rows:
            return fail("the demo run rolled back but logged no problem entry")

        # 7. Export writes both files.
        run_dir = app._run_dir
        if run_dir is None:
            return fail("no run directory")
        if not (run_dir / "agent_log.jsonl").exists():
            return fail("the run directory has no agent_log.jsonl")
        if not (run_dir / "agent_log.md").exists():
            return fail("the run directory has no agent_log.md")
        rows = [
            json.loads(line)
            for line in (run_dir / "agent_log.jsonl").read_text(encoding="utf-8").strip().split("\n")
        ]
        if not any(r["kind"] == "exchange" for r in rows):
            return fail("agent_log.jsonl has no exchanges")
        if not any(r["kind"] == "decision" for r in rows):
            return fail("agent_log.jsonl has no decisions")
        markdown = (run_dir / "agent_log.md").read_text(encoding="utf-8")
        if "System prompt" not in markdown or "Balasan" not in markdown:
            return fail("agent_log.md is missing the request/reply sections")

        # 8a. Panel-level clear hides history without resurrecting it on the next
        #     poll - the subtle failure mode of resetting the read cursor.
        before = len(recorder)
        panel.clear()
        app.update()
        app._drain_once()
        app.update()
        if len(panel.list._rows) != 0:
            return fail(
                f"panel clear resurrected {len(panel.list._rows)} entries on the next poll"
            )
        if len(recorder) != before:
            return fail("panel clear must not touch the recorder itself")

        # 8b. The user-facing clear empties both, with the confirmation stubbed.
        import suno_agents.ui.app as app_module

        app_module.messagebox.askyesno = lambda *a, **k: True
        app._clear_log()
        app.update()
        app._drain_once()
        app.update()
        if len(panel.list._rows) != 0 or len(recorder) != 0:
            return fail(
                f"clearing the log did not empty it: rows={len(panel.list._rows)} "
                f"recorder={len(recorder)}"
            )
        if "kosong" not in panel.summary.cget("text"):
            return fail(f"summary after clear reads {panel.summary.cget('text')!r}")

        try:
            app.destroy()
        except tk.TclError:
            pass
        print("OK")
        return 0
    finally:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

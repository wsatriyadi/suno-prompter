"""Run the main window for real: build, start the offline pipeline, finish it.

Run as a subprocess (see ``tests/test_ui.py``). Asserts the wiring end to end:
the rail reflects every stage, the prompt panel holds the finished prompt, and
every artifact lands on disk.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)


def fail(message: str) -> int:
    print(f"ERROR: {message}")
    return 1


def main() -> int:
    home = tempfile.mkdtemp(prefix="lirik-run-")
    os.environ["LIRIK_HOME"] = home
    try:
        from suno_agents.domain.graph import STAGE_ORDER
        from suno_agents.ui.app import StudioApp

        app = StudioApp(offline=True)
        app.withdraw()
        app.update()
        app.session.settings.max_revisions = 12
        app._start_run()

        deadline = time.time() + 120
        while app.session.result is None and time.time() < deadline:
            app.update()
            app._drain_once()
            time.sleep(0.005)
        # one more pass so the terminal events reach the widgets
        app.update()
        app._drain_once()
        app.update()

        result = app.session.result
        if result is None:
            return fail("the offline pipeline never finished")
        if not result.passed:
            return fail(f"pipeline finished without passing: {result.stopped_reason}")
        if result.state.revisions != 2:
            return fail(f"expected 2 rollbacks, got {result.state.revisions}")

        # rail state
        for stage in STAGE_ORDER:
            status = app.rail._status.get(stage)
            if status != "done":
                return fail(f"rail status for {stage} is {status!r}, expected 'done'")

        # prompt panel state
        panel = app.prompt_panel
        if panel.prompt is None:
            return fail("prompt panel has no prompt")
        if panel.title_box.get_value() != panel.prompt.title:
            return fail("title field does not match the prompt")
        if panel.style_box.get_value() != panel.prompt.style:
            return fail("style field does not match the prompt")
        if panel.exclude_box.get_value() != panel.prompt.exclude_style_text:
            return fail("exclude field does not match the prompt")
        if "[" not in panel.lyrics_box.get_value():
            return fail("lyrics field lost its section tags")
        if panel.review_badge._text.startswith("belum"):
            return fail("review badge was never updated")
        if not panel.style_box.counter.cget("text").endswith("/ 1000"):
            return fail("style counter is not showing the Suno limit")

        # inspector: reviewers must expose scores and issues
        app._show_right("inspect")
        app._select_stage("prompt_reviewer")
        app.update()
        verdict = result.state.prompt_review
        if verdict is None or not verdict.dimensions:
            return fail("final prompt review has no dimension scores")
        if len(verdict.dimensions) < 8:
            return fail(f"only {len(verdict.dimensions)} dimensions scored")
        if app.inspector.body_text.get("1.0", "end").strip() == "":
            return fail("inspector rendered nothing for the prompt reviewer")

        app._select_stage("music_reviewer")
        app.update()
        if "VERDICT" not in app.inspector.body_text.get("1.0", "end"):
            return fail("inspector did not render the music review verdict")

        # artifacts
        run_dir = app._run_dir
        if run_dir is None or not run_dir.exists():
            return fail("no run directory was created")
        for name in (
            "state.json",
            "report.md",
            "suno_prompt.txt",
            "title.txt",
            "style.txt",
            "exclude.txt",
            "lyrics.txt",
            "lossless_checklist.md",
            "lyrics_full.md",
            "music_blueprint.md",
            "arrangement.md",
            "brief.md",
            "review_log.json",
        ):
            if not (run_dir / name).exists():
                return fail(f"missing artifact: {name}")

        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        if not state.get("success"):
            return fail("state.json does not record success")
        if len(state.get("review_log", [])) != 5:
            return fail(f"review log has {len(state.get('review_log', []))} entries, expected 5")
        # The two scripted failures must be visible in the log.
        failed = [v for v in state["review_log"] if not v["passed"]]
        if len(failed) != 2:
            return fail(f"expected 2 failed verdicts in the log, got {len(failed)}")
        if not any("MG8_alignment" in v.get("gates_failed", []) for v in failed):
            return fail("the alignment gate failure is not recorded")
        if not any("PG8_vocal_consistency" in v.get("gates_failed", []) for v in failed):
            return fail("the vocal-lock gate failure is not recorded")

        # the prompt file must be paste-ready in Suno's field order
        text = (run_dir / "suno_prompt.txt").read_text(encoding="utf-8")
        for heading in ("TITLE", "STYLE OF MUSIC", "EXCLUDE STYLES", "LYRICS"):
            if heading not in text:
                return fail(f"suno_prompt.txt is missing the {heading} block")

        print("OK")
        return 0
    finally:
        try:
            app.destroy()  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

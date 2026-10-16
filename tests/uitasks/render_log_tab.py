"""Render the Log Agen tab with a real run and a real selection, then screenshot.

Run as a subprocess. Two purposes: prove the tab paints the right content in a
visible window (assertions alone can pass on a blank canvas), and leave a
screenshot a human can look at.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "lirik-log.png")


def main() -> int:
    home = tempfile.mkdtemp(prefix="lirik-log-shot-")
    os.environ["LIRIK_HOME"] = home
    from suno_agents.ui.app import StudioApp

    app = StudioApp(offline=True)
    app.geometry("1560x940+20+20")
    try:
        app.update()
        app.deiconify()
        app.update()

        app.session.settings.max_revisions = 12
        app._start_run()
        deadline = time.time() + 120
        while app.session.result is None and time.time() < deadline:
            app.update()
            app._drain_once()
            time.sleep(0.005)
        app.update()
        app._drain_once()
        app.update()

        if app.session.result is None:
            print("ERROR: pipeline did not finish")
            return 2

        app._show_right("log")
        app.update()
        # Select an exchange so the detail pane shows a real prompt and reply.
        app.agent_log.set_filter("exchanges")
        app.update()
        if app.agent_log.list._rows:
            # Pick a middle one: the composer, whose prompt is the interesting one.
            index = min(2, len(app.agent_log.list._rows) - 1)
            app.agent_log.list.select(index)
        app.update()

        app.lift()
        app.attributes("-topmost", True)
        for _ in range(6):
            app.update()
            time.sleep(0.08)

        summary = app.agent_log.summary.cget("text")
        print(f"SUMMARY {summary}")
        print(f"ROWS {len(app.agent_log.list._rows)}")

        try:
            from PIL import ImageGrab

            image = ImageGrab.grab()
            image.save(OUT)
            print(f"SCREENSHOT {OUT} {image.size}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: screenshot failed: {type(exc).__name__}: {exc}")
            return 3
        return 0
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

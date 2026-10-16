"""Reproduce and verify the consultant-turn failure path.

The reported traceback was:

    NameError: cannot access free variable 'exc' where it is not associated with
    a value in enclosing scope

It happened when a chat turn failed, because the worker thread built a callback
with ``self.after(0, lambda: self._after_intake_error(exc))`` inside an
``except ... as exc`` block. Python unbinds ``exc`` when that block exits, so by
the time the callback ran the name was gone - meaning the *error* path crashed
instead of telling the user what went wrong. Calling ``after()`` from a worker is
also unsafe in Tk, so the fix removes both problems by passing results through a
queue drained on the UI thread.

This script drives the real window with a deliberately failing provider and
asserts a readable message appears instead of a callback traceback.
"""

from __future__ import annotations

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


def main() -> int:
    home = tempfile.mkdtemp(prefix="lirik-intake-")
    os.environ["LIRIK_HOME"] = home
    captured: list[str] = []
    try:
        from suno_agents.ui.app import StudioApp

        app = StudioApp()
        app.withdraw()
        app.update()
        # Swallow the modal dialog so the test can continue.
        import suno_agents.ui.app as app_module

        app_module.messagebox.showerror = lambda *a, **k: captured.append(str(a[1] if len(a) > 1 else a))

        # Force the consultant turn to fail the way a dead endpoint does.
        def explode(*_args, **_kwargs):
            raise RuntimeError("endpoint unreachable: connection refused")

        app.session.send_user_message = explode  # type: ignore[assignment]

        failures: list[str] = []
        original_report = app.report_callback_exception

        def report(exc_type, exc_value, exc_tb):
            failures.append(f"{exc_type.__name__}: {exc_value}")
            original_report(exc_type, exc_value, exc_tb)

        app.report_callback_exception = report  # type: ignore[assignment]

        app.chat_panel.composer.insert_text("halo, buatkan lagu galau")
        app._on_send("halo, buatkan lagu galau")

        deadline = time.time() + 15
        while time.time() < deadline and not captured:
            app.update()
            app._drain_once()
            time.sleep(0.02)
        app.update()

        transcript = app.chat_panel.view.get_text()
        if failures:
            return fail(f"a Tk callback raised: {failures[0]}")
        if "NameError" in transcript:
            return fail("the NameError still reaches the transcript")
        if "Gagal menghubungi model" not in transcript:
            return fail(f"no readable error in the transcript:\n{transcript[-400:]}")
        if "endpoint unreachable" not in transcript:
            return fail("the underlying cause is missing from the message")
        if not captured:
            return fail("the user was never shown a dialog")
        if "Periksa Pengaturan" not in captured[0]:
            return fail(f"the dialog does not say what to do: {captured[0]!r}")
        # The busy state must be released, or the composer stays locked forever.
        if app.chat_panel.run_button.enabled is False:
            return fail("the UI stayed busy after a failed turn")
        if app.chat_panel.composer.text.cget("state") == "disabled":
            return fail("the composer stayed disabled after a failed turn")

        # And a later successful turn must still work, proving the queue drains.
        app.session.send_user_message = lambda text: {
            "reply": f"Baik, aku catat: {text}",
            "ready_to_build": False,
            "brief": None,
            "missing_information": [],
            "suggested_choices": [],
        }
        app._on_send("coba lagi")
        deadline = time.time() + 15
        while time.time() < deadline and "aku catat" not in app.chat_panel.view.get_text():
            app.update()
            app._drain_once()
            time.sleep(0.02)
        if "aku catat" not in app.chat_panel.view.get_text():
            return fail("a successful turn after a failure never rendered")

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

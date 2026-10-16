"""Launcher used by the packaged executables.

PyInstaller needs a real script as its entry point. The windowed build must not
import tkinter before it is needed (a console-less process importing tkinter at
module level can fail before any error is reportable), so the import happens inside
``main`` and a failure is shown in a dialog rather than vanishing.
"""

from __future__ import annotations

import os
import sys


def _has_console() -> bool:
    """True when there is somewhere for text to go.

    A windowed build is linked without a console, so ``sys.stdout`` is ``None`` and a
    message must be shown in a dialog. A console build must never open a modal dialog
    on error: it would block forever on a machine or a CI job with nobody to dismiss
    it, which is exactly how a failing diagnostic turned into a silent 120-second hang.
    """
    try:
        return sys.stdout is not None and sys.stderr is not None
    except Exception:  # noqa: BLE001
        return False


def _fail(message: str, detail: str = "") -> int:
    """Report a startup failure where the user can actually see it."""
    text = message + (f"\n\n{detail}" if detail else "")
    if _has_console():
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
        return 1
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Lirik Studio tidak bisa dijalankan", text)
        root.destroy()
    except Exception:  # noqa: BLE001 - no display either; nothing left but to exit
        pass
    return 1


def _configure_frozen_environment() -> None:
    """Make the frozen layout behave like the source tree.

    The bundle is unpacked to a temporary directory, and the app writes its settings
    to the per-user config directory, so nothing here points at the bundle for
    writable state - that would be lost on exit.
    """
    bundle = getattr(sys, "_MEIPASS", "")
    if bundle and bundle not in sys.path:
        sys.path.insert(0, bundle)


def main() -> int:
    _configure_frozen_environment()
    try:
        from suno_agents.__main__ import main as cli_main
    except Exception as exc:  # noqa: BLE001 - import failure must be visible
        import traceback

        return _fail(
            "Gagal memuat aplikasi (import error).",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()[-1500:]}",
        )
    try:
        return int(cli_main() or 0)
    except SystemExit as exc:  # argparse and --help exit this way
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001 - any crash must be visible
        import traceback

        return _fail(
            "Aplikasi berhenti karena kesalahan.",
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()[-1500:]}",
        )


if __name__ == "__main__":
    # A windowed build re-enters through this guard; keep it side-effect free.
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())

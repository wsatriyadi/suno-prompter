"""Entry point: ``python -m suno_agents`` opens the studio window."""

from __future__ import annotations

import argparse
import sys


def _console_print(text: str) -> None:
    """Write to stdout, or to a message box when stdout does not exist.

    A windowed build (``--windowed``) has no console at all: ``print`` raises or
    silently vanishes, so ``--doctor`` and ``--headless-demo`` would appear to do
    nothing. Falling back to a dialog is the difference between a diagnostic tool
    and a mystery.
    """
    try:
        if sys.stdout is not None and sys.stdout.isatty() is not False:
            print(text)
            return
        print(text)
    except Exception:  # noqa: BLE001 - no stdout in a windowed build
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Lirik Studio", text[:20000])
            root.destroy()
        except Exception:  # noqa: BLE001 - nothing left to do but continue
            pass


def _bind_report(text_holder: dict) -> None:
    """Redirect diagnostics into ``text_holder['text']`` instead of stdout."""
    import io

    buffer = io.StringIO()
    sys.stdout = buffer
    text_holder["buffer"] = buffer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="suno_agents",
        description="Lirik Studio - nine specialist agents that produce one Suno prompt.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Start with the offline demo runner (no provider calls).",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print environment/provider diagnostics and exit.",
    )
    parser.add_argument(
        "--headless-demo",
        action="store_true",
        help="Run the full pipeline offline in the terminal and print the prompt.",
    )
    parser.add_argument(
        "--analyze",
        nargs="+",
        metavar="FILE",
        help="Measure audio reference files and print the profile, then exit. "
        "Useful for checking a file, and for verifying a packaged build can decode.",
    )
    args = parser.parse_args(argv)

    if args.doctor or args.headless_demo or args.analyze:
        # Capture first, then show: a windowed build has nowhere for print() to go.
        import io

        buffer = io.StringIO()
        original = sys.stdout
        sys.stdout = buffer
        try:
            if args.doctor:
                code = _doctor()
            elif args.analyze:
                code = _analyze(list(args.analyze))
            else:
                code = _headless_demo()
        finally:
            sys.stdout = original
        _console_print(buffer.getvalue())
        return code

    from .ui.app import run_app

    run_app(offline=args.offline)
    return 0


def _doctor() -> int:
    """Report what the app found: audio tooling, providers, config path."""
    from .audio.ffbin import bundled_search_dirs, require_ffmpeg
    from .config.models import user_config_dir
    from .config.store import SettingsStore
    from .domain.graph import describe_graph

    print("Lirik Studio diagnostics")
    print("=" * 60)
    try:
        ffmpeg, ffprobe = require_ffmpeg()
        print(f"ffmpeg : {ffmpeg}")
        print(f"ffprobe: {ffprobe}")
    except Exception as exc:  # noqa: BLE001 - diagnostics
        print(f"ffmpeg : MISSING ({exc})")
        searched = bundled_search_dirs()
        if searched:
            print("         dicari juga di: " + ", ".join(searched))

    try:
        import numpy  # noqa: F401

        print("numpy  : ok")
    except Exception as exc:  # noqa: BLE001
        print(f"numpy  : MISSING ({exc})")

    try:
        import tkinter

        print(f"tkinter: {tkinter.TkVersion}")
    except Exception as exc:  # noqa: BLE001
        print(f"tkinter: MISSING ({exc})")

    store = SettingsStore()
    settings = store.load()
    print(f"config : {store.path}")
    print(f"output : {settings.output_dir or user_config_dir() / 'runs'}")
    print(f"providers: {len(settings.providers)}")
    for provider in settings.providers:
        mark = "active" if provider.id == settings.active_provider_id else "      "
        creds = "key" if provider.has_credentials else "no-key"
        print(f"  [{mark}] {provider.name:28s} {provider.base_url} ({creds})")
    print()
    print(describe_graph())
    return 0


def _analyze(paths: list[str]) -> int:
    """Measure reference audio and print the full profile.

    Exists so the audio path can be verified without a display - and because "check
    this file" is a reasonable thing to want from a command line.
    """
    from .audio import AudioEngineError, build_profile

    try:
        profile = build_profile(paths)
    except AudioEngineError as exc:
        print(f"Tidak bisa menganalisa: {exc}")
        return 1
    if profile.error:
        print(f"Gagal: {profile.error}")
        return 1
    print(profile.summary)
    print()
    print(
        f"VERDICT: {profile.lossless_verdict} | quality {profile.quality_score}/100 | "
        f"{len(profile.per_file)} file dianalisa"
    )
    if profile.warnings:
        print("Peringatan:")
        for warning in profile.warnings:
            print(f"- {warning}")
    return 0


def _headless_demo() -> int:
    """Exercise the whole pipeline with the mock runner and print the result."""
    from .pipeline.orchestrator import Orchestrator
    from .testing.demo import build_demo_runner

    orchestrator = Orchestrator(build_demo_runner(), settings=None)
    result = orchestrator.run(None)
    print(result.report)
    print()
    print(f"PASSED: {result.passed} | revisions: {result.state.revisions}")
    return 0 if result.prompt is not None else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

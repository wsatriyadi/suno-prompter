"""Interface tests.

Two layers, on purpose:

* **Pure theme maths** runs anywhere. Palette ramps, wrapping, truncation and
  status mapping are logic, and logic deserves assertions that never need a
  display.
* **Windowed scenarios** run as *subprocesses*. Tk does not tolerate many roots
  inside one process (it starts failing to load Tcl libraries and can hang), and
  a hang inside pytest is worse than no test. So each scenario is a small script
  under ``tests/uitasks/`` that exits non-zero with a precise message, and these
  tests assert on its exit status and stdout. That also makes the failure
  diagnosable outside of pytest.

The scenarios are not smoke tests: they build the real application, run the real
nine-stage pipeline through the real event pump, and then inspect the actual
widget state and the actual files on disk. One of them also saves a screenshot,
so the interface can be looked at rather than assumed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest

from suno_agents.ui.theme import (
    Metrics,
    Palette,
    StatusCode,
    level_colour,
    meter_blocks,
    status_colour,
    truncate,
    wrap_words,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASKS = os.path.join(ROOT, "tests", "uitasks")


def _has_display() -> bool:
    """Probe Tk in a subprocess: a try/except in-process can poison the runtime."""
    probe = (
        "import tkinter,sys\n"
        "r=tkinter.Tk();r.withdraw();r.update();r.destroy();print('yes')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ROOT,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and "yes" in result.stdout


HAS_DISPLAY = _has_display()
windowed = pytest.mark.skipif(not HAS_DISPLAY, reason="no usable display / Tk runtime")


def run_task(script: str, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.join(TASKS, script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": ROOT},
    )


# --------------------------------------------------------------------------- #
# pure theme maths
# --------------------------------------------------------------------------- #


def test_truncate_respects_the_limit_and_marks_elision():
    assert truncate("short", 20) == "short"
    out = truncate("a very long sentence that will not fit here", 12)
    assert len(out) <= 12
    assert out.endswith("\u2026")
    assert truncate("", 5) == ""


def test_truncate_collapses_whitespace():
    assert truncate("line one\n\n   line two", 40) == "line one line two"


def test_wrap_words_never_exceeds_width():
    text = "the quick brown fox jumps over the lazy dog again and again"
    for width in (10, 14, 18, 24):
        for line in wrap_words(text, width):
            assert len(line) <= width or " " not in line


def test_wrap_words_handles_empty_and_whitespace():
    assert wrap_words("", 10) == [""]
    assert wrap_words("   ", 10) == [""]


def test_meter_blocks_is_proportional_and_clamped():
    assert meter_blocks(0.0, 10) == [False] * 10
    assert meter_blocks(1.0, 10) == [True] * 10
    assert sum(meter_blocks(0.5, 10)) == 5
    assert meter_blocks(-3.0, 4) == [False] * 4
    assert meter_blocks(9.9, 4) == [True] * 4


def test_level_colour_walks_the_spectrum_and_clamps():
    palette = Palette()
    low = level_colour(palette, 0.0)
    mid = level_colour(palette, 0.5)
    high = level_colour(palette, 1.0)
    assert low == palette.spectrum[0]
    assert high == palette.spectrum[-1]
    assert len({low, mid, high}) == 3
    assert level_colour(palette, -1.0) == low
    assert level_colour(palette, 4.0) == high


def test_stage_colour_tokens_resolve_to_real_colours():
    palette = Palette()
    for token in ("c1", "c2", "c3", "c4", "c5", "c6"):
        assert palette.stage_colour(token).startswith("#")
    assert palette.stage_colour("nonsense") == palette.accent


def test_status_colour_maps_every_lifecycle_state():
    palette = Palette()
    assert status_colour(palette, StatusCode.PENDING) == palette.fg_faint
    assert status_colour(palette, "running") == palette.accent
    assert status_colour(palette, "done") == palette.mint
    assert status_colour(palette, "failed") == palette.red
    assert status_colour(palette, "not-a-status") == palette.fg_faint


def test_every_stage_has_a_glyph_and_a_palette_colour():
    from suno_agents.domain.graph import STAGE_INFO, STAGE_ORDER
    from suno_agents.ui.theme import STAGE_GLYPHS

    palette = Palette()
    for stage in STAGE_ORDER:
        assert stage in STAGE_GLYPHS, stage
        colour = palette.stage_colour(STAGE_INFO[stage].accent)
        assert colour.startswith("#")
    # Glyphs must be distinguishable, or the rail loses its signal.
    assert len(set(STAGE_GLYPHS.values())) == len(STAGE_ORDER)


def test_metrics_are_on_a_four_pixel_grid():
    metrics = Metrics()
    for value in (
        metrics.unit,
        metrics.pad_sm,
        metrics.pad_md,
        metrics.pad_lg,
        metrics.pad_xl,
        metrics.header_h,
        metrics.status_h,
    ):
        assert value % 4 == 0


# --------------------------------------------------------------------------- #
# windowed scenarios (subprocesses)
# --------------------------------------------------------------------------- #


@windowed
def test_settings_dialog_custom_provider_round_trip():
    result = run_task("settings_roundtrip.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


@windowed
def test_main_window_runs_the_pipeline_and_writes_artifacts():
    result = run_task("run_pipeline_ui.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


@windowed
def test_consultant_failure_reports_instead_of_crashing():
    """The reported NameError on ``exc``, on the path users hit when a call fails."""
    result = run_task("intake_failure.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


@windowed
def test_agent_log_tab_shows_requests_replies_and_decisions():
    """The Log Agen tab must be useful, not just present."""
    result = run_task("agent_log_tab.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


@windowed
def test_settings_routing_tab_round_trip():
    """The reported bug: saved routing vanished when the tab was reopened."""
    result = run_task("routing_roundtrip.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


@windowed
def test_main_window_renders_and_screenshots():
    out = os.path.join(ROOT, "artifacts", "lirik-studio.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    result = run_task("render_window.py", out, "full")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "SCREENSHOT" in result.stdout
    assert os.path.exists(out) and os.path.getsize(out) > 20_000


@pytest.mark.skipif(not HAS_DISPLAY, reason="no usable display / Tk runtime")
def test_rendered_window_is_not_blank():
    """Guard against the classic silent failure: a window that paints nothing."""
    path = os.path.join(ROOT, "artifacts", "lirik-studio.png")
    if not os.path.exists(path):
        pytest.skip("screenshot not produced in this session")
    from PIL import Image

    with Image.open(path) as image:
        colors = image.convert("RGB").getcolors(maxcolors=1_000_000) or []
    # A blank window would have a handful of colours; a real UI has many.
    assert len(colors) > 500, f"only {len(colors)} distinct colours - window looks blank"

# --------------------------------------------------------------------------- #
# worker-thread hand-off safety
#
# The reported traceback was a NameError on `exc` raised from a Tk callback: the
# worker built `lambda: self._after_intake_error(exc)` inside `except ... as exc`,
# and Python unbinds that name when the block exits. A static check is the right
# guard here - the equivalent runtime test would need a live provider failure.
# --------------------------------------------------------------------------- #


def test_no_ui_code_captures_an_exception_in_a_deferred_callback():
    """A callback built from ``except ... as name`` must not close over the name."""
    import re
    from pathlib import Path

    ui_dir = Path(ROOT) / "suno_agents" / "ui"
    offenders: list[str] = []
    pattern = re.compile(r"(lambda|def)[^\n]*\b(after|call_later)\b[^\n]*\bexc\b")
    for path in ui_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.name))
        # Also catch the direct form: after(0, lambda: ...(exc))
        for match in re.finditer(r"after\([^)]*lambda[^)]*\bexc\b", text):
            offenders.append(f"{path.name}: {match.group(0)[:60]}")
    assert not offenders, (
        "deferring a callback that closes over an `except as` name raises NameError "
        f"once the block exits: {offenders}"
    )


def test_ui_does_not_call_tk_scheduling_from_worker_threads():
    """``after()``/``after_idle()`` are not safe off the UI thread.

    Every worker must hand its result over through a queue that ``_drain_once``
    consumes, which is the pattern the pipeline already used and the intake path
    did not.
    """
    from pathlib import Path

    text = (Path(ROOT) / "suno_agents" / "ui" / "app.py").read_text(encoding="utf-8")
    worker_bodies = re.findall(r"def worker\(\) -> None:(.*?)\n        threading\.Thread", text, re.DOTALL)
    assert worker_bodies, "expected worker functions in app.py"
    for body in worker_bodies:
        assert ".after(" not in body, (
            "a worker thread must not schedule Tk callbacks; put the result in a "
            f"queue instead: {body.strip()[:200]}"
        )


def test_the_main_window_has_queues_for_every_worker_handoff():
    from pathlib import Path

    text = (Path(ROOT) / "suno_agents" / "ui" / "app.py").read_text(encoding="utf-8")
    assert "_deltas" in text and "_intake_results" in text
    # Both queues must be drained in the single UI-thread pump.
    drain = text[text.index("def _drain_once"):]
    drain = drain[: drain.index("def ", 10)]
    assert "_deltas.get_nowait()" in drain
    assert "_intake_results.get_nowait()" in drain

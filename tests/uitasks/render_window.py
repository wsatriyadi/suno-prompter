"""Build the real app, populate it with a finished run, and save a screenshot.

Run as a subprocess (see ``tests/test_ui.py``). Tk does not tolerate many
``Tk()`` roots inside one process, and this script also captures the rendered
window so a human can look at the interface instead of trusting assertions.
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

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "lirik-ui.png")
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"


def main() -> int:
    home = tempfile.mkdtemp(prefix="lirik-shot-")
    os.environ["LIRIK_HOME"] = home
    from suno_agents.pipeline.orchestrator import Orchestrator
    from suno_agents.testing.demo import build_demo_runner
    from suno_agents.ui.app import StudioApp

    app = StudioApp(offline=True)
    app.geometry("1560x940+20+20")
    try:
        app.update()
        app.deiconify()
        app.update()

        # Drive the intake turn so the transcript is real, not empty.
        app.chat_panel.view.add_message(
            "user",
            "Aku mau lagu galau tapi tetap enak didengar di perjalanan pulang. "
            "Penyanyi perempuan, ada bagian melayu sedikit.",
            at="21:04",
        )
        payload = app.session.send_user_message(
            "Aku mau lagu galau tapi tetap enak didengar di perjalanan pulang. "
            "Penyanyi perempuan, ada bagian melayu sedikit."
        )
        app.chat_panel.view.add_message("assistant", payload["reply"], at="21:04")
        app._refresh_brief_stage()

        # Reference panel with a degraded reference: the honest-caveat path.
        from suno_agents.pipeline.session import ReferenceBundle

        ref_panel = next(
            widget for widget in app._walk(app) if type(widget).__name__ == "ReferencePanel"
        )
        ref_panel.set_bundle(
            ReferenceBundle(
                paths=["C:/music/referensi-lama.mp3"],
                summary=(
                    "file: referensi-lama.mp3\ncodec: mp3 (lossy) | 128 kbps | "
                    "44100 Hz | 2 ch | 213.4 s\n"
                    "tempo: 92.3 BPM (confidence 0.81) | key: C major\n"
                    "loudness: -9.4 LUFS | true peak: -0.2 dBFS\n"
                    "dynamic range: 5.1 dB | crest factor: 7.4 dB\n"
                    "stereo width: 0.31 | correlation: 0.88\n"
                    "spectral centroid: 2140 Hz | rolloff95: 9.8 kHz\n"
                    "lowpass cutoff: 16.0 kHz (conf 0.92) - 128 kbps class\n"
                    "bands: sub -14.2 dB, low -2.1 dB, low_mid 0.0 dB, mid -3.4 dB, "
                    "high_mid -6.8 dB, high -18.5 dB, air -34.1 dB\n"
                    "energy curve rises to bucket 21 then drops\n\n"
                    "Lossless verdict: lossy-transcoded | quality 38/100"
                ),
                lossless="lossy-transcoded",
                quality=38,
            )
        )

        if MODE == "full":
            # Run the whole pipeline so the rail, the inspector and the prompt
            # panel all hold finished state in the picture.
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

        app._show_right("prompt")
        app.update()
        app.update_idletasks()
        time.sleep(0.4)
        app.update()

        # Keep the window on screen long enough for the grab to see it.
        app.lift()
        app.attributes("-topmost", True)
        app.update()
        time.sleep(0.5)
        app.update()

        try:
            from PIL import ImageGrab

            image = ImageGrab.grab()
            image.save(OUT)
            print(f"SCREENSHOT {OUT} {image.size}")
        except Exception as exc:  # noqa: BLE001 - reporting is the point
            print(f"ERROR: screenshot failed: {type(exc).__name__}: {exc}")
            return 3

        print("OK")
        return 0
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

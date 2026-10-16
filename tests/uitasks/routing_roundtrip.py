"""Exercise the settings dialog's routing tab as a user would.

Run as a subprocess (see ``tests/test_ui.py``). The bug being pinned: configuring
nine stages, closing the dialog, and reopening it showed nothing - because opening
the tab collected an unpopulated form over the saved values. This script asserts
the saved routing survives load, re-load, save, and a full disk round trip.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)


def fail(message: str) -> int:
    print(f"ERROR: {message}")
    return 1


def main() -> int:
    home = tempfile.mkdtemp(prefix="lirik-routing-")
    os.environ["LIRIK_HOME"] = home
    try:
        import tkinter as tk

        from suno_agents.config.models import AppSettings, ProviderConfig, StageOverride
        from suno_agents.config.store import SettingsStore
        from suno_agents.domain.graph import STAGE_ORDER
        from suno_agents.ui.settings import SettingsDialog
        from suno_agents.ui.theme import Palette, build_fonts, resolve_typography

        store = SettingsStore()
        root = tk.Tk()
        root.withdraw()
        palette = Palette()
        fonts = build_fonts(resolve_typography(root))

        # Seed three providers and a routing plan the user cares about.
        settings = AppSettings()
        settings.providers = [
            ProviderConfig(id="base", name="Base", base_url="http://127.0.0.1:9/v1", model="m-base"),
            ProviderConfig(id="strong", name="Strong", base_url="http://127.0.0.1:9/v1", model="m-strong"),
            ProviderConfig(id="cheap", name="Cheap", base_url="http://127.0.0.1:9/v1", model="m-cheap"),
        ]
        settings.active_provider_id = "base"
        settings.ensure_defaults()
        settings.stage_overrides["composer"] = StageOverride(
            provider_id="strong", model="", temperature=0.85, enabled=True
        )
        settings.stage_overrides["lyricist"] = StageOverride(
            provider_id="strong", model="m-strong-turbo", temperature=0.95, enabled=True
        )
        settings.stage_overrides["lyric_reviewer"] = StageOverride(
            provider_id="cheap", model="m-cheap-critic", temperature=0.15, enabled=True
        )
        store.save(settings)
        expected = {k: v.to_dict() for k, v in settings.stage_overrides.items()}

        dialog = SettingsDialog(root, palette, fonts, store.load(), store, on_saved=None)
        dialog.withdraw()
        root.update_idletasks()

        # 1. Opening the routing tab must show the saved plan, not blanks.
        dialog._show_tab("routing")
        root.update_idletasks()
        rows = dialog.routing.rows
        for stage, override in (
            ("composer", "provider_id"),
            ("lyricist", None),
            ("lyric_reviewer", None),
        ):
            shown_provider = rows[stage]["provider_var"].get()
            shown_model = rows[stage]["model"].get()
            if stage == "composer":
                if shown_provider != "strong":
                    return fail(f"composer provider shows {shown_provider!r}, expected 'strong'")
                if shown_model != "":
                    return fail(f"composer model shows {shown_model!r}, expected blank (inherit)")
            if stage == "lyricist" and shown_model != "m-strong-turbo":
                return fail(f"lyricist model shows {shown_model!r}")
            if stage == "lyric_reviewer" and shown_provider != "cheap":
                return fail(f"lyric_reviewer provider shows {shown_provider!r}")
        # The resolved label must name the effective target, not the raw override.
        composer_label = rows["composer"]["resolved"].cget("text")
        if "m-strong" not in composer_label or "override" not in composer_label:
            return fail(f"composer resolved label reads {composer_label!r}")
        intake_label = rows["intake"]["resolved"].cget("text")
        if "inherit" not in intake_label:
            return fail(f"intake resolved label reads {intake_label!r}")

        # 2. Re-producing the old failure: a collect during the load path.
        dialog.routing.load(dialog._collect_all())
        after_reload = {k: v.to_dict() for k, v in dialog.settings.stage_overrides.items()}
        for stage, want in expected.items():
            got = after_reload.get(stage, {})
            for field in ("provider_id", "model", "temperature", "enabled"):
                if got.get(field) != want.get(field):
                    return fail(
                        f"{stage}.{field} became {got.get(field)!r} (was {want.get(field)!r})"
                    )

        # 3. Switch tabs repeatedly; nothing may drift.
        for _ in range(3):
            dialog._show_tab("providers")
            root.update_idletasks()
            dialog._show_tab("routing")
            root.update_idletasks()
        again = {k: v.to_dict() for k, v in dialog.settings.stage_overrides.items()}
        for stage, want in expected.items():
            if again.get(stage, {}).get("model") != want.get("model"):
                return fail(f"{stage}.model drifted to {again[stage].get('model')!r} after tab switches")

        # 4. Edit routing through the widgets, save, and confirm it lands on disk.
        rows["arranger"]["provider_var"].set("cheap")
        rows["arranger"]["model"].delete(0, "end")
        rows["arranger"]["model"].insert(0, "m-cheap-arranger")
        rows["arranger"]["temp"].delete(0, "end")
        rows["arranger"]["temp"].insert(0, "0.35")
        dialog._on_save()
        root.update_idletasks()

        reloaded = store.load()
        arranger = reloaded.stage_overrides["arranger"]
        if arranger.provider_id != "cheap" or arranger.model != "m-cheap-arranger":
            return fail(f"arranger override not saved: {arranger.to_dict()}")
        if abs(arranger.temperature - 0.35) > 1e-9:
            return fail(f"arranger temperature saved as {arranger.temperature}")
        # Everything we did NOT edit must still be intact. `arranger` was edited
        # on purpose above, so it is excluded from that comparison.
        for stage, want in expected.items():
            if stage == "arranger":
                continue
            got = reloaded.stage_overrides[stage].to_dict()
            if got.get("provider_id") != want.get("provider_id") or got.get("model") != want.get("model"):
                return fail(f"{stage} was damaged by saving the editor: {got}")

        raw = json.loads(store.path.read_text(encoding="utf-8"))
        if raw["stage_overrides"]["composer"]["provider_id"] != "strong":
            return fail("composer override missing from the file on disk")

        # 5. Nine distinct models, the user's stated goal. _on_save() closes the
        #    dialog, so this needs a fresh one - which also re-proves that a newly
        #    opened dialog *loads* what the previous save wrote.
        dialog2 = SettingsDialog(root, palette, fonts, store.load(), store, on_saved=None)
        dialog2.withdraw()
        root.update_idletasks()
        dialog2._show_tab("routing")
        root.update_idletasks()
        # The plan written a moment ago must be visible immediately.
        if dialog2.routing.rows["arranger"]["model"].get() != "m-cheap-arranger":
            return fail(
                "a reopened dialog did not show the arranger override that was just saved"
            )
        for index, stage in enumerate(STAGE_ORDER):
            dialog2.routing.rows[stage]["provider_var"].set("base")
            dialog2.routing.rows[stage]["model"].delete(0, "end")
            dialog2.routing.rows[stage]["model"].insert(0, f"per-stage-{index}")
        dialog2._on_save()
        final = store.load()
        models = {s: final.routing_for(s).model for s in STAGE_ORDER}
        if len(set(models.values())) != len(STAGE_ORDER):
            return fail(f"nine distinct models did not persist: {models}")

        # 6. And a third dialog sees all nine.
        dialog3 = SettingsDialog(root, palette, fonts, store.load(), store, on_saved=None)
        dialog3.withdraw()
        dialog3._show_tab("routing")
        root.update_idletasks()
        for index, stage in enumerate(STAGE_ORDER):
            shown = dialog3.routing.rows[stage]["model"].get()
            if shown != f"per-stage-{index}":
                return fail(f"reopened dialog shows {stage} model as {shown!r}")
        # The model dropdown must actually offer model ids (not an empty list).
        values = dialog3.routing.rows["composer"]["model"].cget("values")
        if not values:
            return fail("model dropdown has no choices, so picking per stage means typing")

        for widget in (dialog3,):
            try:
                widget.destroy()
            except tk.TclError:
                pass
        root.destroy()
        print("OK")
        return 0
    finally:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

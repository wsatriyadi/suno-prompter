"""Build the settings dialog, exercise the custom-provider flow, verify persistence.

Run as a subprocess (see ``tests/test_ui.py``). Prints ``OK`` on success and a
precise reason on failure, so a broken dialog is diagnosable from the exit code
and stdout alone.
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
    home = tempfile.mkdtemp(prefix="lirik-settings-")
    os.environ["LIRIK_HOME"] = home
    try:
        import tkinter as tk

        from suno_agents.config.models import AppSettings
        from suno_agents.config.store import SettingsStore
        from suno_agents.ui.settings import SettingsDialog
        from suno_agents.ui.theme import Palette, build_fonts, resolve_typography

        store = SettingsStore()
        root = tk.Tk()
        root.withdraw()
        palette = Palette()
        fonts = build_fonts(resolve_typography(root))
        settings = AppSettings()
        settings.ensure_defaults()

        dialog = SettingsDialog(root, palette, fonts, settings, store, on_saved=None)
        dialog.withdraw()
        root.update_idletasks()

        # --- adding a preset -------------------------------------------------
        before = len(dialog.settings.providers)
        dialog._add_preset("ollama")
        if len(dialog.settings.providers) != before + 1:
            return fail("preset was not added")
        ollama = dialog.settings.provider_by_id("ollama")
        if ollama is None or "11434" not in ollama.base_url:
            return fail("ollama preset has the wrong base_url")

        # --- adding a custom OpenAI-compatible provider ----------------------
        dialog._add_provider()
        editor = dialog.provider_editor
        for field, value in (
            ("name", "vLLM lokal"),
            ("base_url", "http://127.0.0.1:8000/v1"),
            ("model", "qwen3-32b-instruct"),
            ("api_key", "env:MY_VLLM_KEY"),
            ("temperature", "0.55"),
            ("max_tokens", "8192"),
            ("timeout", "240"),
            ("notes", "RTX 4090, no key needed"),
        ):
            getattr(editor, field).set(value)
        editor.json_mode.set("object")
        editor.streaming.set(True)
        dialog._save_provider()

        provider = dialog.settings.provider_by_id(dialog.selected_id)
        if provider is None:
            return fail("custom provider not found after save")
        checks = {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "qwen3-32b-instruct",
            "api_key": "env:MY_VLLM_KEY",
            "notes": "RTX 4090, no key needed",
        }
        for field, expected in checks.items():
            actual = getattr(provider, field)
            if actual != expected:
                return fail(f"{field}: expected {expected!r}, got {actual!r}")
        if abs(provider.temperature - 0.55) > 1e-9:
            return fail(f"temperature not parsed: {provider.temperature}")
        if provider.max_tokens != 8192:
            return fail(f"max_tokens not parsed: {provider.max_tokens}")
        if provider.chat_url != "http://127.0.0.1:8000/v1/chat/completions":
            return fail(f"chat_url wrong: {provider.chat_url}")
        if provider.models_url != "http://127.0.0.1:8000/v1/models":
            return fail(f"models_url wrong: {provider.models_url}")
        if not provider.has_credentials:
            return fail("provider should count as credentialed (env: indirection)")
        if provider.validate():
            return fail(f"validate() rejected a good provider: {provider.validate()}")

        # --- routing tab and pipeline tab ------------------------------------
        dialog._show_tab("routing")
        root.update_idletasks()
        dialog.routing.rows["composer"]["temp"].delete(0, "end")
        dialog.routing.rows["composer"]["temp"].insert(0, "0.85")
        dialog.routing.rows["composer"]["model"].delete(0, "end")
        dialog.routing.rows["composer"]["model"].insert(0, "qwen3-32b-instruct")
        dialog.routing.rows["lyric_reviewer"]["temp"].delete(0, "end")
        dialog.routing.rows["lyric_reviewer"]["temp"].insert(0, "0.15")

        dialog._show_tab("pipeline")
        dialog.max_revisions.set("30")
        dialog.stage_attempts.set("2")
        dialog.offline_demo.set(False)
        dialog.output_dir.set(os.path.join(home, "hasil"))

        dialog._collect_all()
        collected = dialog.settings
        collected.active_provider_id = provider.id
        collected.ensure_defaults()

        if abs(collected.stage_overrides["composer"].temperature - 0.85) > 1e-9:
            return fail("routing temperature for composer was not collected")
        if collected.stage_overrides["composer"].model != "qwen3-32b-instruct":
            return fail("routing model for composer was not collected")
        if abs(collected.stage_overrides["lyric_reviewer"].temperature - 0.15) > 1e-9:
            return fail("routing temperature for lyric_reviewer was not collected")
        if collected.max_revisions != 30 or collected.stage_attempts != 2:
            return fail("pipeline settings were not collected")
        if collected.effective_model("composer") != "qwen3-32b-instruct":
            return fail("effective_model did not honour the stage override")
        if abs(collected.effective_temperature("composer") - 0.85) > 1e-9:
            return fail("effective_temperature did not honour the stage override")

        # --- persistence round trip ------------------------------------------
        store.save(collected)
        dialog.destroy()
        root.destroy()

        reloaded = store.load()
        saved = reloaded.provider_by_id(provider.id)
        if saved is None:
            return fail("provider did not survive the disk round trip")
        if saved.api_key != "env:MY_VLLM_KEY":
            return fail(f"api_key changed on disk: {saved.api_key!r}")
        if reloaded.active_provider_id != provider.id:
            return fail("active_provider_id did not survive")
        if abs(reloaded.stage_overrides["composer"].temperature - 0.85) > 1e-9:
            return fail("stage override did not survive")
        if reloaded.max_revisions != 30:
            return fail("max_revisions did not survive")
        if reloaded.output_dir != os.path.join(home, "hasil"):
            return fail("output_dir did not survive")

        # The settings file must be valid JSON and must not be a temp file.
        raw = json.loads(store.path.read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            return fail(f"unexpected settings version: {raw.get('version')!r}")
        leftovers = [p for p in os.listdir(os.path.dirname(store.path)) if p.endswith(".tmp")]
        if leftovers:
            return fail(f"atomic write left temp files behind: {leftovers}")

        print("OK")
        return 0
    finally:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

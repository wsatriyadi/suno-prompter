"""Settings dialog: LLM providers, per-stage routing, pipeline behaviour.

Design decisions worth stating:

* The provider list is on the left, the editor on the right - the same master/
  detail shape as any router UI, because that is what this is.
* "Test" does a real one-token round trip and reports latency, because a silent
  misconfiguration costs the user a 20-minute pipeline run to discover.
* Per-stage routing exists because the smart play is a strong model for the
  composer and lyricist and a cheap one for the reviewers. Defaults are sane so a
  user can ignore the tab entirely.
* Secrets: pasting a key stores it in the config file; typing ``env:NAME`` reads
  it from the environment instead. The dialog explains that in one line.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Callable

from ..config.models import (
    PRESET_PROVIDERS,
    AppSettings,
    ProviderConfig,
    StageOverride,
    default_provider,
    preset_provider,
    slugify,
)
from ..config.store import SettingsStore, export_providers, import_providers
from ..domain.graph import STAGE_INFO, STAGE_ORDER, STAGE_TEMPERATURE_HINTS
from ..llm.client import LLMClient
from ..llm.wire import LLMError
from .theme import Fonts, Metrics, Palette, truncate
from .widgets import (
    AutoWrapLabel,
    Badge,
    Card,
    Divider,
    PillButton,
    ScrollFrame,
    SectionLabel,
)

M = Metrics()

#: Shown in a stage's provider chooser when the stage should inherit the active
#: provider instead of being pinned to a specific one.
INHERIT_LABEL = "(provider aktif)"


class TextField(tk.Frame):
    """Labelled single-line entry with an optional row of preset chips."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        label: str,
        *,
        hint: str = "",
        show: str | None = None,
        width: int = 34,
    ) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        tk.Label(
            self, text=label.upper(), bg=palette.surface, fg=palette.fg_faint,
            font=fonts.label, anchor="w",
        ).pack(anchor="w")
        self.var = tk.StringVar()
        self.entry = tk.Entry(
            self,
            textvariable=self.var,
            width=width,
            bg=palette.surface_alt,
            fg=palette.fg,
            insertbackground=palette.accent,
            selectbackground=palette.accent_dim,
            selectforeground=palette.fg,
            font=fonts.mono if show is None else fonts.mono,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.line_soft,
            highlightcolor=palette.accent,
            show=show or "",
        )
        self.entry.pack(fill="x", ipady=4, pady=(2, 0))
        if hint:
            AutoWrapLabel(
                self, hint, palette, fonts, fg=palette.fg_faint,
            ).pack(fill="x", pady=(2, 0))

    def get(self) -> str:
        return self.var.get()

    def set(self, value: str) -> None:
        self.var.set(value or "")


class ProviderList(tk.Frame):
    """Master list of configured providers."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        on_select: Callable[[str], None],
        on_add: Callable[[], None],
        on_duplicate: Callable[[], None],
        on_remove: Callable[[], None],
        on_preset: Callable[[str], None],
        on_import: Callable[[], None],
        on_export: Callable[[], None],
    ) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts
        self._on_select = on_select
        self._badges: dict[str, tk.Label] = {}

        head = tk.Frame(self, bg=palette.surface)
        head.pack(fill="x")
        SectionLabel(head, "Provider", palette, fonts).pack(side="left")
        PillButton(head, "+", on_add, palette, fonts, kind="ghost", width=26, height=22,
                   tooltip="Tambah provider kosong").pack(side="right")
        PillButton(head, "\u29c9", on_duplicate, palette, fonts, kind="ghost", width=26, height=22,
                   tooltip="Duplikat provider terpilih").pack(side="right", padx=(0, 4))

        self.list_frame = tk.Frame(self, bg=palette.surface)
        self.list_frame.pack(fill="x", pady=(6, 6))

        Divider(self, palette).pack(fill="x", pady=6)

        SectionLabel(self, "Tambah dari preset", palette, fonts).pack(anchor="w")
        presets = tk.Frame(self, bg=palette.surface)
        presets.pack(fill="x", pady=(4, 6))
        for preset in PRESET_PROVIDERS:
            PillButton(
                presets,
                preset["name"],
                (lambda pid=preset["id"]: on_preset(pid)),
                palette,
                fonts,
                kind="ghost",
                height=24,
                width=124,
                tooltip=preset.get("notes", ""),
            ).pack(side="top", anchor="w", pady=1)

        Divider(self, palette).pack(fill="x", pady=6)
        row = tk.Frame(self, bg=palette.surface)
        row.pack(fill="x")
        PillButton(row, "Impor", on_import, palette, fonts, kind="ghost", height=24,
                   glyph="\u2913", tooltip="Impor pak provider dari JSON").pack(side="left")
        PillButton(row, "Ekspor", on_export, palette, fonts, kind="ghost", height=24,
                   glyph="\u2911", tooltip="Simpan pak provider ke JSON").pack(side="left", padx=6)
        PillButton(row, "Hapus", on_remove, palette, fonts, kind="danger", height=24,
                   glyph="\u2715", tooltip="Hapus provider terpilih").pack(side="right")

    def render(self, providers: list[ProviderConfig], selected_id: str) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        self._badges.clear()
        for provider in providers:
            selected = provider.id == selected_id
            row = tk.Frame(
                self.list_frame,
                bg=self._palette.surface_high if selected else self._palette.surface_alt,
                cursor="hand2",
            )
            row.pack(fill="x", pady=1)
            marker = tk.Frame(
                row,
                bg=self._palette.accent if selected else self._palette.line,
                width=2,
                height=34,
            )
            marker.pack(side="left", fill="y")
            text = tk.Frame(row, bg=row["bg"])
            text.pack(side="left", fill="x", expand=True, padx=8, pady=4)
            name = tk.Label(
                text,
                text=truncate(provider.name or provider.id, 26),
                bg=row["bg"],
                fg=self._palette.fg if selected else self._palette.fg_muted,
                font=self._fonts.body_bold if selected else self._fonts.body,
                anchor="w",
            )
            name.pack(anchor="w")
            detail = f"{truncate(provider.model, 24)} \u00b7 {truncate(provider.base_url, 28)}"
            tk.Label(
                text, text=detail, bg=row["bg"], fg=self._palette.fg_faint,
                font=self._fonts.mono_small, anchor="w",
            ).pack(anchor="w")
            status = tk.Label(
                row,
                text="\u2713" if provider.has_credentials else "\u26a0",
                bg=row["bg"],
                fg=self._palette.mint if provider.has_credentials else self._palette.amber,
                font=self._fonts.body,
            )
            status.pack(side="right", padx=8)
            for widget in (row, text, name, status):
                widget.bind("<Button-1>", lambda _e, pid=provider.id: self._on_select(pid))


class ProviderEditor(tk.Frame):
    """Detail editor for one provider, plus the connection test."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        *,
        on_test: Callable[[], None],
        on_save: Callable[[], None],
        on_fetch_models: Callable[[], None],
    ) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts

        grid = tk.Frame(self, bg=palette.surface)
        grid.pack(fill="x")
        left = tk.Frame(grid, bg=palette.surface)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(grid, bg=palette.surface)
        right.pack(side="left", fill="both", expand=True)

        self.name = TextField(left, palette, fonts, "Nama provider", width=26)
        self.name.pack(fill="x", pady=(0, 6))
        self.base_url = TextField(
            left, palette, fonts, "Base URL", width=26,
            hint="Berakhir di /v1 untuk sebagian besar server. /chat/completions juga boleh.",
        )
        self.base_url.pack(fill="x", pady=(0, 6))
        self.api_key = TextField(
            left, palette, fonts, "API key", width=26, show="\u2022",
            hint="Tempel kunci, atau tulis env:NAMA_VAR untuk membaca dari environment.",
        )
        self.api_key.pack(fill="x", pady=(0, 6))

        self.model = TextField(right, palette, fonts, "Model", width=26)
        self.model.pack(fill="x", pady=(0, 6))
        self.model_row = tk.Frame(right, bg=palette.surface)
        self.model_row.pack(fill="x", pady=(0, 6))
        PillButton(
            self.model_row, "Ambil daftar model", on_fetch_models, palette, fonts,
            kind="ghost", height=24, glyph="\u21bb", tooltip="GET /models pada endpoint ini",
        ).pack(side="left")
        self.models_hint = tk.Label(
            self.model_row, text="", bg=palette.surface, fg=palette.fg_faint,
            font=fonts.small, anchor="w",
        )
        self.models_hint.pack(side="left", padx=8)

        nums = tk.Frame(right, bg=palette.surface)
        nums.pack(fill="x")
        self.temperature = TextField(nums, palette, fonts, "Temperature", width=8)
        self.temperature.pack(side="left", padx=(0, 6))
        self.max_tokens = TextField(
            nums, palette, fonts, "Max token output", width=8,
            hint="Kosongkan atau 0 = tanpa batas (model memakai maksimumnya sendiri).",
        )
        self.max_tokens.pack(side="left", padx=(0, 6))
        self.timeout = TextField(nums, palette, fonts, "Timeout (s)", width=8)
        self.timeout.pack(side="left")

        modes = tk.Frame(self, bg=palette.surface)
        modes.pack(fill="x", pady=(10, 0))
        tk.Label(
            modes, text="MODE JSON", bg=palette.surface, fg=palette.fg_faint,
            font=fonts.label, anchor="w",
        ).pack(anchor="w")
        self.json_mode = tk.StringVar(value="object")
        row = tk.Frame(modes, bg=palette.surface)
        row.pack(fill="x")
        for value, label in (
            ("object", "json_object"),
            ("schema", "json_schema"),
            ("off", "tanpa JSON mode"),
        ):
            tk.Radiobutton(
                row, text=label, value=value, variable=self.json_mode,
                bg=palette.surface, fg=palette.fg, selectcolor=palette.surface_alt,
                activebackground=palette.surface, activeforeground=palette.accent,
                font=fonts.small, bd=0, highlightthickness=0,
            ).pack(side="left", padx=(0, 12))

        self.streaming = tk.BooleanVar(value=True)
        tk.Checkbutton(
            modes, text="Server mendukung stream=true (untuk teks berjalan)",
            variable=self.streaming, bg=palette.surface, fg=palette.fg,
            selectcolor=palette.surface_alt, activebackground=palette.surface,
            activeforeground=palette.accent, font=fonts.small, bd=0,
            highlightthickness=0, anchor="w",
        ).pack(anchor="w", pady=(4, 0))

        self.notes = TextField(
            self, palette, fonts, "Catatan", width=70,
            hint="Muncul sebagai tooltip di daftar provider.",
        )
        self.notes.pack(fill="x", pady=(10, 0))

        actions = tk.Frame(self, bg=palette.surface)
        actions.pack(fill="x", pady=(12, 0))
        PillButton(
            actions, "Simpan provider", on_save, palette, fonts, kind="primary",
            glyph="\u2713", height=30,
        ).pack(side="left")
        PillButton(
            actions, "Tes koneksi", on_test, palette, fonts, kind="ghost",
            glyph="\u26a1", height=30, tooltip="Kirim satu permintaan kecil dan ukur latensi",
        ).pack(side="left", padx=8)
        self.test_badge = Badge(actions, "belum dites", palette, fonts)
        self.test_badge.pack(side="left", padx=6)

        self.status = AutoWrapLabel(
            self, "", palette, fonts, fg=palette.fg_muted, font=fonts.mono_small
        )
        self.status.pack(fill="x", pady=(8, 0))

    # ------------------------------------------------------------------ api
    def load(self, provider: ProviderConfig | None) -> None:
        if provider is None:
            return
        self.name.set(provider.name)
        self.base_url.set(provider.base_url)
        self.api_key.set(provider.api_key)
        self.model.set(provider.model)
        self.temperature.set(str(provider.temperature))
        self.max_tokens.set(str(provider.max_tokens) if provider.max_tokens else "")
        self.timeout.set(str(provider.timeout))
        self.json_mode.set(provider.json_mode)
        self.streaming.set(bool(provider.supports_streaming))
        self.notes.set(provider.notes)
        self.test_badge.set("belum dites", self._palette.fg_muted)
        self.models_hint.configure(text="")
        self.status.configure(text="")

    def collect(self, provider: ProviderConfig) -> ProviderConfig:
        """Write the form back into a provider, keeping unedited fields."""
        provider.name = self.name.get().strip() or provider.name
        provider.base_url = self.base_url.get().strip() or provider.base_url
        provider.api_key = self.api_key.get().strip()
        provider.model = self.model.get().strip() or provider.model
        provider.temperature = _safe_float(self.temperature.get(), provider.temperature)
        # Empty means "no cap" (0), which is the default and the only setting that
        # cannot truncate a reasoning model mid-answer.
        raw_cap = self.max_tokens.get().strip()
        provider.max_tokens = int(_safe_float(raw_cap, 0)) if raw_cap else 0
        provider.timeout = _safe_float(self.timeout.get(), provider.timeout)
        provider.json_mode = self.json_mode.get()
        provider.supports_streaming = bool(self.streaming.get())
        provider.notes = self.notes.get().strip()
        return provider

    def set_test_result(self, ok: bool, message: str) -> None:
        self.test_badge.set(
            "siap" if ok else "gagal",
            self._palette.mint if ok else self._palette.red,
            "\u2713" if ok else "\u2715",
        )
        self.status.set_text(message)
        self.status.configure(fg=self._palette.mint if ok else self._palette.red)

    def show_models(self, models: list[str]) -> None:
        if not models:
            self.models_hint.configure(text="endpoint tidak mengembalikan daftar model")
            return
        self.models_hint.configure(
            text=f"{len(models)} model: {', '.join(models[:3])}\u2026"
        )


class RoutingTab(tk.Frame):
    """Per-stage provider, model and temperature assignment.

    Two defects made this tab useless, and both are fixed here:

    * ``collect()`` used to be reachable before ``load()`` had ever run (opening the
      tab called it via the dialog's collect-all path). It read empty widgets and
      overwrote the saved overrides with blanks, so the user's routing was destroyed
      simply by *looking* at the tab. ``collect()`` now refuses until armed by a
      ``load()``, which makes that impossible by construction rather than by
      remembering to reorder two calls.
    * the model was a free-text box with no memory of what the endpoint offers, so
      picking a model per stage meant retyping exact ids from ``/models``. A cached
      model list is fetched once in the background and offered as a dropdown that
      still accepts free text.
    """

    def __init__(self, master, palette: Palette, fonts: Fonts,
                 *, on_fetch_models: Callable[[], list[str]] | None = None) -> None:
        super().__init__(master, bg=palette.surface)
        self._palette = palette
        self._fonts = fonts
        self.rows: dict[str, dict[str, tk.Widget]] = {}
        self._on_fetch_models = on_fetch_models
        #: Guards against collecting a form that was never populated.
        self._loaded = False
        self._model_cache: dict[str, list[str]] = {}
        #: The settings the rows were loaded from; the resolved label asks this
        #: object what each stage will use, so the label cannot drift from reality.
        self._live_settings: AppSettings | None = None

        intro = (
            "Model kuat untuk pencipta nada & lirik, model murah untuk reviewer. "
            "Kosongkan provider untuk memakai provider aktif."
        )
        AutoWrapLabel(self, intro, palette, fonts).pack(fill="x", pady=(0, 10))

        header = tk.Frame(self, bg=palette.surface)
        header.pack(fill="x")
        for text, width in (
            ("TAHAP", 20), ("PROVIDER", 18), ("MODEL (OPSIONAL)", 20),
            ("TEMP", 6), ("MAX TOKEN", 9), ("AKTIF", 6),
        ):
            tk.Label(
                header, text=text, bg=palette.surface, fg=palette.fg_faint,
                font=fonts.label, width=width, anchor="w",
            ).pack(side="left")

        body = ScrollFrame(self, palette)
        body.pack(fill="both", expand=True, pady=(4, 0))
        self._scroll = body
        for stage in STAGE_ORDER:
            row = tk.Frame(body.inner, bg=palette.surface)
            row.pack(fill="x", pady=1)
            info = STAGE_INFO[stage]
            tk.Label(
                row,
                text=truncate(info.title, 20),
                bg=palette.surface,
                fg=palette.fg,
                font=fonts.small,
                width=20,
                anchor="w",
            ).pack(side="left")
            provider_var = tk.StringVar()
            combo = tk.OptionMenu(row, provider_var, "")
            combo.configure(
                bg=palette.surface_alt, fg=palette.fg, font=fonts.small,
                highlightthickness=0, bd=0, activebackground=palette.surface_high,
                activeforeground=palette.accent, width=16, anchor="w",
            )
            combo["menu"].configure(
                bg=palette.surface_alt, fg=palette.fg, font=fonts.small,
                activebackground=palette.accent, activeforeground=palette.bg,
            )
            combo.pack(side="left", padx=(0, 6))
            model_entry = ttk.Combobox(
                row, font=fonts.mono_small, width=26, values=[],
                state="normal",
            )
            model_entry.pack(side="left", padx=(0, 6), ipady=2)
            temp_entry = tk.Entry(
                row, bg=palette.surface_alt, fg=palette.fg, font=fonts.mono_small,
                relief="flat", bd=0, highlightthickness=1,
                highlightbackground=palette.line_soft, highlightcolor=palette.accent,
                width=5, insertbackground=palette.accent,
            )
            temp_entry.pack(side="left", padx=(0, 6), ipady=3)
            # An output cap is exactly the setting that silently ruins a run when it
            # is too small, so it has to be visible. Blank or 0 means no cap: the
            # request then carries no max_tokens and the model uses its own maximum.
            token_entry = tk.Entry(
                row, bg=palette.surface_alt, fg=palette.fg, font=fonts.mono_small,
                relief="flat", bd=0, highlightthickness=1,
                highlightbackground=palette.line_soft, highlightcolor=palette.accent,
                width=8, insertbackground=palette.accent,
            )
            token_entry.pack(side="left", padx=(0, 6), ipady=3)
            enabled = tk.BooleanVar(value=True)
            tk.Checkbutton(
                row, variable=enabled, bg=palette.surface, fg=palette.fg,
                selectcolor=palette.surface_alt, activebackground=palette.surface,
                bd=0, highlightthickness=0,
                command=lambda s=stage: self._sync_resolved(s),
            ).pack(side="left")
            resolved = tk.Label(
                row, text="", bg=palette.surface, fg=palette.fg_faint,
                font=fonts.mono_small, anchor="w",
            )
            resolved.pack(side="left", padx=(8, 0))
            # Choosing a provider re-seeds the model choices for that provider.
            provider_var.trace_add(
                "write", lambda *_a, s=stage: self._on_provider_change(s)
            )
            self.rows[stage] = {
                "provider_var": provider_var,
                "provider_menu": combo,
                "model": model_entry,
                "temp": temp_entry,
                "tokens": token_entry,
                "enabled": enabled,
                "resolved": resolved,
            }

    def load(self, settings: AppSettings) -> None:
        """Populate every row from the resolved routing, then allow collecting.

        The provider shown is the one the stage will *actually* use, and the model
        placeholder shows what it inherits when left blank - so an empty box reads as
        "inherit <model>" instead of as a blank the user must fill.
        """
        for stage, widgets in self.rows.items():
            override = settings.stage_overrides.get(stage) or StageOverride()
            target = settings.routing_for(stage)

            # Provider chooser: keyed by id, displayed by name, so two providers
            # that share a name cannot be confused for one another.
            menu: tk.OptionMenu = widgets["provider_menu"]  # type: ignore[assignment]
            menu["menu"].delete(0, "end")
            menu["menu"].add_command(
                label=INHERIT_LABEL,
                command=lambda var=widgets["provider_var"]: var.set(INHERIT_LABEL),
            )
            for provider in settings.providers:
                label = f"{provider.name} ({provider.model})"
                menu["menu"].add_command(
                    label=label,
                    command=lambda value=provider.id, var=widgets["provider_var"]: var.set(value),
                )
            widgets["provider_var"].set(override.provider_id or INHERIT_LABEL)

            # Model: an editable combobox seeded from whatever we know about the
            # selected provider, plus whatever is saved.
            model_box = widgets["model"]
            known = self._known_models(settings, override.provider_id or (target.provider.id if target.provider else ""))
            saved_model = override.model or ""
            self._fill_model_choices(model_box, known, saved_model, target.model)
            model_box.delete(0, "end")
            model_box.insert(0, saved_model)

            temp_entry: tk.Entry = widgets["temp"]  # type: ignore[assignment]
            temp_entry.delete(0, "end")
            temp_entry.insert(
                0,
                str(
                    override.temperature
                    if override.temperature is not None
                    else STAGE_TEMPERATURE_HINTS.get(stage, 0.7)
                ),
            )
            token_entry: tk.Entry = widgets["tokens"]  # type: ignore[assignment]
            token_entry.delete(0, "end")
            token_entry.insert(0, str(override.max_tokens) if override.max_tokens else "")
            widgets["enabled"].set(bool(override.enabled))  # type: ignore[union-attr]
            widgets["resolved"].configure(text=target.describe())  # type: ignore[union-attr]
        self._loaded = True

    # ------------------------------------------------------------------ models
    def _known_models(self, settings: AppSettings, provider_id: str) -> list[str]:
        """Model ids we can offer for a provider: cache first, then its default."""
        cached = self._model_cache.get(provider_id)
        if cached:
            return cached
        provider = settings.provider_by_id(provider_id)
        return [provider.model] if provider and provider.model else []

    def _fill_model_choices(
        self, widget, known: list[str], saved: str, inherited: str
    ) -> None:
        """Load a combobox with known ids, keeping it editable."""
        values = list(dict.fromkeys([saved, *known, inherited]))
        try:
            widget.configure(values=[v for v in values if v])
        except tk.TclError:  # pragma: no cover - plain Entry fallback
            pass

    def fetch_models_for(self, provider_id: str, models: list[str]) -> None:
        """Store a fetched model list and refresh the affected rows."""
        if not models:
            return
        self._model_cache[provider_id] = models
        for widgets in self.rows.values():
            if widgets["provider_var"].get() == provider_id:
                current = widgets["model"].get()
                self._fill_model_choices(widgets["model"], models, current, current)

    def collect(self, settings: AppSettings) -> None:
        """Write the form back into ``settings``.

        Refuses to run before :meth:`load`, because the only thing an unpopulated
        form can do is overwrite real configuration with blanks. That is precisely
        how the saved routing used to be destroyed just by opening this tab.
        """
        if not self._loaded:
            return
        for stage, widgets in self.rows.items():
            chosen = widgets["provider_var"].get()
            provider = (
                None if chosen == INHERIT_LABEL else settings.provider_by_id(chosen)
            )
            override = settings.stage_overrides.get(stage) or StageOverride()
            if not override.enabled and override.provider_id and not chosen:
                # Defensive: an override that already exists stays put unless the
                # form was actually populated for it.
                continue
            override.provider_id = provider.id if provider else ""
            override.model = widgets["model"].get().strip()
            override.temperature = _safe_float(
                widgets["temp"].get(), STAGE_TEMPERATURE_HINTS.get(stage, 0.7)
            )
            override.enabled = bool(widgets["enabled"].get())
            settings.stage_overrides[stage] = override

    def _provider_id_for(self, stage: str) -> str:
        chosen = self.rows[stage]["provider_var"].get()
        return "" if chosen == INHERIT_LABEL else chosen

    def _on_provider_change(self, stage: str) -> None:
        """Re-seed model suggestions and refresh the resolved label."""
        widgets = self.rows[stage]
        provider_id = self._provider_id_for(stage)
        known = self._model_cache.get(provider_id, [])
        if not known and self._on_fetch_models is not None and provider_id:
            known = self._on_fetch_models(provider_id) or []
            if known:
                self._model_cache[provider_id] = known
        current = widgets["model"].get()
        self._fill_model_choices(widgets["model"], known, current, current)
        self._sync_resolved(stage)

    def _sync_resolved(self, stage: str) -> None:
        """Show what the stage will actually use, read from the live settings.

        Deliberately not a preview computed by the widget: it asks the same
        ``routing_for`` the orchestrator asks, so the label cannot drift from
        behaviour.
        """
        widgets = self.rows[stage]
        label: tk.Label = widgets["resolved"]  # type: ignore[assignment]
        if self._live_settings is None:
            return
        probe = _clone_settings(self._live_settings)
        self.collect_into(probe)
        target = probe.routing_for(stage)
        label.configure(
            text=target.describe(),
            fg=self._palette.mint if target.overridden else self._palette.fg_faint,
        )

    def collect_into(self, settings: AppSettings) -> None:
        """Apply only the rows that have been loaded, without clearing others."""
        if not self._loaded:
            return
        for stage, widgets in self.rows.items():
            chosen = widgets["provider_var"].get()
            provider = (
                None if chosen == INHERIT_LABEL else settings.provider_by_id(chosen)
            )
            override = settings.stage_overrides.get(stage) or StageOverride()
            override.provider_id = provider.id if provider else ""
            override.model = widgets["model"].get().strip()
            override.temperature = _safe_float(
                widgets["temp"].get(), STAGE_TEMPERATURE_HINTS.get(stage, 0.7)
            )
            raw_tokens = widgets["tokens"].get().strip()
            override.max_tokens = int(_safe_float(raw_tokens, 0)) if raw_tokens else None
            override.enabled = bool(widgets["enabled"].get())
            settings.stage_overrides[stage] = override

    def mark_stale(self) -> None:
        """Force the next tab visit to reload, e.g. after providers change."""
        self._loaded = False


class SettingsDialog(tk.Toplevel):
    """Modal settings window: providers, stage routing, pipeline, about."""

    def __init__(
        self,
        master,
        palette: Palette,
        fonts: Fonts,
        settings: AppSettings,
        store: SettingsStore,
        *,
        on_saved: Callable[[AppSettings], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._palette = palette
        self._fonts = fonts
        self._store = store
        self._on_saved = on_saved
        self.settings = _clone_settings(settings)
        self.selected_id = self.settings.active_provider_id or (
            self.settings.providers[0].id if self.settings.providers else ""
        )

        self.title("Pengaturan Lirik Studio")
        self.configure(bg=palette.bg)
        self.geometry("1040x700")
        self.minsize(880, 600)
        self.transient(master)
        self.grab_set()

        shell = tk.Frame(self, bg=palette.bg)
        shell.pack(fill="both", expand=True, padx=16, pady=16)

        head = tk.Frame(shell, bg=palette.bg)
        head.pack(fill="x")
        tk.Label(
            head, text="Pengaturan", bg=palette.bg, fg=palette.fg,
            font=fonts.display_m, anchor="w",
        ).pack(side="left")
        self.save_state = tk.Label(
            head, text="", bg=palette.bg, fg=palette.fg_faint, font=fonts.small, anchor="e"
        )
        self.save_state.pack(side="right")

        body = tk.Frame(shell, bg=palette.bg)
        body.pack(fill="both", expand=True, pady=(12, 12))

        self.tabs = tk.Frame(body, bg=palette.bg)
        self.tabs.pack(fill="x")
        self._tab_buttons: dict[str, PillButton] = {}
        for key, label in (
            ("providers", "Provider LLM"),
            ("routing", "Routing per tahap"),
            ("pipeline", "Pipeline"),
            ("about", "Tentang"),
        ):
            button = PillButton(
                self.tabs, label, (lambda k=key: self._show_tab(k)), palette, fonts,
                kind="ghost", height=28,
            )
            button.pack(side="left", padx=(0, 6))
            self._tab_buttons[key] = button

        self.stage_frame = tk.Frame(body, bg=palette.bg)
        self.stage_frame.pack(fill="both", expand=True)

        self.pages: dict[str, tk.Frame] = {
            "providers": self._build_providers(self.stage_frame),
            "routing": self._build_routing(self.stage_frame),
            "pipeline": self._build_pipeline(self.stage_frame),
            "about": self._build_about(self.stage_frame),
        }

        footer = tk.Frame(shell, bg=palette.bg)
        footer.pack(fill="x")
        PillButton(
            footer, "Simpan", self._on_save, palette, fonts, kind="primary",
            glyph="\u2713", height=32,
        ).pack(side="right")
        PillButton(
            footer, "Batal", self.destroy, palette, fonts, kind="ghost", height=32,
        ).pack(side="right", padx=8)

        self._render_provider_list()
        self._show_tab("providers")
        self.bind("<Escape>", lambda _e: self.destroy())

    # ------------------------------------------------------------------ tabs
    def _build_providers(self, parent) -> tk.Frame:
        page = tk.Frame(parent, bg=self._palette.bg)
        left = Card(page, self._palette, self._fonts, eyebrow="Daftar", title="Provider")
        left.configure(width=320)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.provider_list = ProviderList(
            left.body,
            self._palette,
            self._fonts,
            on_select=self._select_provider,
            on_add=self._add_provider,
            on_duplicate=self._duplicate_provider,
            on_remove=self._remove_provider,
            on_preset=self._add_preset,
            on_import=self._import_pack,
            on_export=self._export_pack,
        )
        self.provider_list.pack(fill="both", expand=True)

        right = Card(page, self._palette, self._fonts, eyebrow="Editor", title="Detail provider")
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        scroll = ScrollFrame(right.body, self._palette)
        scroll.pack(fill="both", expand=True)
        self.provider_editor = ProviderEditor(
            scroll.inner,
            self._palette,
            self._fonts,
            on_test=self._test_provider,
            on_save=self._save_provider,
            on_fetch_models=self._fetch_models,
        )
        self.provider_editor.pack(fill="both", expand=True)
        return page

    def _build_routing(self, parent) -> tk.Frame:
        page = tk.Frame(parent, bg=self._palette.bg)
        card = Card(page, self._palette, self._fonts, eyebrow="Kontrol", title="Routing per tahap")
        card.pack(fill="both", expand=True)
        self.routing = RoutingTab(
            card.body, self._palette, self._fonts, on_fetch_models=self._models_for_routing
        )
        self.routing.pack(fill="both", expand=True)
        return page

    def _models_for_routing(self, provider_id: str) -> list[str]:
        """Fetch a provider's model list once, for the routing dropdowns.

        Runs synchronously and only on request: this is a LAN endpoint, and a
        cached list beats retyping exact model ids for nine stages.
        """
        provider = self.settings.provider_by_id(provider_id)
        if provider is None:
            return []
        try:
            return LLMClient(provider).list_models()
        except LLMError:
            return []

    def _build_pipeline(self, parent) -> tk.Frame:
        page = tk.Frame(parent, bg=self._palette.bg)
        card = Card(page, self._palette, self._fonts, eyebrow="Perilaku", title="Pipeline & revisi")
        card.pack(side="left", fill="both", expand=True)
        body = card.body

        self.max_revisions = TextField(
            body, self._palette, self._fonts, "Batas revisi total", width=10,
            hint="Berapa kali reviewer boleh memutar ulang pipeline dalam satu run.",
        )
        self.max_revisions.pack(anchor="w", pady=(0, 10))
        self.stage_attempts = TextField(
            body, self._palette, self._fonts, "Percobaan per tahap", width=10,
            hint="Berapa kali satu tahap diulang sebelum error dianggap fatal.",
        )
        self.stage_attempts.pack(anchor="w", pady=(0, 10))

        self.auto_brief = tk.BooleanVar()
        tk.Checkbutton(
            body, text="Izinkan pipeline berjalan tanpa percakapan (pakai brief apa adanya)",
            variable=self.auto_brief, bg=self._palette.surface, fg=self._palette.fg,
            selectcolor=self._palette.surface_alt, activebackground=self._palette.surface,
            activeforeground=self._palette.accent, font=self._fonts.small,
            bd=0, highlightthickness=0, anchor="w", wraplength=420, justify="left",
        ).pack(anchor="w", pady=(6, 0))

        self.offline_demo = tk.BooleanVar()
        tk.Checkbutton(
            body,
            text="Mode demo offline (agent tiruan, tanpa provider - untuk mencoba UI)",
            variable=self.offline_demo, bg=self._palette.surface, fg=self._palette.fg,
            selectcolor=self._palette.surface_alt, activebackground=self._palette.surface,
            activeforeground=self._palette.accent, font=self._fonts.small,
            bd=0, highlightthickness=0, anchor="w", wraplength=420, justify="left",
        ).pack(anchor="w", pady=(4, 0))

        self.output_dir = TextField(
            body, self._palette, self._fonts, "Folder hasil", width=48,
            hint="Kosongkan untuk memakai folder bawaan di AppData.",
        )
        self.output_dir.pack(fill="x", pady=(12, 0))
        PillButton(
            body, "Pilih folder", self._pick_output_dir, self._palette, self._fonts,
            kind="ghost", height=24, glyph="\u2026",
        ).pack(anchor="w", pady=(6, 0))

        side = Card(page, self._palette, self._fonts, eyebrow="Catatan", title="Cara kerja loop")
        side.pack(side="left", fill="both", expand=True, padx=(12, 0))
        tk.Label(
            side.body,
            text=(
                "Setiap reviewer punya dua lapis penilaian:\n\n"
                "1. Gate mekanis dihitung Python - batas karakter, tag, negasi, "
                "suku kata, gate klise. Ini fakta dan tidak bisa dibantah model.\n\n"
                "2. Skor dimensi dari model untuk hal yang memang soal rasa.\n\n"
                "Kalau tidak lulus, reviewer menyebut tahap akar masalahnya, dan "
                "pipeline mundur ke tahap itu - bukan selalu mundur satu langkah. "
                "Kalau permintaan reviewer tidak masuk akal, app membatasi ke tahap "
                "legal terdekat, dan bila tahap yang sama gagal berulang kali, "
                "target mundurnya otomatis naik satu tahap (maksimum sesuai batas)."
            ),
            bg=self._palette.surface,
            fg=self._palette.fg_muted,
            font=self._fonts.small,
            justify="left",
            anchor="nw",
            wraplength=400,
        ).pack(fill="both", expand=True)
        return page

    def _build_about(self, parent) -> tk.Frame:
        page = tk.Frame(parent, bg=self._palette.bg)
        card = Card(page, self._palette, self._fonts, eyebrow="Tim", title="Sembilan spesialis")
        card.pack(fill="both", expand=True)
        scroll = ScrollFrame(card.body, self._palette)
        scroll.pack(fill="both", expand=True)
        for index, stage in enumerate(STAGE_ORDER, start=1):
            info = STAGE_INFO[stage]
            block = tk.Frame(scroll.inner, bg=self._palette.surface)
            block.pack(fill="x", pady=(0, 10))
            tk.Label(
                block,
                text=f"{index}. {info.title}  \u00b7  {info.role}",
                bg=self._palette.surface,
                fg=self._palette.stage_colour(info.accent),
                font=self._fonts.body_bold,
                anchor="w",
            ).pack(anchor="w")
            AutoWrapLabel(block, info.specialty, self._palette, self._fonts).pack(fill="x")
            tk.Label(
                block,
                text=f"keluaran: {info.produces}",
                bg=self._palette.surface,
                fg=self._palette.fg_faint,
                font=self._fonts.mono_small,
                anchor="w",
            ).pack(anchor="w")
        return page

    def _show_tab(self, key: str) -> None:
        for page in self.pages.values():
            page.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        for name, button in self._tab_buttons.items():
            button.configure(bg=self._palette.accent if name == key else self._palette.surface_alt)
            button.redraw()
        if key == "routing":
            # Load FIRST, and never through _collect_all: that path used to run the
            # tab's collect() on an unpopulated form, wiping the saved routing.
            self.routing._live_settings = self.settings
            self.routing.load(self._collect_all_providers_only())
        if key == "pipeline":
            self._load_pipeline_fields()

    def _collect_all_providers_only(self) -> AppSettings:
        """Fold in the provider editor and pipeline fields, leaving routing alone."""
        settings = self.settings
        settings.max_revisions = int(_safe_float(self.max_revisions.get(), settings.max_revisions))
        settings.stage_attempts = int(_safe_float(self.stage_attempts.get(), settings.stage_attempts))
        settings.auto_approve_brief = bool(self.auto_brief.get())
        settings.offline_demo = bool(self.offline_demo.get())
        settings.output_dir = self.output_dir.get().strip()
        selected = settings.provider_by_id(self.selected_id)
        if selected is not None:
            self.provider_editor.collect(selected)
        return settings

    # ------------------------------------------------------------------ provider ops
    def _render_provider_list(self) -> None:
        self.provider_list.render(self.settings.providers, self.selected_id)

    def _select_provider(self, provider_id: str) -> None:
        self._save_provider(silent=True)
        self.selected_id = provider_id
        self._render_provider_list()
        self.provider_editor.load(self.settings.provider_by_id(provider_id))

    def _add_provider(self) -> None:
        self._save_provider(silent=True)
        provider = ProviderConfig(
            id=_unique_id("provider", self.settings.providers),
            name="Provider baru",
            base_url="http://127.0.0.1:8000/v1",
            model="model-name",
            notes="OpenAI-compatible endpoint.",
        )
        self.settings.providers.append(provider)
        self.selected_id = provider.id
        self._render_provider_list()
        self.routing.mark_stale()
        self.provider_editor.load(provider)

    def _duplicate_provider(self) -> None:
        self._save_provider(silent=True)
        source = self.settings.provider_by_id(self.selected_id)
        if source is None:
            return
        clone = ProviderConfig.from_dict(source.to_dict())
        clone.id = _unique_id(source.id, self.settings.providers)
        clone.name = f"{source.name} (copy)"
        self.settings.providers.append(clone)
        self.selected_id = clone.id
        self._render_provider_list()
        self.routing.mark_stale()
        self.provider_editor.load(clone)

    def _remove_provider(self) -> None:
        if len(self.settings.providers) <= 1:
            messagebox.showinfo("Tidak bisa dihapus", "Minimal satu provider harus ada.", parent=self)
            return
        provider = self.settings.provider_by_id(self.selected_id)
        if provider is None:
            return
        if not messagebox.askyesno(
            "Hapus provider", f"Hapus '{provider.name}'?", parent=self
        ):
            return
        self.settings.providers = [p for p in self.settings.providers if p.id != provider.id]
        for stage, override in self.settings.stage_overrides.items():
            if override.provider_id == provider.id:
                override.provider_id = ""
        self.selected_id = self.settings.providers[0].id
        self._render_provider_list()
        self.provider_editor.load(self.settings.providers[0])
        self.routing.mark_stale()

    def _add_preset(self, preset_id: str) -> None:
        preset = preset_provider(preset_id)
        if preset is None:
            return
        preset.id = _unique_id(preset.id, self.settings.providers)
        self.settings.providers.append(preset)
        self.selected_id = preset.id
        self._render_provider_list()
        self.routing.mark_stale()
        self.provider_editor.load(preset)
        self._set_state(f"Preset '{preset.name}' ditambahkan. Isi API key bila perlu.")

    def _import_pack(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Impor pak provider", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        try:
            self.settings.providers = import_providers(path, self.settings.providers)
        except Exception as exc:  # noqa: BLE001 - surfaced in a dialog
            messagebox.showerror("Impor gagal", str(exc), parent=self)
            return
        self.selected_id = self.settings.providers[-1].id
        self._render_provider_list()
        self.provider_editor.load(self.settings.providers[-1])
        self._set_state(f"Impor berhasil: {Path(path).name}")

    def _export_pack(self) -> None:
        self._save_provider(silent=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="Ekspor pak provider", defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        export_providers(self.settings.providers, path)
        self._set_state(f"Ekspor berhasil: {Path(path).name}")

    def _save_provider(self, silent: bool = False) -> None:
        provider = self.settings.provider_by_id(self.selected_id)
        if provider is None:
            return
        self.provider_editor.collect(provider)
        problems = provider.validate()
        if problems and not silent:
            self.provider_editor.set_test_result(False, "; ".join(problems))
            return
        if provider.id == self.settings.active_provider_id:
            pass
        self._render_provider_list()
        if not silent:
            self._set_state(f"Provider '{provider.name}' disimpan (belum ditulis ke disk).")

    def _test_provider(self) -> None:
        provider = self.settings.provider_by_id(self.selected_id)
        if provider is None:
            return
        self.provider_editor.collect(provider)
        problems = provider.validate()
        if problems:
            self.provider_editor.set_test_result(False, "; ".join(problems))
            return
        self._set_state(f"Menguji {provider.name}\u2026")
        self.update_idletasks()
        client = LLMClient(provider)
        ok, message, latency = client.test_connection()
        self.provider_editor.set_test_result(ok, message)
        self._set_state(f"Tes {provider.name}: {message}")

    def _fetch_models(self) -> None:
        provider = self.settings.provider_by_id(self.selected_id)
        if provider is None:
            return
        self.provider_editor.collect(provider)
        self._set_state(f"Mengambil daftar model dari {provider.models_url}\u2026")
        self.update_idletasks()
        try:
            models = LLMClient(provider).list_models()
        except LLMError as exc:
            self.provider_editor.set_test_result(False, str(exc))
            return
        self.provider_editor.show_models(models)
        if models:
            self._set_state(f"{len(models)} model tersedia; contoh: {', '.join(models[:5])}")

    # ------------------------------------------------------------------ pipeline fields
    def _load_pipeline_fields(self) -> None:
        self.max_revisions.set(str(self.settings.max_revisions))
        self.stage_attempts.set(str(self.settings.stage_attempts))
        self.auto_brief.set(bool(self.settings.auto_approve_brief))
        self.offline_demo.set(bool(self.settings.offline_demo))
        self.output_dir.set(self.settings.output_dir)

    def _pick_output_dir(self) -> None:
        path = filedialog.askdirectory(parent=self, title="Pilih folder hasil")
        if path:
            self.output_dir.set(path)

    def _collect_all(self) -> AppSettings:
        """Fold every tab's widgets back into the settings object."""
        settings = self.settings
        settings.max_revisions = int(_safe_float(self.max_revisions.get(), settings.max_revisions))
        settings.stage_attempts = int(_safe_float(self.stage_attempts.get(), settings.stage_attempts))
        settings.auto_approve_brief = bool(self.auto_brief.get())
        settings.offline_demo = bool(self.offline_demo.get())
        settings.output_dir = self.output_dir.get().strip()
        selected = settings.provider_by_id(self.selected_id)
        if selected is not None:
            self.provider_editor.collect(selected)
        # collect() is a no-op until the tab has been loaded, so saving from the
        # provider tab can never blank out routing the user configured earlier.
        self.routing.collect(settings)
        return settings

    def _on_save(self) -> None:
        settings = self._collect_all()
        settings.ensure_defaults()
        try:
            self._store.save(settings)
        except OSError as exc:
            messagebox.showerror("Gagal menyimpan", str(exc), parent=self)
            return
        if self._on_saved is not None:
            self._on_saved(settings)
        self.destroy()

    def set_active_provider(self, provider_id: str) -> None:
        """Mark a provider as the pipeline default (called from the main window)."""
        self.settings.active_provider_id = provider_id
        self._set_state("Provider aktif diperbarui.")

    def _set_state(self, text: str) -> None:
        self.save_state.configure(text=truncate(text, 90))


def _safe_float(value: str, default: float) -> float:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return float(default)


def _unique_id(base: str, existing: list[ProviderConfig]) -> str:
    taken = {p.id for p in existing}
    candidate = slugify(base)
    if candidate not in taken:
        return candidate
    index = 2
    while f"{candidate}-{index}" in taken:
        index += 1
    return f"{candidate}-{index}"


def _clone_settings(settings: AppSettings) -> AppSettings:
    """Deep copy so Cancel really cancels."""
    return AppSettings.from_dict(settings.to_dict())


__all__ = ["SettingsDialog", "ProviderEditor", "ProviderList", "RoutingTab", "TextField"]

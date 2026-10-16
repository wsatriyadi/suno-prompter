"""JSON persistence for settings, presets and finished runs."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .models import AppSettings, ProviderConfig, user_config_dir


class SettingsError(RuntimeError):
    """Raised when settings on disk are unusable."""


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically so a crash cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


class SettingsStore:
    """Loads and saves :class:`AppSettings` as a single JSON document.

    A corrupt file is never silently destroyed: it is moved aside to
    ``settings.json.corrupt-<timestamp>`` and defaults are returned.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else user_config_dir() / "settings.json"

    def load(self) -> AppSettings:
        if not self.path.exists():
            settings = AppSettings()
            settings.ensure_defaults()
            return settings
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._quarantine(exc)
            settings = AppSettings()
            settings.ensure_defaults()
            return settings
        if not isinstance(raw, dict):
            self._quarantine(ValueError("settings root is not an object"))
            settings = AppSettings()
            settings.ensure_defaults()
            return settings
        try:
            return AppSettings.from_dict(raw)
        except (TypeError, ValueError) as exc:
            self._quarantine(exc)
            settings = AppSettings()
            settings.ensure_defaults()
            return settings

    def save(self, settings: AppSettings) -> Path:
        _atomic_write(self.path, settings.to_json())
        return self.path

    def _quarantine(self, exc: Exception) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        broken = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        try:
            shutil.move(str(self.path), str(broken))
        except OSError:
            return self.path
        return broken


def export_providers(providers: list[ProviderConfig], path: Path | str) -> Path:
    """Write a provider pack users can share (keys included, as-is)."""
    target = Path(path)
    payload = {"version": 1, "providers": [p.to_dict() for p in providers]}
    _atomic_write(target, json.dumps(payload, indent=2, ensure_ascii=False))
    return target


def import_providers(path: Path | str, existing: list[ProviderConfig]) -> list[ProviderConfig]:
    """Merge a provider pack into ``existing``; ids already present are replaced."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    incoming = [ProviderConfig.from_dict(p) for p in raw.get("providers", [])]
    if not incoming:
        raise SettingsError("Provider pack contains no providers.")
    by_id = {p.id: p for p in existing}
    for provider in incoming:
        by_id[provider.id] = provider
    return list(by_id.values())


@dataclass
class RunRecord:
    """One finished pipeline run, on disk."""

    run_id: str
    title: str
    created_at: str
    directory: Path
    style_length: int = 0
    verdict: str = ""
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "title": self.title,
            "created_at": self.created_at,
            "directory": str(self.directory),
            "style_length": self.style_length,
            "verdict": self.verdict,
            "provider": self.provider,
        }


class RunStore:
    """Keeps the last N runs as ``<dir>/runs/<timestamp>-<slug>/`` folders."""

    def __init__(self, root: Path | str | None = None, keep: int = 100) -> None:
        self.root = Path(root) if root else user_config_dir() / "runs"
        self.keep = keep

    def new_dir(self, title: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", (title or "untitled").lower()).strip("-")[:40]
        directory = self.root / f"{stamp}-{slug or 'untitled'}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_run(
        self,
        title: str,
        files: dict[str, str],
        *,
        provider: str = "",
        verdict: str = "",
    ) -> RunRecord:
        directory = self.new_dir(title)
        for name, content in files.items():
            _atomic_write(directory / name, content)
        record = RunRecord(
            run_id=directory.name,
            title=title or "untitled",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            directory=directory,
            style_length=len(files.get("style.txt", "")),
            verdict=verdict,
            provider=provider,
        )
        _atomic_write(directory / "run.json", json.dumps(record.to_dict(), indent=2))
        self._prune()
        return record

    def list_runs(self) -> list[RunRecord]:
        if not self.root.exists():
            return []
        records: list[RunRecord] = []
        for child in sorted(self.root.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            meta = child / "run.json"
            if not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records.append(
                RunRecord(
                    run_id=child.name,
                    title=data.get("title", child.name),
                    created_at=data.get("created_at", ""),
                    directory=child,
                    style_length=int(data.get("style_length", 0)),
                    verdict=data.get("verdict", ""),
                    provider=data.get("provider", ""),
                )
            )
        return records

    def _prune(self) -> None:
        records = sorted([p for p in self.root.iterdir() if p.is_dir()])
        for stale in records[: max(0, len(records) - self.keep)]:
            shutil.rmtree(stale, ignore_errors=True)

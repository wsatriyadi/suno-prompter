"""Persistent settings model: providers, stage routing, app preferences.

Everything here is JSON-serialisable and versioned. The on-disk store lives in
:mod:`suno_agents.config.providers`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

SETTINGS_VERSION = 1

#: API key values may be indirect so secrets stay out of the config file.
ENV_KEY_PREFIX = "env:"


def resolve_secret(value: str) -> str:
    """Resolve ``env:NAME`` indirection; anything else is returned verbatim."""
    if value.startswith(ENV_KEY_PREFIX):
        return os.environ.get(value[len(ENV_KEY_PREFIX) :].strip(), "")
    return value


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "provider"


def user_config_dir() -> Path:
    """Per-user writable directory for settings, presets and run history."""
    override = os.environ.get("LIRIK_HOME")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif os.uname().sysname == "Darwin":  # pragma: no cover - platform specific
        base = Path.home() / "Library" / "Application Support"
    else:  # pragma: no cover - platform specific
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "lirik-studio"


@dataclass
class ProviderConfig:
    """One OpenAI-compatible chat-completions endpoint."""

    id: str = ""
    name: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.8
    top_p: float = 1.0
    #: Output cap. **0 means no cap**: the field is omitted from the request and the
    #: provider applies its own maximum.
    #:
    #: Default 0 on purpose. A reasoning model spends completion tokens on hidden
    #: reasoning before it emits a single visible character, so any fixed cap is a
    #: guess that can be wrong in the expensive direction: measured on one model,
    #: a 300-token cap produced 1032 characters of reasoning and zero characters of
    #: answer, and 16000 was still not enough for a real blueprint that needed
    #: ~15.7k tokens with additional reasoning on top. Sending no limit lets the
    #: endpoint decide, which is the only setting that cannot silently truncate.
    max_tokens: int = 0
    #: Seconds to wait for a response. Generous because an uncapped reply from a
    #: reasoning model can take several minutes, and a timeout mid-generation wastes
    #: the whole call.
    timeout: float = 600.0
    max_retries: int = 2
    json_mode: str = "object"  # object | schema | off
    supports_streaming: bool = True
    #: Extra HTTP headers, e.g. {"HTTP-Referer": "..."} for OpenRouter.
    extra_headers: dict[str, str] = field(default_factory=dict)
    #: Extra JSON merged into the request body, e.g. {"reasoning_effort": "low"}.
    extra_body: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = slugify(self.name or self.model or "provider")
        if not self.name:
            self.name = self.id

    @property
    def chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def models_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        return f"{base}/models"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key.strip()) or self.base_url.strip().startswith(
            ("http://localhost", "http://127.0.0.1", "http://0.0.0.0")
        )

    def resolved_key(self) -> str:
        return resolve_secret(self.api_key)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.base_url.strip().startswith(("http://", "https://")):
            problems.append("Base URL must start with http:// or https://")
        if not self.model.strip():
            problems.append("Model name is empty")
        if not 0.0 <= self.temperature <= 2.0:
            problems.append("Temperature must be between 0 and 2")
        if self.json_mode not in {"object", "schema", "off"}:
            problems.append("JSON mode must be object, schema or off")
        if self.max_tokens != 0 and self.max_tokens < 256:
            problems.append("Max tokens must be 0 (tanpa batas) or at least 256")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StageOverride:
    """Per-pipeline-stage model routing (cheap model for intake, etc.)."""

    provider_id: str = ""
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageOverride":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


DEFAULT_STAGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "intake": {"temperature": 0.6},
    "audio_analyst": {"temperature": 0.4},
    "composer": {"temperature": 0.75},
    "lyricist": {"temperature": 0.9},
    "lyric_reviewer": {"temperature": 0.2},
    "arranger": {"temperature": 0.5},
    "music_reviewer": {"temperature": 0.25},
    "prompt_engineer": {"temperature": 0.55},
    "prompt_reviewer": {"temperature": 0.2},
}


@dataclass(frozen=True)
class RoutingTarget:
    """Fully resolved per-stage routing decision.

    Resolved once, here, so the runtime and the settings UI cannot disagree about
    which provider/model/temperature a stage will actually use. Before this existed
    the UI showed the raw override while the orchestrator resolved its own answer,
    and the two drifted: the UI displayed an inherited model, and the runtime
    silently ignored the override's model entirely.
    """

    stage: str
    provider: ProviderConfig | None
    model: str
    temperature: float
    max_tokens: int | None
    #: True when this stage has a working override instead of inheriting.
    overridden: bool
    #: When not overridden, whether the model came from the active provider.
    inherited_from_active: bool

    @property
    def provider_name(self) -> str:
        return self.provider.name if self.provider is not None else "-"

    def describe(self) -> str:
        source = "override" if self.overridden else "inherits active provider"
        cap = "tanpa batas" if not self.max_tokens else f"{self.max_tokens} tok"
        return f"{self.provider_name} / {self.model} / {cap} [{source}]"


@dataclass
class AppSettings:
    """Everything the app persists besides provider credentials."""

    version: int = SETTINGS_VERSION
    active_provider_id: str = ""
    providers: list[ProviderConfig] = field(default_factory=list)
    stage_overrides: dict[str, StageOverride] = field(default_factory=dict)
    max_revisions: int = 24
    stage_attempts: int = 3
    auto_approve_brief: bool = True
    offline_demo: bool = False
    output_dir: str = ""
    last_export_dir: str = ""
    language: str = "id"
    theme: str = "console"
    window: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["providers"] = [p.to_dict() for p in self.providers]
        data["stage_overrides"] = {k: v.to_dict() for k, v in self.stage_overrides.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        settings = cls(
            version=int(data.get("version", SETTINGS_VERSION)),
            active_provider_id=data.get("active_provider_id", ""),
            providers=[ProviderConfig.from_dict(p) for p in data.get("providers", [])],
            stage_overrides={
                k: StageOverride.from_dict(v)
                for k, v in (data.get("stage_overrides") or {}).items()
            },
            max_revisions=int(data.get("max_revisions", 24)),
            stage_attempts=int(data.get("stage_attempts", 3)),
            auto_approve_brief=bool(data.get("auto_approve_brief", True)),
            offline_demo=bool(data.get("offline_demo", False)),
            output_dir=data.get("output_dir", ""),
            last_export_dir=data.get("last_export_dir", ""),
            language=data.get("language", "id"),
            theme=data.get("theme", "console"),
            window=dict(data.get("window") or {}),
        )
        settings.ensure_defaults()
        return settings

    def ensure_defaults(self) -> None:
        """Fill missing stage overrides and guarantee a usable provider list."""
        for stage, defaults in DEFAULT_STAGE_OVERRIDES.items():
            override = self.stage_overrides.get(stage)
            if override is None:
                override = StageOverride(enabled=True)
                self.stage_overrides[stage] = override
            if override.temperature is None:
                override.temperature = float(defaults.get("temperature", 0.7))
        if not self.providers:
            self.providers = [default_provider()]
        known = {p.id for p in self.providers}
        if self.active_provider_id not in known:
            self.active_provider_id = self.providers[0].id

    def provider_by_id(self, provider_id: str | None) -> ProviderConfig | None:
        if not provider_id:
            return None
        for provider in self.providers:
            if provider.id == provider_id:
                return provider
        return None

    @property
    def active_provider(self) -> ProviderConfig | None:
        return self.provider_by_id(self.active_provider_id)

    def effective_provider(self, stage_id: str) -> ProviderConfig | None:
        """Return the provider a stage should use, honouring per-stage routing."""
        base = self.active_provider
        override = self.stage_overrides.get(stage_id)
        if override and override.enabled and override.provider_id:
            candidate = self.provider_by_id(override.provider_id)
            if candidate is not None:
                return candidate
        return base

    def effective_temperature(self, stage_id: str) -> float:
        base = self.active_provider
        temp = base.temperature if base else 0.7
        override = self.stage_overrides.get(stage_id)
        if override and override.temperature is not None:
            return float(override.temperature)
        return temp

    def effective_model(self, stage_id: str) -> str:
        provider = self.effective_provider(stage_id)
        model = provider.model if provider else ""
        override = self.stage_overrides.get(stage_id)
        if override and override.enabled and override.model.strip():
            return override.model.strip()
        return model

    def effective_max_tokens(self, stage_id: str) -> int:
        """Resolved output cap for a stage. **0 means no cap.**

        A per-stage override of 0 is meaningful rather than absent: ``None`` inherits
        the provider, while an explicit 0 asks for the provider's own maximum. That
        distinction is what lets someone keep one stage bounded and leave the rest
        unbounded.
        """
        override = self.stage_overrides.get(stage_id)
        if override and override.enabled and override.max_tokens is not None:
            return max(0, int(override.max_tokens))
        provider = self.effective_provider(stage_id)
        return max(0, int(provider.max_tokens)) if provider is not None else 0

    def routing_for(self, stage_id: str) -> RoutingTarget:
        """The single resolved answer for "what runs this stage?".

        Every consumer - the orchestrator, the session, the settings dialog - asks
        this question through this method, so what the UI shows is by construction
        what the pipeline does.
        """
        override = self.stage_overrides.get(stage_id)
        active = self.active_provider
        provider = self.effective_provider(stage_id)
        overridden = bool(
            override
            and override.enabled
            and (override.provider_id or override.model.strip())
            and provider is not None
        )
        model = ""
        if overridden and override is not None and override.model.strip():
            model = override.model.strip()
        if not model:
            model = provider.model if provider is not None else ""
        return RoutingTarget(
            stage=stage_id,
            provider=provider,
            model=model,
            temperature=self.effective_temperature(stage_id),
            max_tokens=self.effective_max_tokens(stage_id),
            overridden=overridden,
            inherited_from_active=(
                provider is not None and active is not None and provider.id == active.id
            ),
        )

    def redacted(self) -> dict[str, Any]:
        """Settings dump safe for logs and bug reports."""
        data = self.to_dict()
        for provider in data["providers"]:
            if provider.get("api_key"):
                provider["api_key"] = "***"
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def default_provider() -> ProviderConfig:
    return ProviderConfig(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        notes="Cloud default. Paste an API key in Settings -> LLM Providers.",
    )


PRESET_PROVIDERS: list[dict[str, Any]] = [
    {
        "id": "openai",
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "notes": "Official OpenAI endpoint.",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-4.5",
        "extra_headers": {"HTTP-Referer": "https://localhost", "X-Title": "Lirik Studio"},
        "notes": "One key, many models. Great for per-stage routing.",
    },
    {
        "id": "ollama",
        "name": "Ollama (local)",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3:14b",
        "api_key": "ollama",
        "notes": "Local models, no key needed. Start with: ollama serve",
    },
    {
        "id": "lmstudio",
        "name": "LM Studio (local)",
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "local-model",
        "api_key": "lm-studio",
        "notes": "Enable the local server tab in LM Studio.",
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "notes": "Strong lyric writing for the price.",
    },
    {
        "id": "groq",
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "notes": "Very fast; good for reviewers on a budget.",
    },
    {
        "id": "custom",
        "name": "Custom OpenAI-compatible",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "your-model",
        "notes": "vLLM, llama.cpp server, Together, Fireworks, anything /v1.",
    },
]


def preset_provider(preset_id: str) -> ProviderConfig | None:
    for preset in PRESET_PROVIDERS:
        if preset["id"] == preset_id:
            return ProviderConfig.from_dict(preset)
    return None


def providers_from_presets(ids: Iterable[str]) -> list[ProviderConfig]:
    out: list[ProviderConfig] = []
    for pid in ids:
        provider = preset_provider(pid)
        if provider is not None:
            out.append(provider)
    return out

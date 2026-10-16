"""Domain contracts for the nine-specialist pipeline.

Every artifact that travels between agents is a dataclass defined here, with
``to_dict`` / ``from_dict`` / ``validate`` so the pipeline can persist a run,
resume it, and *check the model's work* instead of trusting it.

The golden rule encoded below: a review is not prose. It is a scoring sheet
with numeric dimensions, hard gates, machine-readable issues, and an explicit
``restart_from`` stage that the orchestrator validates against the stage graph
(:mod:`suno_agents.domain.graph`) before honouring it.

``from_dict`` on every type accepts an already-constructed instance and returns
it unchanged. That matters: the orchestrator hands artifacts back into agents,
and a second round-trip through the parser must be a no-op rather than a
silent mutation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_NOTE_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min)?", re.IGNORECASE)

_FLAT_TO_SHARP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "Fb": "E",
}


def as_float(value: Any, default: float = 0.0) -> float:
    """Tolerant float coercion; never raises, never returns NaN or infinity."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:[.,]\d+)?", value)
        if match:
            try:
                return float(match.group(0).replace(",", "."))
            except ValueError:
                return default
    return default


def as_int(value: Any, default: int = 0) -> int:
    return int(round(as_float(value, float(default))))


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "pass", "passed", "lulus", "ok"}:
            return True
        if lowered in {"false", "no", "n", "0", "fail", "failed", "gagal"}:
            return False
    return default


def as_str_list(value: Any) -> list[str]:
    """Coerce scalars / comma-joined strings / lists into a clean string list."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n;]|,(?![^(]*\))", value)
        return [p.strip(" -\t") for p in parts if p and p.strip(" -\t")]
    if isinstance(value, Mapping):
        return [str(v).strip() for v in value.values() if str(v).strip()]
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("value") or item.get("line")
                if text:
                    out.append(str(text).strip())
            elif item is not None:
                out.append(str(item).strip())
        return out
    return [str(value).strip()]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalise_key(raw: str) -> tuple[str, str]:
    """Split a free-text key into ``(root, mode)``; defaults to C major."""
    match = _NOTE_RE.match(raw or "")
    if not match:
        return "C", "major"
    letter = match.group(1).upper()
    accidental = match.group(2) or ""
    root = _FLAT_TO_SHARP.get(f"{letter}{accidental}", f"{letter}{accidental}")
    mode_raw = (match.group(3) or "major").lower()
    mode = "minor" if mode_raw.startswith("min") else "major"
    return root, mode


# --------------------------------------------------------------------------- #
# 1. Intake - conversation agent output
# --------------------------------------------------------------------------- #


@dataclass
class CreativeBrief:
    """What the user actually wants, in the user's own language plus structure."""

    working_title: str = ""
    concept: str = ""
    genre: list[str] = field(default_factory=list)
    subgenre: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)
    era_reference: str = ""
    tempo_bpm: float = 0.0
    tempo_feel: str = ""
    key: str = ""
    time_signature: str = "4/4"
    language: str = ""
    vocal_gender: str = "unspecified"  # male | female | duet | instrumental | unspecified
    vocal_type: str = ""  # e.g. "powerful male baritone", "breathy alto"
    vocal_delivery: str = ""
    instrumental: bool = False
    instrumentation: list[str] = field(default_factory=list)
    structure: list[str] = field(default_factory=list)
    reference_artists: list[str] = field(default_factory=list)
    reference_songs: list[str] = field(default_factory=list)
    theme: str = ""
    story_arc: str = ""
    must_include: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)
    target_platform: str = "suno"
    delivery_target: str = "24-bit/48 kHz lossless WAV master"
    notes_for_team: str = ""
    open_questions: list[str] = field(default_factory=list)
    user_expertise: str = "non-musician"  # non-musician | hobbyist | musician | producer
    confidence: float = 0.7

    @property
    def is_ready(self) -> bool:
        """A brief is workable when we know a direction and have a concept."""
        return bool((self.genre or self.mood or self.concept) and self.concept)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "CreativeBrief") -> "CreativeBrief":
        if isinstance(data, cls):
            return data
        data = data or {}
        instrumental = as_bool(data.get("instrumental"), False)
        gender = str(data.get("vocal_gender", "") or "unspecified").strip().lower()
        if instrumental:
            gender = "instrumental"
        if gender not in {"male", "female", "duet", "instrumental", "unspecified"}:
            gender = "unspecified"
        return cls(
            working_title=str(data.get("working_title", "") or "").strip(),
            concept=str(data.get("concept", "") or "").strip(),
            genre=as_str_list(data.get("genre")),
            subgenre=as_str_list(data.get("subgenre")),
            mood=as_str_list(data.get("mood")),
            era_reference=str(data.get("era_reference", "") or ""),
            tempo_bpm=as_float(data.get("tempo_bpm"), 0.0),
            tempo_feel=str(data.get("tempo_feel", "") or ""),
            key=str(data.get("key", "") or ""),
            time_signature=str(data.get("time_signature", "4/4") or "4/4"),
            language=str(data.get("language", "") or ""),
            vocal_gender=gender,
            vocal_type=str(data.get("vocal_type", "") or ""),
            vocal_delivery=str(data.get("vocal_delivery", "") or ""),
            instrumental=instrumental,
            instrumentation=as_str_list(data.get("instrumentation")),
            structure=as_str_list(data.get("structure")),
            reference_artists=as_str_list(data.get("reference_artists")),
            reference_songs=as_str_list(data.get("reference_songs")),
            theme=str(data.get("theme", "") or ""),
            story_arc=str(data.get("story_arc", "") or ""),
            must_include=as_str_list(data.get("must_include")),
            must_avoid=as_str_list(data.get("must_avoid")),
            target_platform=str(data.get("target_platform", "suno") or "suno"),
            delivery_target=str(
                data.get("delivery_target", "") or "24-bit/48 kHz lossless WAV master"
            ),
            notes_for_team=str(data.get("notes_for_team", "") or ""),
            open_questions=as_str_list(data.get("open_questions")),
            user_expertise=str(data.get("user_expertise", "") or "non-musician"),
            confidence=clamp(as_float(data.get("confidence"), 0.7), 0.0, 1.0),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.concept.strip():
            problems.append("Brief has no concept/creative direction.")
        if not self.genre and not self.mood:
            problems.append("Brief has neither genre nor mood.")
        if len(self.concept.split()) < 3:
            problems.append("Concept is too thin to write a song from (<3 words).")
        if self.tempo_bpm and not 40 <= self.tempo_bpm <= 220:
            problems.append(f"Tempo {self.tempo_bpm} BPM is outside 40-220.")
        if self.vocal_gender == "instrumental":
            self.instrumental = True
        return problems

    def summary(self) -> str:
        vocals = (
            "instrumental (no vocals requested)"
            if self.instrumental
            else f"{self.vocal_gender} | {self.vocal_type or '-'} | {self.vocal_delivery or '-'}"
        )
        lines = [
            f"Working title: {self.working_title or '(untitled)'}",
            f"Concept: {self.concept}",
            f"Genre: {', '.join(self.genre) or '-'}"
            + (f" (sub: {', '.join(self.subgenre)})" if self.subgenre else ""),
            f"Mood: {', '.join(self.mood) or '-'}",
            f"Era/reference: {self.era_reference or '-'}",
            f"Tempo: {self.tempo_bpm or 'free'} BPM {self.tempo_feel} | Key: {self.key or 'free'} "
            f"| Meter: {self.time_signature}",
            f"Language: {self.language or '-'}",
            f"Vocals: {vocals}",
            f"Instrumentation: {', '.join(self.instrumentation) or '-'}",
            f"Structure: {' > '.join(self.structure) or '-'}",
            f"References: {', '.join(self.reference_artists + self.reference_songs) or '-'}",
            f"Theme: {self.theme or '-'}",
            f"Story arc: {self.story_arc or '-'}",
            f"Must include: {', '.join(self.must_include) or '-'}",
            f"Must avoid: {', '.join(self.must_avoid) or '-'}",
            f"Delivery target: {self.delivery_target}",
            f"Notes: {self.notes_for_team or '-'}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Composer - the musical skeleton
# --------------------------------------------------------------------------- #


@dataclass
class ChordEvent:
    """One harmony event: ``beats`` beats of a chord starting at ``bar``."""

    bar: int = 1
    beats: float = 4.0
    symbol: str = ""  # "Am7", "Fadd9", "C/G"
    function: str = ""  # roman numeral, e.g. "vi7"
    voicing_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "ChordEvent") -> "ChordEvent":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            bar=as_int(data.get("bar"), 1),
            beats=as_float(data.get("beats"), 4.0),
            symbol=str(data.get("symbol", "") or "").strip(),
            function=str(data.get("function", "") or "").strip(),
            voicing_note=str(data.get("voicing_note", "") or "").strip(),
        )


@dataclass
class SectionPlan:
    """One song section: bars, harmony, energy, texture, what it must achieve."""

    name: str = ""
    bars: int = 8
    start_beat: float = 0.0
    chords: list[ChordEvent] = field(default_factory=list)
    energy: int = 5  # 0-10
    dynamics: str = ""
    texture: str = ""
    drums: str = ""
    bass: str = ""
    melodic_device: str = ""
    purpose: str = ""
    lyrics_needed: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chords"] = [c.to_dict() for c in self.chords]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "SectionPlan") -> "SectionPlan":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            name=str(data.get("name", "") or "").strip(),
            bars=as_int(data.get("bars"), 8),
            start_beat=as_float(data.get("start_beat"), 0.0),
            chords=[ChordEvent.from_dict(c) for c in data.get("chords") or []],
            energy=as_int(data.get("energy"), 5),
            dynamics=str(data.get("dynamics", "") or ""),
            texture=str(data.get("texture", "") or ""),
            drums=str(data.get("drums", "") or ""),
            bass=str(data.get("bass", "") or ""),
            melodic_device=str(data.get("melodic_device", "") or ""),
            purpose=str(data.get("purpose", "") or ""),
            lyrics_needed=as_bool(data.get("lyrics_needed"), True),
        )


@dataclass
class MusicBlueprint:
    """The composer's deliverable: key, groove, harmony, melodic DNA, arc."""

    key: str = "C"
    mode: str = "minor"
    scale_notes: list[str] = field(default_factory=list)
    tempo_bpm: float = 0.0
    time_signature: str = "4/4"
    groove: str = ""
    swing: str = ""
    harmonic_language: str = ""
    chord_progression_summary: str = ""
    sections: list[SectionPlan] = field(default_factory=list)
    total_bars: int = 0
    estimated_seconds: float = 0.0
    melodic_motif: str = ""
    motif_development: str = ""
    hook_definition: str = ""
    tension_curve: str = ""
    instrumentation: list[str] = field(default_factory=list)
    arrangement_notes: str = ""
    harmonic_risks: list[str] = field(default_factory=list)
    theory_rationale: str = ""
    delivery_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = [s.to_dict() for s in self.sections]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "MusicBlueprint") -> "MusicBlueprint":
        if isinstance(data, cls):
            return data
        data = data or {}
        root, mode = normalise_key(str(data.get("key", "") or "C"))
        declared_mode = str(data.get("mode", "") or "").strip().lower()
        if declared_mode.startswith("min"):
            mode = "minor"
        elif declared_mode.startswith("maj"):
            mode = "major"
        if str(data.get("key", "")).strip().lower().endswith("minor"):
            mode = "minor"
        return cls(
            key=root,
            mode=mode,
            scale_notes=as_str_list(data.get("scale_notes")),
            tempo_bpm=as_float(data.get("tempo_bpm"), 0.0),
            time_signature=str(data.get("time_signature", "4/4") or "4/4"),
            groove=str(data.get("groove", "") or ""),
            swing=str(data.get("swing", "") or ""),
            harmonic_language=str(data.get("harmonic_language", "") or ""),
            chord_progression_summary=str(data.get("chord_progression_summary", "") or ""),
            sections=[SectionPlan.from_dict(s) for s in data.get("sections") or []],
            total_bars=as_int(data.get("total_bars"), 0),
            estimated_seconds=as_float(data.get("estimated_seconds"), 0.0),
            melodic_motif=str(data.get("melodic_motif", "") or ""),
            motif_development=str(data.get("motif_development", "") or ""),
            hook_definition=str(data.get("hook_definition", "") or ""),
            tension_curve=str(data.get("tension_curve", "") or ""),
            instrumentation=as_str_list(data.get("instrumentation")),
            arrangement_notes=str(data.get("arrangement_notes", "") or ""),
            harmonic_risks=as_str_list(data.get("harmonic_risks")),
            theory_rationale=str(data.get("theory_rationale", "") or ""),
            delivery_notes=str(data.get("delivery_notes", "") or ""),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.sections:
            problems.append("Blueprint has no sections.")
        if not self.chord_progression_summary and not any(s.chords for s in self.sections):
            problems.append("Blueprint defines no harmony at all.")
        if self.tempo_bpm and not 40 <= self.tempo_bpm <= 220:
            problems.append(f"Blueprint tempo {self.tempo_bpm} BPM is outside 40-220.")
        if not self.melodic_motif.strip():
            problems.append("No melodic motif described.")
        if not self.hook_definition.strip():
            problems.append("No hook defined.")
        names = [s.name.lower() for s in self.sections]
        if names and not any("chorus" in n or "hook" in n or "refrain" in n for n in names):
            problems.append("No chorus/hook section present in the structure.")
        duplicates = {name for name in names if names.count(name) > 3}
        if duplicates:
            problems.append(f"Suspicious repeated section names: {', '.join(sorted(duplicates))}.")
        if self.sections:
            total = sum(s.bars for s in self.sections)
            if total != self.total_bars:
                self.total_bars = total
        return problems

    def summary(self) -> str:
        lines = [
            f"Key: {self.key} {self.mode} | Tempo: {self.tempo_bpm} BPM | Meter: {self.time_signature}",
            f"Groove: {self.groove} {self.swing}".strip(),
            f"Harmony: {self.chord_progression_summary} ({self.harmonic_language})",
            f"Motif: {self.melodic_motif}",
            f"Motif development: {self.motif_development}",
            f"Hook: {self.hook_definition}",
            f"Tension curve: {self.tension_curve}",
            f"Instrumentation: {', '.join(self.instrumentation) or '-'}",
            f"Format: {self.total_bars} bars (~{self.estimated_seconds:.0f}s)",
        ]
        for section in self.sections:
            chords = " | ".join(f"{c.symbol}({c.function})" for c in section.chords) or "-"
            lines.append(
                f"  [{section.name}] {section.bars} bars, energy {section.energy}/10, "
                f"chords: {chords}; drums: {section.drums}; bass: {section.bass}; "
                f"device: {section.melodic_device}; purpose: {section.purpose}"
            )
        if self.arrangement_notes:
            lines.append(f"Arrangement: {self.arrangement_notes}")
        if self.theory_rationale:
            lines.append(f"Why it works: {self.theory_rationale}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Lyricist - words, voice, persona
# --------------------------------------------------------------------------- #


@dataclass
class LyricLine:
    text: str = ""
    syllables: int = 0
    cue: str = ""  # delivery cue, e.g. "whispered", "belted"
    adlib: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str | "LyricLine") -> "LyricLine":
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return cls(text=data.strip())
        data = data or {}
        return cls(
            text=str(data.get("text", "") or "").strip(),
            syllables=as_int(data.get("syllables"), 0),
            cue=str(data.get("cue", "") or "").strip(),
            adlib=str(data.get("adlib", "") or "").strip(),
        )


@dataclass
class LyricSection:
    name: str = ""
    lines: list[LyricLine] = field(default_factory=list)
    performance_cue: str = ""
    backing_vocals: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "LyricSection") -> "LyricSection":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            name=str(data.get("name", "") or "").strip(),
            lines=[LyricLine.from_dict(line) for line in data.get("lines") or []],
            performance_cue=str(data.get("performance_cue", "") or ""),
            backing_vocals=str(data.get("backing_vocals", "") or ""),
            notes=str(data.get("notes", "") or ""),
        )

    @property
    def plain(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)


@dataclass
class VocalSpec:
    """Who sings, and how. Explicit, because Suno steers hard on this."""

    present: bool = True
    gender: str = "unspecified"  # male | female | mixed | instrumental
    voice_type: str = ""  # "warm baritone", "bright mezzo-soprano"
    register: str = ""  # chest | mixed | falsetto | head
    timbre: str = ""
    delivery: str = ""
    vibrato: str = ""
    accent: str = ""
    age_impression: str = ""
    harmony_layers: str = ""
    doubles: str = ""
    lead_style_reference: str = ""
    pronoun_persona: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "VocalSpec") -> "VocalSpec":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            present=as_bool(data.get("present"), True),
            gender=str(data.get("gender", "") or "unspecified").strip().lower(),
            voice_type=str(data.get("voice_type", "") or ""),
            register=str(data.get("register", "") or ""),
            timbre=str(data.get("timbre", "") or ""),
            delivery=str(data.get("delivery", "") or ""),
            vibrato=str(data.get("vibrato", "") or ""),
            accent=str(data.get("accent", "") or ""),
            age_impression=str(data.get("age_impression", "") or ""),
            harmony_layers=str(data.get("harmony_layers", "") or ""),
            doubles=str(data.get("doubles", "") or ""),
            lead_style_reference=str(data.get("lead_style_reference", "") or ""),
            pronoun_persona=str(data.get("pronoun_persona", "") or ""),
        )

    def describe(self) -> str:
        if not self.present:
            return "Instrumental - no lead vocals. Textural vocal chops/chants only if requested."
        parts = [
            self.gender,
            self.voice_type,
            f"{self.register} register" if self.register else "",
            self.timbre,
            self.delivery,
            self.vibrato,
            self.accent,
        ]
        text = ", ".join(part for part in parts if part)
        if self.harmony_layers:
            text += f"; harmony: {self.harmony_layers}"
        if self.doubles:
            text += f"; doubles: {self.doubles}"
        if self.age_impression:
            text += f"; sounds {self.age_impression}"
        return text or "unspecified"


@dataclass
class LyricSheet:
    """The lyricist's deliverable: sections, title options, vocal persona."""

    title_options: list[str] = field(default_factory=list)
    working_title: str = ""
    language: str = ""
    perspective: str = ""  # first person | second person | third person
    tense: str = ""
    rhyme_scheme: str = ""
    meter_notes: str = ""
    sections: list[LyricSection] = field(default_factory=list)
    vocal: VocalSpec = field(default_factory=VocalSpec)
    instrumental: bool = False
    instrumental_direction: str = ""
    hook_line: str = ""
    emotional_arc: str = ""
    imagery_palette: list[str] = field(default_factory=list)
    banned_words: list[str] = field(default_factory=list)
    syllable_budget_per_section: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = [s.to_dict() for s in self.sections]
        data["vocal"] = self.vocal.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "LyricSheet") -> "LyricSheet":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            title_options=as_str_list(data.get("title_options")),
            working_title=str(data.get("working_title", "") or ""),
            language=str(data.get("language", "") or ""),
            perspective=str(data.get("perspective", "") or ""),
            tense=str(data.get("tense", "") or ""),
            rhyme_scheme=str(data.get("rhyme_scheme", "") or ""),
            meter_notes=str(data.get("meter_notes", "") or ""),
            sections=[LyricSection.from_dict(s) for s in data.get("sections") or []],
            vocal=VocalSpec.from_dict(data.get("vocal") or {}),
            instrumental=as_bool(data.get("instrumental"), False),
            instrumental_direction=str(data.get("instrumental_direction", "") or ""),
            hook_line=str(data.get("hook_line", "") or ""),
            emotional_arc=str(data.get("emotional_arc", "") or ""),
            imagery_palette=as_str_list(data.get("imagery_palette")),
            banned_words=as_str_list(data.get("banned_words")),
            syllable_budget_per_section={
                str(k): as_int(v)
                for k, v in (data.get("syllable_budget_per_section") or {}).items()
            },
            notes=str(data.get("notes", "") or ""),
        )

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"[{section.name}]\n{section.plain}" for section in self.sections if section.plain
        )

    @property
    def line_count(self) -> int:
        return sum(len(section.lines) for section in self.sections)

    def missing_vocal_fields(self) -> list[str]:
        """Which vocal-spec fields are absent. Empty means the spec is complete.

        Single source of truth: the lyricist's own validator and the lyric
        reviewer's hard gate both call this, so a repair request and a gate failure
        can never disagree about what is missing.
        """
        if self.instrumental:
            return []
        checks = (
            ("gender", "" if self.vocal.gender in {"", "unspecified"} else self.vocal.gender),
            ("voice_type", self.vocal.voice_type),
            ("register", self.vocal.register),
            ("delivery", self.vocal.delivery),
        )
        return [name for name, value in checks if not str(value).strip()]

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.sections:
            problems.append("Lyric sheet has no sections.")
        lyric_lines = sum(1 for s in self.sections for line in s.lines if line.text.strip())
        if self.instrumental:
            if lyric_lines > 0:
                problems.append("Brief is instrumental but the sheet still contains lyric lines.")
            if not self.instrumental_direction.strip():
                problems.append("Instrumental sheet must describe the vocal texture direction.")
        elif lyric_lines < 8:
            problems.append(f"Only {lyric_lines} lyric lines - too thin for a full song.")
        if not self.instrumental and not self.hook_line.strip():
            problems.append("No hook line identified.")
        if not self.title_options:
            problems.append("No title options offered.")
        # Checked here - and not only in the gates - so the agent's cheap one-shot
        # repair can fix it before a reviewer ever sees the sheet. Found live: the
        # model omitted voice_type/register/delivery twice, the gate caught it both
        # times, and each catch cost a full generation plus a rollback.
        missing_vocal = self.missing_vocal_fields()
        if missing_vocal:
            problems.append(
                "Vocal specification is incomplete ("
                + ", ".join(missing_vocal)
                + "). Fill every field; the renderer steers on all of them."
            )
        for section in self.sections:
            if (
                section.name.lower().startswith(("verse", "chorus", "bridge"))
                and not section.lines
            ):
                problems.append(f"Section '{section.name}' is empty.")
        if not self.instrumental and self.vocal.gender == "unspecified":
            problems.append("Vocal gender is still unspecified.")
        if self.vocal.gender == "instrumental":
            self.instrumental = True
        return problems

    def as_plain_lyrics(self) -> str:
        """Suno-ready lyric body including our section-level delivery cues."""
        blocks: list[str] = []
        for section in self.sections:
            if not section.lines and not section.performance_cue:
                continue
            header = f"[{section.name}]"
            if section.performance_cue:
                header = f"[{section.name} - {section.performance_cue}]"
            body_lines: list[str] = []
            for line in section.lines:
                text = line.text.strip()
                if line.cue:
                    text = f"{text} ({line.cue})" if text else ""
                if line.adlib:
                    text = f"{text}\n({line.adlib})"
                if text:
                    body_lines.append(text)
            blocks.append("\n".join([header, *body_lines]))
        return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 4. Reviews - machine-checkable verdicts
# --------------------------------------------------------------------------- #


def parse_dimensions(value: Any) -> dict[str, float]:
    """Extract ``{dimension: score}`` without inventing data that is not there.

    A model may answer with flat keys, nested ``{"score": n}`` objects, or a list
    of ``{"name": ..., "score": ...}``. It may also omit scores entirely. The old
    behaviour of emitting ``0.0`` for every missing key was actively harmful: a
    reviewer that simply forgot to score looked like a reviewer that scored
    everything zero, which produced phantom "below the floor" issues and blocked
    otherwise acceptable work.

    Returns an empty dict when there is nothing usable, and that emptiness is
    meaningful - :func:`suno_agents.quality.verdict.finalise_review` then falls
    back to the model's ``overall_score`` instead.
    """
    if value is None:
        return {}
    items: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        items = [(str(k), v) for k, v in value.items()]
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for entry in value:
            if isinstance(entry, Mapping):
                name = entry.get("name") or entry.get("dimension") or entry.get("key")
                if name:
                    items.append((str(name), entry.get("score", entry.get("value"))))
    out: dict[str, float] = {}
    for key, raw in items:
        if isinstance(raw, Mapping):
            raw = raw.get("score", raw.get("value"))
        score = as_float(raw, float("nan"))
        if score != score:  # NaN: unparseable, so genuinely absent
            continue
        out[key] = score
    return out


#: Stage-name variants models invent, mapped onto our ids. Kept lowercase.
_STAGE_NAME_ALIASES = {
    "brief": "intake",
    "briefing": "intake",
    "conversation": "intake",
    "analyst": "audio_analyst",
    "audio": "audio_analyst",
    "audio_reference": "audio_analyst",
    "reference_analysis": "audio_analyst",
    "music": "composer",
    "composition": "composer",
    "melody": "composer",
    "lyric": "lyricist",
    "lyrics": "lyricist",
    "lyric_writer": "lyricist",
    "arrangement": "arranger",
    "prosody": "arranger",
    "prompt": "prompt_engineer",
    "suno": "prompt_engineer",
    "style": "prompt_engineer",
    "review": "",
    "none": "",
}


def normalise_stage_name(value: Any) -> str:
    """Lowercase a stage name and map common aliases onto real stage ids.

    A model answering ``"stage": "review"`` or ``"restart_from": "Lyrics"`` must not
    break routing. Unknown names are preserved so the stage graph can decide
    whether they are legal.
    """
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return ""
    if key in _STAGE_NAME_ALIASES:
        return _STAGE_NAME_ALIASES[key]
    return key


@dataclass
class ReviewIssue:
    """One actionable defect found by a reviewer."""

    code: str = ""
    severity: str = "major"  # blocker | major | minor | polish
    dimension: str = ""
    message: str = ""
    fix: str = ""
    evidence: str = ""
    restart_from: str = ""

    SEVERITY_RANK = {"blocker": 0, "major": 1, "minor": 2, "polish": 3}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str | "ReviewIssue") -> "ReviewIssue":
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return cls(message=data)
        data = data or {}
        severity = str(data.get("severity", "major") or "major").strip().lower()
        if severity not in cls.SEVERITY_RANK:
            severity = "major"
        return cls(
            code=str(data.get("code", "") or "").strip(),
            severity=severity,
            dimension=str(data.get("dimension", "") or "").strip(),
            message=str(
                data.get("message", "") or data.get("issue", "") or ""
            ).strip(),
            fix=str(data.get("fix", "") or data.get("suggestion", "") or "").strip(),
            evidence=str(data.get("evidence", "") or "").strip(),
            restart_from=str(data.get("restart_from", "") or "").strip(),
        )

    @property
    def rank(self) -> int:
        return self.SEVERITY_RANK.get(self.severity, 1)


@dataclass
class ReviewVerdict:
    """A reviewer's structured output, built on top of dimension scores."""

    stage: str = ""
    passed: bool = False
    verdict: str = "revise"  # pass | revise | reject
    overall_score: float = 0.0
    dimensions: dict[str, float] = field(default_factory=dict)
    gates_failed: list[str] = field(default_factory=list)
    issues: list[ReviewIssue] = field(default_factory=list)
    restart_from: str = ""
    reasoning: str = ""
    strengths: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [i.to_dict() for i in self.issues]
        return data

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | "ReviewVerdict", *, stage: str = ""
    ) -> "ReviewVerdict":
        if isinstance(data, cls):
            if stage and not data.stage:
                data.stage = stage
            return data
        data = data or {}
        dimensions = parse_dimensions(data.get("dimensions") or data.get("scores"))
        issues = [
            ReviewIssue.from_dict(i) for i in (data.get("issues") or data.get("findings") or [])
        ]
        return cls(
            stage=str(data.get("stage", stage) or stage),
            passed=as_bool(data.get("passed"), False),
            verdict=str(data.get("verdict", "revise") or "revise").strip().lower(),
            overall_score=as_float(data.get("overall_score"), 0.0),
            dimensions=dimensions,
            gates_failed=as_str_list(data.get("gates_failed") or data.get("failed_gates")),
            issues=issues,
            restart_from=normalise_stage_name(data.get("restart_from")),
            reasoning=str(
                data.get("reasoning")
                or data.get("summary")
                or data.get("reason")
                or data.get("notes")
                or ""
            ),
            strengths=as_str_list(data.get("strengths")),
            raw=dict(data),
        )

    @property
    def blockers(self) -> list[ReviewIssue]:
        return [i for i in self.issues if i.severity == "blocker"]

    @property
    def majors(self) -> list[ReviewIssue]:
        return [i for i in self.issues if i.severity == "major"]

    def sorted_issues(self) -> list[ReviewIssue]:
        return sorted(self.issues, key=lambda issue: issue.rank)

    def issue_brief(self, limit: int = 8) -> str:
        lines = [
            f"- [{i.severity}] {i.dimension or i.code}: {i.message}"
            + (f" -> fix: {i.fix}" if i.fix else "")
            for i in self.sorted_issues()[:limit]
        ]
        return "\n".join(lines) or "- (no issues reported)"


# --------------------------------------------------------------------------- #
# 5. Arranger - lyric meets melody
# --------------------------------------------------------------------------- #


@dataclass
class LineMapping:
    """How one lyric line sits on the grid and on the melody."""

    section: str = ""
    line: str = ""
    start_bar: int = 1
    start_beat: float = 1.0
    bars: float = 1.0
    syllables: int = 0
    syllables_per_beat: float = 0.0
    melodic_motion: str = ""
    contour: str = ""  # ascending | descending | arch | static
    emphasis: str = ""
    breath_mark: str = ""
    melisma: str = ""
    stress_conflicts: list[str] = field(default_factory=list)
    harmony_support: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "LineMapping") -> "LineMapping":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            section=str(data.get("section", "") or ""),
            line=str(data.get("line", "") or ""),
            start_bar=as_int(data.get("start_bar"), 1),
            start_beat=as_float(data.get("start_beat"), 1.0),
            bars=as_float(data.get("bars"), 1.0),
            syllables=as_int(data.get("syllables"), 0),
            syllables_per_beat=as_float(data.get("syllables_per_beat"), 0.0),
            melodic_motion=str(data.get("melodic_motion", "") or ""),
            contour=str(data.get("contour", "") or ""),
            emphasis=str(data.get("emphasis", "") or ""),
            breath_mark=str(data.get("breath_mark", "") or ""),
            melisma=str(data.get("melisma", "") or ""),
            stress_conflicts=as_str_list(data.get("stress_conflicts")),
            harmony_support=str(data.get("harmony_support", "") or ""),
        )


@dataclass
class Arrangement:
    """The arranger's deliverable: syllable-accurate lyric/nada alignment."""

    song_form: str = ""
    total_bars: int = 0
    total_seconds: float = 0.0
    vocal_entry_bar: int = 0
    hook_bar: int = 0
    mappings: list[LineMapping] = field(default_factory=list)
    section_timeline: list[dict[str, Any]] = field(default_factory=list)
    prosody_notes: str = ""
    syllable_report: str = ""
    stress_notes: str = ""
    breath_plan: str = ""
    dynamic_plan: str = ""
    transition_plan: str = ""
    hook_placement: str = ""
    instrumental_breaks: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    resolved_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mappings"] = [m.to_dict() for m in self.mappings]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "Arrangement") -> "Arrangement":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            song_form=str(data.get("song_form", "") or ""),
            total_bars=as_int(data.get("total_bars"), 0),
            total_seconds=as_float(data.get("total_seconds"), 0.0),
            vocal_entry_bar=as_int(data.get("vocal_entry_bar"), 0),
            hook_bar=as_int(data.get("hook_bar"), 0),
            mappings=[LineMapping.from_dict(m) for m in data.get("mappings") or []],
            section_timeline=[
                dict(s) for s in data.get("section_timeline") or [] if isinstance(s, Mapping)
            ],
            prosody_notes=str(data.get("prosody_notes", "") or ""),
            syllable_report=str(data.get("syllable_report", "") or ""),
            stress_notes=str(data.get("stress_notes", "") or ""),
            breath_plan=str(data.get("breath_plan", "") or ""),
            dynamic_plan=str(data.get("dynamic_plan", "") or ""),
            transition_plan=str(data.get("transition_plan", "") or ""),
            hook_placement=str(data.get("hook_placement", "") or ""),
            instrumental_breaks=as_str_list(data.get("instrumental_breaks")),
            conflicts=as_str_list(data.get("conflicts")),
            resolved_by=as_str_list(data.get("resolved_by")),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.mappings:
            problems.append("No lyric-to-note mappings produced.")
        if not self.total_bars:
            problems.append("Total bar count missing.")
        if not self.song_form.strip():
            problems.append("Song form missing.")
        conflicts = [m for m in self.mappings if m.stress_conflicts]
        if conflicts and not self.resolved_by:
            problems.append(
                f"{len(conflicts)} lines still have unresolved stress conflicts."
            )
        unmapped = [m for m in self.mappings if not m.line.strip()]
        if unmapped:
            problems.append(f"{len(unmapped)} mappings have no lyric text.")
        return problems

    def summary(self) -> str:
        lines = [
            f"Form: {self.song_form} | {self.total_bars} bars (~{self.total_seconds:.0f}s)",
            f"Vocal enters at bar {self.vocal_entry_bar}, hook lands bar {self.hook_bar}",
            f"Hook placement: {self.hook_placement}",
            f"Syllable report: {self.syllable_report}",
            f"Stress: {self.stress_notes}",
            f"Breath plan: {self.breath_plan}",
            f"Dynamics: {self.dynamic_plan}",
            f"Transitions: {self.transition_plan}",
            f"Breaks: {', '.join(self.instrumental_breaks) or '-'}",
        ]
        for mapping in self.mappings[:40]:
            lines.append(
                f"  [{mapping.section}] bar {mapping.start_bar}+{mapping.start_beat:.1f} "
                f"({mapping.bars} bars, {mapping.syllables} syll, "
                f"{mapping.syllables_per_beat:.2f} syl/beat, {mapping.contour}) "
                f'"{mapping.line}"'
                + (f" | emphasis {mapping.emphasis}" if mapping.emphasis else "")
                + (f" | melisma {mapping.melisma}" if mapping.melisma else "")
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. Final Suno prompt
# --------------------------------------------------------------------------- #


@dataclass
class SunoPrompt:
    """Exactly what the user pastes into Suno."""

    title: str = ""
    style: str = ""
    exclude_styles: list[str] = field(default_factory=list)
    lyrics: str = ""
    vocal_gender_tag: str = ""
    meta_notes: str = ""
    lossless_checklist: list[str] = field(default_factory=list)
    model_target: str = "v5"
    character_budget: dict[str, int] = field(default_factory=dict)
    provider_notes: str = ""

    @property
    def exclude_style_text(self) -> str:
        return ", ".join(self.exclude_styles)

    @property
    def style_length(self) -> int:
        return len(self.style)

    @property
    def lyrics_length(self) -> int:
        return len(self.lyrics)

    @property
    def exclude_length(self) -> int:
        return len(self.exclude_style_text)

    def to_suno_block(self) -> str:
        """Copy-paste-ready block, in the order the Suno UI asks for fields."""
        return (
            f"TITLE\n{self.title}\n\n"
            f"STYLE OF MUSIC ({self.style_length} chars)\n{self.style}\n\n"
            f"EXCLUDE STYLES ({self.exclude_length} chars)\n"
            f"{self.exclude_style_text}\n\n"
            f"LYRICS ({self.lyrics_length} chars)\n{self.lyrics}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "SunoPrompt") -> "SunoPrompt":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            title=str(data.get("title", "") or "").strip(),
            style=str(data.get("style", "") or "").strip(),
            exclude_styles=as_str_list(data.get("exclude_styles")),
            lyrics=str(data.get("lyrics", "") or "").strip(),
            vocal_gender_tag=str(data.get("vocal_gender_tag", "") or "").strip(),
            meta_notes=str(data.get("meta_notes", "") or ""),
            lossless_checklist=as_str_list(data.get("lossless_checklist")),
            model_target=str(data.get("model_target", "v5") or "v5"),
            character_budget={
                str(k): as_int(v) for k, v in (data.get("character_budget") or {}).items()
            },
            provider_notes=str(data.get("provider_notes", "") or ""),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.title.strip():
            problems.append("Prompt has no title.")
        if len(self.title) > 80:
            problems.append(f"Title is {len(self.title)} chars (Suno caps at 80).")
        if not self.style.strip():
            problems.append("Style field is empty.")
        if not self.lyrics.strip():
            problems.append("Lyrics field is empty.")
        if len(self.style) > 1000:
            problems.append(f"Style is {len(self.style)} chars (Suno caps at 1000).")
        if len(self.lyrics) > 5000:
            problems.append(f"Lyrics are {len(self.lyrics)} chars (Suno caps at 5000).")
        if len(self.exclude_style_text) > 200:
            problems.append(
                f"Exclude styles are {len(self.exclude_style_text)} chars (Suno caps at 200)."
            )
        if "[" in self.style:
            problems.append("Style field must not contain bracketed tags (they belong in lyrics).")
        if self.lyrics and "[" not in self.lyrics:
            problems.append("Lyrics field has no [Section] tags - structure will be a coin flip.")
        return problems


# --------------------------------------------------------------------------- #
# 7. Run bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class StageRecord:
    """One execution of one stage."""

    stage: str = ""
    index: int = 0
    attempt: int = 0
    started_at: str = ""
    duration_s: float = 0.0
    model: str = ""
    status: str = "done"  # done | failed | skipped
    summary: str = ""
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "StageRecord") -> "StageRecord":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            stage=str(data.get("stage", "")),
            index=as_int(data.get("index"), 0),
            attempt=as_int(data.get("attempt"), 0),
            started_at=str(data.get("started_at", "")),
            duration_s=as_float(data.get("duration_s"), 0.0),
            model=str(data.get("model", "")),
            status=str(data.get("status", "done")),
            summary=str(data.get("summary", "")),
            error=str(data.get("error", "")),
            prompt_tokens=as_int(data.get("prompt_tokens"), 0),
            completion_tokens=as_int(data.get("completion_tokens"), 0),
        )


@dataclass
class PipelineState:
    """Everything produced so far; persisted whole so a run can be inspected."""

    run_id: str = ""
    created_at: str = ""
    provider: str = ""
    brief: CreativeBrief | None = None
    reference_summary: str = ""
    reference_lossless: str = "unknown"
    reference_quality: int = 0
    blueprint: MusicBlueprint | None = None
    lyrics: LyricSheet | None = None
    lyric_review: ReviewVerdict | None = None
    arrangement: Arrangement | None = None
    music_review: ReviewVerdict | None = None
    prompt: SunoPrompt | None = None
    prompt_review: ReviewVerdict | None = None
    #: Every verdict produced during the run, in order, including the failures.
    #: The three ``*_review`` fields above hold only the final verdict per stage.
    review_log: list[ReviewVerdict] = field(default_factory=list)
    history: list[StageRecord] = field(default_factory=list)
    revisions: int = 0
    warnings: list[str] = field(default_factory=list)
    finished: bool = False
    success: bool = False
    final_report: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "provider": self.provider,
            "brief": self.brief.to_dict() if self.brief else None,
            "reference_summary": self.reference_summary,
            "reference_lossless": self.reference_lossless,
            "reference_quality": self.reference_quality,
            "blueprint": self.blueprint.to_dict() if self.blueprint else None,
            "lyrics": self.lyrics.to_dict() if self.lyrics else None,
            "lyric_review": self.lyric_review.to_dict() if self.lyric_review else None,
            "arrangement": self.arrangement.to_dict() if self.arrangement else None,
            "music_review": self.music_review.to_dict() if self.music_review else None,
            "prompt": self.prompt.to_dict() if self.prompt else None,
            "prompt_review": self.prompt_review.to_dict() if self.prompt_review else None,
            "review_log": [v.to_dict() for v in self.review_log],
            "history": [h.to_dict() for h in self.history],
            "revisions": self.revisions,
            "warnings": list(self.warnings),
            "finished": self.finished,
            "success": self.success,
            "final_report": self.final_report,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "PipelineState") -> "PipelineState":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            run_id=str(data.get("run_id", "")),
            created_at=str(data.get("created_at", "")),
            provider=str(data.get("provider", "")),
            brief=CreativeBrief.from_dict(data["brief"]) if data.get("brief") else None,
            reference_summary=str(data.get("reference_summary", "")),
            reference_lossless=str(data.get("reference_lossless", "unknown")),
            reference_quality=as_int(data.get("reference_quality"), 0),
            blueprint=(
                MusicBlueprint.from_dict(data["blueprint"]) if data.get("blueprint") else None
            ),
            lyrics=LyricSheet.from_dict(data["lyrics"]) if data.get("lyrics") else None,
            lyric_review=(
                ReviewVerdict.from_dict(data["lyric_review"])
                if data.get("lyric_review")
                else None
            ),
            arrangement=(
                Arrangement.from_dict(data["arrangement"]) if data.get("arrangement") else None
            ),
            music_review=(
                ReviewVerdict.from_dict(data["music_review"])
                if data.get("music_review")
                else None
            ),
            prompt=SunoPrompt.from_dict(data["prompt"]) if data.get("prompt") else None,
            prompt_review=(
                ReviewVerdict.from_dict(data["prompt_review"])
                if data.get("prompt_review")
                else None
            ),
            review_log=[
                ReviewVerdict.from_dict(v) for v in data.get("review_log") or []
            ],
            history=[StageRecord.from_dict(h) for h in data.get("history") or []],
            revisions=as_int(data.get("revisions"), 0),
            warnings=as_str_list(data.get("warnings")),
            finished=as_bool(data.get("finished"), False),
            success=as_bool(data.get("success"), False),
            final_report=str(data.get("final_report", "")),
        )


__all__ = [
    "NOTE_NAMES",
    "as_bool",
    "as_float",
    "as_int",
    "as_str_list",
    "clamp",
    "normalise_key",
    "normalise_stage_name",
    "parse_dimensions",
    "CreativeBrief",
    "ChordEvent",
    "SectionPlan",
    "MusicBlueprint",
    "LyricLine",
    "LyricSection",
    "VocalSpec",
    "LyricSheet",
    "ReviewIssue",
    "ReviewVerdict",
    "LineMapping",
    "Arrangement",
    "SunoPrompt",
    "StageRecord",
    "PipelineState",
]

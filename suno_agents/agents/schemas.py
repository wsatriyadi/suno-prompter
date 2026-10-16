"""JSON schemas handed to the models.

These are deliberately *descriptive* (``strict: False``) rather than enforcing:
local OpenAI-compatible servers vary wildly in how much JSON-schema they honour,
so the schemas double as documentation and the Python dataclasses in
:mod:`suno_agents.domain.models` do the real coercing and rejecting.
"""

from __future__ import annotations

from .prompts import REVIEW_SCHEMA

SECTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "bars": {"type": "integer"},
        "start_beat": {"type": "number"},
        "chords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bar": {"type": "integer"},
                    "beats": {"type": "number"},
                    "symbol": {"type": "string"},
                    "function": {"type": "string"},
                    "voicing_note": {"type": "string"},
                },
            },
        },
        "energy": {"type": "integer"},
        "dynamics": {"type": "string"},
        "texture": {"type": "string"},
        "drums": {"type": "string"},
        "bass": {"type": "string"},
        "melodic_device": {"type": "string"},
        "purpose": {"type": "string"},
        "lyrics_needed": {"type": "boolean"},
    },
}

LYRIC_LINE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "syllables": {"type": "integer"},
        "cue": {"type": "string"},
        "adlib": {"type": "string"},
    },
}

LYRIC_SECTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "lines": {"type": "array", "items": LYRIC_LINE_SCHEMA},
        "performance_cue": {"type": "string"},
        "backing_vocals": {"type": "string"},
        "notes": {"type": "string"},
    },
}

VOCAL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "gender": {"type": "string", "enum": ["male", "female", "mixed", "instrumental"]},
        "voice_type": {"type": "string"},
        "register": {"type": "string"},
        "timbre": {"type": "string"},
        "delivery": {"type": "string"},
        "vibrato": {"type": "string"},
        "accent": {"type": "string"},
        "age_impression": {"type": "string"},
        "harmony_layers": {"type": "string"},
        "doubles": {"type": "string"},
        "lead_style_reference": {"type": "string"},
        "pronoun_persona": {"type": "string"},
    },
}

BRIEF_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "working_title": {"type": "string"},
        "concept": {"type": "string"},
        "genre": {"type": "array", "items": {"type": "string"}},
        "subgenre": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "array", "items": {"type": "string"}},
        "era_reference": {"type": "string"},
        "tempo_bpm": {"type": "number"},
        "tempo_feel": {"type": "string"},
        "key": {"type": "string"},
        "time_signature": {"type": "string"},
        "language": {"type": "string"},
        "vocal_gender": {
            "type": "string",
            "enum": ["male", "female", "duet", "instrumental", "unspecified"],
        },
        "vocal_type": {"type": "string"},
        "vocal_delivery": {"type": "string"},
        "instrumental": {"type": "boolean"},
        "instrumentation": {"type": "array", "items": {"type": "string"}},
        "structure": {"type": "array", "items": {"type": "string"}},
        "reference_artists": {"type": "array", "items": {"type": "string"}},
        "reference_songs": {"type": "array", "items": {"type": "string"}},
        "theme": {"type": "string"},
        "story_arc": {"type": "string"},
        "must_include": {"type": "array", "items": {"type": "string"}},
        "must_avoid": {"type": "array", "items": {"type": "string"}},
        "delivery_target": {"type": "string"},
        "notes_for_team": {"type": "string"},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "user_expertise": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

INTAKE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "ready_to_build": {"type": "boolean"},
        "brief": BRIEF_SCHEMA,
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "suggested_choices": {"type": "array", "items": {"type": "string"}},
    },
}

REFERENCE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string"},
        "genre_read": {"type": "array", "items": {"type": "string"}},
        "production_notes": {"type": "array", "items": {"type": "string"}},
        "arrangement_lessons": {"type": "array", "items": {"type": "string"}},
        "transferable_to_new_song": {"type": "array", "items": {"type": "string"}},
        "do_not_copy": {"type": "array", "items": {"type": "string"}},
        "lossless_reality_check": {"type": "string"},
        "quality_warnings": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
}

BLUEPRINT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "mode": {"type": "string", "enum": ["major", "minor"]},
        "scale_notes": {"type": "array", "items": {"type": "string"}},
        "tempo_bpm": {"type": "number"},
        "time_signature": {"type": "string"},
        "groove": {"type": "string"},
        "swing": {"type": "string"},
        "harmonic_language": {"type": "string"},
        "chord_progression_summary": {"type": "string"},
        "sections": {"type": "array", "items": SECTION_SCHEMA},
        "total_bars": {"type": "integer"},
        "estimated_seconds": {"type": "number"},
        "melodic_motif": {"type": "string"},
        "motif_development": {"type": "string"},
        "hook_definition": {"type": "string"},
        "tension_curve": {"type": "string"},
        "instrumentation": {"type": "array", "items": {"type": "string"}},
        "arrangement_notes": {"type": "string"},
        "harmonic_risks": {"type": "array", "items": {"type": "string"}},
        "theory_rationale": {"type": "string"},
        "delivery_notes": {"type": "string"},
    },
}

LYRIC_SHEET_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title_options": {"type": "array", "items": {"type": "string"}},
        "working_title": {"type": "string"},
        "language": {"type": "string"},
        "perspective": {"type": "string"},
        "tense": {"type": "string"},
        "rhyme_scheme": {"type": "string"},
        "meter_notes": {"type": "string"},
        "sections": {"type": "array", "items": LYRIC_SECTION_SCHEMA},
        "vocal": VOCAL_SCHEMA,
        "instrumental": {"type": "boolean"},
        "instrumental_direction": {"type": "string"},
        "hook_line": {"type": "string"},
        "emotional_arc": {"type": "string"},
        "imagery_palette": {"type": "array", "items": {"type": "string"}},
        "banned_words": {"type": "array", "items": {"type": "string"}},
        "syllable_budget_per_section": {"type": "object"},
        "notes": {"type": "string"},
    },
}

ARRANGEMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "song_form": {"type": "string"},
        "total_bars": {"type": "integer"},
        "total_seconds": {"type": "number"},
        "vocal_entry_bar": {"type": "integer"},
        "hook_bar": {"type": "integer"},
        "mappings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "line": {"type": "string"},
                    "start_bar": {"type": "integer"},
                    "start_beat": {"type": "number"},
                    "bars": {"type": "number"},
                    "syllables": {"type": "integer"},
                    "syllables_per_beat": {"type": "number"},
                    "melodic_motion": {"type": "string"},
                    "contour": {"type": "string"},
                    "emphasis": {"type": "string"},
                    "breath_mark": {"type": "string"},
                    "melisma": {"type": "string"},
                    "stress_conflicts": {"type": "array", "items": {"type": "string"}},
                    "harmony_support": {"type": "string"},
                },
            },
        },
        "section_timeline": {"type": "array", "items": {"type": "object"}},
        "prosody_notes": {"type": "string"},
        "syllable_report": {"type": "string"},
        "stress_notes": {"type": "string"},
        "breath_plan": {"type": "string"},
        "dynamic_plan": {"type": "string"},
        "transition_plan": {"type": "string"},
        "hook_placement": {"type": "string"},
        "instrumental_breaks": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "string"}},
        "resolved_by": {"type": "array", "items": {"type": "string"}},
    },
}

SUNO_PROMPT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "style": {"type": "string"},
        "exclude_styles": {"type": "array", "items": {"type": "string"}},
        "lyrics": {"type": "string"},
        "vocal_gender_tag": {"type": "string"},
        "meta_notes": {"type": "string"},
        "lossless_checklist": {"type": "array", "items": {"type": "string"}},
        "model_target": {"type": "string"},
        "provider_notes": {"type": "string"},
    },
}

__all__ = [
    "REVIEW_SCHEMA",
    "SECTION_SCHEMA",
    "LYRIC_LINE_SCHEMA",
    "LYRIC_SECTION_SCHEMA",
    "VOCAL_SCHEMA",
    "BRIEF_SCHEMA",
    "INTAKE_SCHEMA",
    "REFERENCE_SCHEMA",
    "BLUEPRINT_SCHEMA",
    "LYRIC_SHEET_SCHEMA",
    "ARRANGEMENT_SCHEMA",
    "SUNO_PROMPT_SCHEMA",
]

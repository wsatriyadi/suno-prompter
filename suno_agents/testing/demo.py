"""A scripted, fully offline stand-in for the nine agents.

This exists for three reasons, all of them practical:

* ``python -m suno_agents --headless-demo`` proves the whole loop - including two
  different rollbacks - without a provider or a network call.
* The offline demo mode lets someone try the interface before deciding which
  endpoint to configure.
* It is the fixture the test suite drives the orchestrator with, so the control
  flow is tested against real artifacts rather than mocks of artifacts.

The script deliberately fails on the first pass twice: the arranger ships an
incomplete lyric-to-note map, and the prompt engineer forgets to lock the vocal
gender in the exclude field. Both failures are caught by deterministic gates, not
by the model's opinion, which is exactly the behaviour worth demonstrating.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from ..agents.lyricist import count_syllables
from ..config.models import AppSettings
from ..domain.models import CreativeBrief
from ..llm.mock import MockRunner

# --------------------------------------------------------------------------- #
# shared song material
# --------------------------------------------------------------------------- #

DEMO_BRIEF_DATA: dict[str, Any] = {
    "working_title": "Tikungan yang Salah",
    "concept": (
        "Seseorang yang masih bertahan setelah perpisahan, menunggu di tempat yang "
        "sama setiap malam, tahu jawabannya sudah jelas tapi belum siap mendengarnya."
    ),
    "genre": ["indie folk", "ballad"],
    "subgenre": ["indonesian indie"],
    "mood": ["melancholic", "warm", "intimate", "hopeful-angst"],
    "era_reference": "late 2010s Indonesian indie folk",
    "tempo_bpm": 92,
    "tempo_feel": "steady, unhurried, room to breathe",
    "key": "C",
    "time_signature": "4/4",
    "language": "id",
    "vocal_gender": "female",
    "vocal_type": "husky female alto, conversational, slightly behind the beat, roughly F3-D5",
    "vocal_delivery": "low-key and close-miked in verses, opens up on the chorus without belting",
    "instrumental": False,
    "instrumentation": [
        "fingerpicked acoustic guitar",
        "brushed drums",
        "upright bass",
        "upright piano",
        "subtle cello",
    ],
    "structure": [
        "Intro",
        "Verse 1",
        "Pre-Chorus",
        "Chorus",
        "Verse 2",
        "Chorus",
        "Bridge",
        "Chorus",
        "Outro",
    ],
    "reference_artists": [],
    "reference_songs": [],
    "theme": "menunggu tanpa kepastian",
    "story_arc": (
        "Verse 1 menggambarkan sisa-sisa kehadiran, chorus menanyakan hal yang tidak "
        "berani ditanyakan, verse 2 menunjukkan waktu yang berjalan sementara posisi "
        "tetap sama, bridge meminta jawaban langsung, chorus terakhir menerima."
    ),
    "must_include": [],
    "must_avoid": ["EDM drop", "auto-tune"],
    "delivery_target": "24-bit/48 kHz WAV master, integrated loudness -12 to -9 LUFS",
    "notes_for_team": (
        "Jangan terdengar dramatis. Ruang kosong adalah instrumen utama; chorus bekerja "
        "karena band berhenti, bukan karena bertambah keras."
    ),
    "open_questions": [],
    "user_expertise": "non-musician",
    "confidence": 0.9,
}

DEMO_LYRICS: list[dict[str, Any]] = [
    {
        "name": "Verse 1",
        "performance_cue": "close-miked, almost spoken",
        "lines": [
            "kopi dingin di meja, kursi yang kamu tinggali",
            "hujan sudah berhenti, aku masih di sini",
            "radio tua menyala, lagu yang tidak selesai",
            "aku hafal setiap jeda yang kamu tinggalkan",
        ],
    },
    {
        "name": "Pre-Chorus",
        "performance_cue": "rising",
        "lines": [
            "malam makin tipis, dinding makin dekat",
            "aku siap kalau jawabannya berat",
        ],
    },
    {
        "name": "Chorus",
        "performance_cue": "full band, still restrained",
        "lines": [
            "aku tidak minta kembali, aku cuma ingin tahu",
            "apakah kamu juga menyimpan nama ini",
            "aku tidak minta kembali, biar waktu yang menjawab",
            "kalau kamu dengar lagu ini, itu aku",
        ],
    },
    {
        "name": "Verse 2",
        "performance_cue": "drier, drums out",
        "lines": [
            "foto di dompet sudah pudar kena tangan",
            "aku simpan tiket yang tidak jadi dipakai",
            "kota ini berganti tapi jalannya sama",
            "aku masih menunggu di tikungan yang salah",
        ],
    },
    {
        "name": "Bridge",
        "performance_cue": "voice and guitar only",
        "lines": [
            "kalau memang sudah selesai, bilang saja",
            "jangan tinggalkan aku di tengah kalimat",
        ],
    },
]

DEMO_HOOK_LINE = "kalau kamu dengar lagu ini, itu aku"

DEMO_BLUEPRINT_SECTIONS: list[dict[str, Any]] = [
    {
        "name": "Intro",
        "bars": 4,
        "energy": 3,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "Am", "function": "vi", "voicing_note": "guitar, no bass"},
            {"bar": 2, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "open strings"},
        ],
        "dynamics": "quiet, room tone",
        "texture": "solo fingerpicked guitar, faint room noise",
        "drums": "none",
        "bass": "enters on the last beat",
        "melodic_device": "motif stated once, unaccompanied",
        "purpose": "state the motif and set the room",
        "lyrics_needed": False,
    },
    {
        "name": "Verse 1",
        "bars": 8,
        "energy": 5,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "C", "function": "I", "voicing_note": "no third on the first pass"},
            {"bar": 3, "beats": 4, "symbol": "G/B", "function": "V", "voicing_note": "walking bass"},
            {"bar": 5, "beats": 4, "symbol": "Am7", "function": "vi", "voicing_note": "add9 colour"},
            {"bar": 7, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "rings open"},
        ],
        "dynamics": "low, conversational",
        "texture": "guitar, brushed drums entering in bar 5, upright bass",
        "drums": "brushed snare on 2 and 4 only",
        "bass": "root on 1, fifth on the and of 3",
        "melodic_device": "motif on scale degrees 5-6-5-3, narrow range",
        "purpose": "establish the situation with concrete objects",
        "lyrics_needed": True,
    },
    {
        "name": "Pre-Chorus",
        "bars": 4,
        "energy": 6,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "sustained"},
            {"bar": 3, "beats": 4, "symbol": "G", "function": "V", "voicing_note": "no resolution, held"},
        ],
        "dynamics": "rising, no release",
        "texture": "cello joins underneath, drums drop to hats",
        "drums": "closed hats on 8ths",
        "bass": "moves to the b7 then up to the fifth",
        "melodic_device": "motif transposed up a step, harmonies in 3rds",
        "purpose": "raise tension and refuse to resolve",
        "lyrics_needed": True,
    },
    {
        "name": "Chorus",
        "bars": 8,
        "energy": 9,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "C", "function": "I", "voicing_note": "band drops for one beat then enters"},
            {"bar": 3, "beats": 4, "symbol": "G", "function": "V", "voicing_note": "open, wide"},
            {"bar": 5, "beats": 4, "symbol": "Am7", "function": "vi", "voicing_note": "piano octaves"},
            {"bar": 7, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "held into the repeat"},
        ],
        "dynamics": "full but not loud, vocal forward",
        "texture": "full band, acoustic doubles, cello counterline",
        "drums": "full kit, no cymbals in the first two bars",
        "bass": "root notes, one passing tone into the IV",
        "melodic_device": "motif inverted on the hook line, highest note of the song (D5)",
        "purpose": "state the thesis and give the listener the quotable line",
        "lyrics_needed": True,
    },
    {
        "name": "Verse 2",
        "bars": 8,
        "energy": 5,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "C", "function": "I", "voicing_note": "as verse 1"},
            {"bar": 3, "beats": 4, "symbol": "G/B", "function": "V", "voicing_note": "as verse 1"},
            {"bar": 5, "beats": 4, "symbol": "Am7", "function": "vi", "voicing_note": "add9 colour"},
            {"bar": 7, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "rings open"},
        ],
        "dynamics": "drier than verse 1, drums out",
        "texture": "guitar and vocal, bass only",
        "drums": "out",
        "bass": "sparse, half notes",
        "melodic_device": "motif extended downward, one extra bar of silence before the pre",
        "purpose": "complicate the story and show time passing",
        "lyrics_needed": True,
    },
    {
        "name": "Chorus 2",
        "bars": 8,
        "energy": 9,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "C", "function": "I", "voicing_note": "as chorus 1"},
            {"bar": 3, "beats": 4, "symbol": "G", "function": "V", "voicing_note": "as chorus 1"},
            {"bar": 5, "beats": 4, "symbol": "Am7", "function": "vi", "voicing_note": "as chorus 1"},
            {"bar": 7, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "as chorus 1"},
        ],
        "dynamics": "same as chorus 1, new harmony vocal enters",
        "texture": "full band plus harmony vocal in 3rds",
        "drums": "full kit",
        "bass": "as chorus 1",
        "melodic_device": "same hook, second half answered by the harmony",
        "purpose": "repeat the thesis with new information",
        "lyrics_needed": False,
    },
    {
        "name": "Bridge",
        "bars": 4,
        "energy": 4,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "Dm", "function": "iv", "voicing_note": "borrowed minor iv"},
            {"bar": 3, "beats": 4, "symbol": "Am", "function": "vi", "voicing_note": "no third"},
        ],
        "dynamics": "suddenly intimate",
        "texture": "voice and guitar only, close",
        "drums": "out",
        "bass": "out",
        "melodic_device": "motif fragmented, one note repeated",
        "purpose": "change the camera angle and ask directly",
        "lyrics_needed": True,
    },
    {
        "name": "Chorus 3",
        "bars": 8,
        "energy": 10,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "C", "function": "I", "voicing_note": "full band, one beat of silence before"},
            {"bar": 3, "beats": 4, "symbol": "G", "function": "V", "voicing_note": "wide"},
            {"bar": 5, "beats": 4, "symbol": "Am7", "function": "vi", "voicing_note": "piano and cello unison"},
            {"bar": 7, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "sustained, no drums under the last line"},
        ],
        "dynamics": "loudest point, then the band drops out for the final line",
        "texture": "everything, then voice alone",
        "drums": "full kit, stops on the last line",
        "bass": "root notes with a walk-up into the final bar",
        "melodic_device": "hook line changed: last phrase resolves to the root instead of hanging",
        "purpose": "pay off the question with acceptance",
        "lyrics_needed": False,
    },
    {
        "name": "Outro",
        "bars": 4,
        "energy": 3,
        "chords": [
            {"bar": 1, "beats": 4, "symbol": "Fmaj7", "function": "IV", "voicing_note": "guitar only"},
            {"bar": 3, "beats": 4, "symbol": "C", "function": "I", "voicing_note": "open, unresolved top note"},
        ],
        "dynamics": "fading to room tone",
        "texture": "guitar and a held cello note",
        "drums": "none",
        "bass": "none",
        "melodic_device": "motif sung once without words, then stops",
        "purpose": "leave the room tone as the last sound",
        "lyrics_needed": False,
    },
]

DEMO_STYLE = (
    "indie folk ballad, Indonesian, warm intimate, 92 BPM, sparse fingerpicked acoustic "
    "guitar, brushed drums, upright bass, husky female alto with conversational delivery, "
    "tape saturation, analog warmth, late-night room reverb"
)

DEMO_EXCLUDE_GOOD = [
    "male vocals",
    "male lead",
    "heavy drums",
    "EDM drop",
    "autotune",
]
DEMO_EXCLUDE_BAD = ["heavy drums", "autotune"]

DEMO_CHECKLIST = [
    "Render di Suno sebagai WAV, jangan MP3, supaya tidak ada encode kedua.",
    "Impor WAV 44.1 kHz ke DAW, naikkan ke 24-bit, dan biarkan true peak di bawah -1 dBTP.",
    "High-pass vokal di 90 Hz sebelum kompresi agar bodi gitar tidak bertumpuk.",
    "Turunkan 200-400 Hz pada gitar akustik sekitar 2 dB supaya upright bass punya ruang.",
    "Pertahankan kontras: chorus jangan dinaikkan lebih dari 6 dB dari verse.",
    "Target integrated loudness -12 sampai -9 LUFS untuk streaming, bukan -6.",
]

DEMO_LOSSLESS_NOTE = (
    "Ruang kosong adalah instrumen di sini: intro dan bridge sengaja dibiarkan "
    "sepi supaya chorus terasa lega. Jangan tambahkan lapisan di bar 1."
)

DEMO_PROMPT_META = (
    "Chorus bekerja karena band berhenti, bukan karena bertambah keras - biarkan Suno "
    "merender ruang itu. Exclude field mengunci vokal perempuan agar render tidak "
    "menukar gender penyanyi."
)

# --------------------------------------------------------------------------- #
# payload builders
# --------------------------------------------------------------------------- #


def _brief() -> CreativeBrief:
    return CreativeBrief.from_dict(DEMO_BRIEF_DATA)


def _lyric_sections_payload() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for section in DEMO_LYRICS:
        lines = [
            {
                "text": text,
                "syllables": count_syllables(text, "id"),
                "cue": "",
                "adlib": "",
            }
            for text in section["lines"]
        ]
        out.append(
            {
                "name": section["name"],
                "lines": lines,
                "performance_cue": section["performance_cue"],
                "backing_vocals": "",
                "notes": "",
            }
        )
    return out


def _lyric_payload() -> dict[str, Any]:
    return {
        "title_options": ["Tikungan yang Salah", "Kopi Dingin", "Tengah Kalimat"],
        "working_title": "Tikungan yang Salah",
        "language": "id",
        "perspective": "first person",
        "tense": "present",
        "rhyme_scheme": "ABAB with slant rhymes on 2 and 4",
        "meter_notes": "8-12 syllables per line, two bars per line at 92 BPM",
        "sections": _lyric_sections_payload(),
        "vocal": {
            "present": True,
            "gender": "female",
            "voice_type": "husky female alto, roughly F3-D5, conversational",
            "register": "chest in verses, mixed on the chorus top note",
            "timbre": "smoky with a slight rasp on long vowels",
            "delivery": "close-miked, slightly behind the beat, barely any vibrato",
            "vibrato": "narrow and slow, used only on the final held note",
            "accent": "neutral Indonesian",
            "age_impression": "late twenties",
            "harmony_layers": "3rds under the second chorus, 3rds and 6ths in the final chorus",
            "doubles": "octave-down double on the hook line only",
            "lead_style_reference": "intimate indie folk alto, not a belter",
            "pronoun_persona": "first person, addressing a second person who is absent",
        },
        "instrumental": False,
        "instrumental_direction": "",
        "hook_line": DEMO_HOOK_LINE,
        "emotional_arc": "settled grief, a question that slowly becomes acceptance",
        "imagery_palette": [
            "cold coffee",
            "worn photo",
            "old radio",
            "unused ticket",
            "the same corner",
            "room tone",
        ],
        "banned_words": ["selamanya", "takdir", "bintang"],
        "syllable_budget_per_section": {
            "Verse 1": 40,
            "Pre-Chorus": 20,
            "Chorus": 44,
            "Verse 2": 40,
            "Bridge": 20,
        },
        "notes": "Baris chorus ketiga sengaja mengulang awal baris pertama untuk memberi jeda napas.",
    }


def _blueprint_payload() -> dict[str, Any]:
    return {
        "key": "C",
        "mode": "major",
        "scale_notes": ["C", "D", "E", "F", "G", "A", "B"],
        "tempo_bpm": 92,
        "time_signature": "4/4",
        "groove": (
            "Fingerpicked guitar in 8ths with a thumb on 1 and 3; brushed snare on 2 and 4; "
            "upright bass root on 1 and fifth on the and of 3; the last bar of every section "
            "loses the drums to let the next one land."
        ),
        "swing": "straight, no swing, humanised timing on the guitar",
        "harmonic_language": "diatonic C major with a borrowed minor iv in the bridge",
        "chord_progression_summary": (
            "Verse I-V-vi-IV; Pre-Chorus IV-V held without resolution; Chorus I-V-vi-IV with "
            "the band dropping for one beat on the downbeat; Bridge iv-vi; Outro IV-I"
        ),
        "sections": DEMO_BLUEPRINT_SECTIONS,
        "total_bars": 56,
        "estimated_seconds": 146.1,
        "melodic_motif": (
            "Scale degrees 5-6-5-3 (dotted 8th, 16th, 8th, quarter) over I, stated alone in the "
            "intro and inverted on the hook line in the chorus"
        ),
        "motif_development": (
            "Verse 1 states it narrowly; the pre-chorus transposes it up a step with 3rds under; "
            "the chorus inverts it and takes it to the highest note of the song; the bridge "
            "fragments it into one repeated note; the outro sings it once without words."
        ),
        "hook_definition": (
            "Bar 5 of the chorus: the line \"kalau kamu dengar lagu ini, itu aku\" lands on the "
            "inverted motif reaching D5 while the drums drop out for one beat, then the full band "
            "returns underneath the held syllable."
        ),
        "tension_curve": (
            "Intro 3 -> Verse 5 -> Pre 6 (held, unresolved) -> Chorus 9 -> Verse 5 (drier than the "
            "first) -> Chorus 9 -> Bridge 4 (voice and guitar only) -> Chorus 10 with the band "
            "stopping under the last line -> Outro 3 fading to room tone."
        ),
        "instrumentation": [
            "fingerpicked acoustic guitar",
            "brushed drums",
            "upright bass",
            "upright piano",
            "cello counterline",
        ],
        "arrangement_notes": (
            "Nobody doubles anything in the verses. The guitar stays mono and centred; the "
            "acoustic double arrives only in the chorus, panned wide, and the cello answers the "
            "vocal in the pre-chorus. The final chorus drops the drums for one beat and then "
            "removes them entirely under the last line. No synth, no auto-tune, no EDM drop."
        ),
        "harmonic_risks": [
            "The pre-chorus V never resolves, so the chorus downbeat must feel like the resolution.",
            "The borrowed iv in the bridge can sound like a mistake if it is not voiced low.",
        ],
        "theory_rationale": (
            "A I-V-vi-IV chorus in C major is deliberately plain so the hook is carried by the "
            "lyric and the arrangement rather than by harmonic novelty. The borrowed iv in the "
            "bridge darkens the last question, and the outro's unresolved top note keeps the "
            "song from feeling resolved."
        ),
        "delivery_notes": (
            "Leave 3 dB of headroom for the chorus; keep the low end mono and let the upright "
            "bass sit under the guitar rather than beside it."
        ),
    }


def _arrangement_payload(complete: bool) -> dict[str, Any]:
    """Build the arrangement map; ``complete=False`` maps only the first line per section."""
    section_starts: dict[str, int] = {}
    cursor = 1
    for section in DEMO_BLUEPRINT_SECTIONS:
        section_starts[section["name"]] = cursor
        cursor += section["bars"]

    mappings: list[dict[str, Any]] = []
    for section in DEMO_LYRICS:
        name = section["name"]
        # The lyric sheet's "Verse 1" maps onto the blueprint's "Verse 1"; the
        # repeated chorus sections share one lyric block.
        blueprint_name = {
            "Chorus": "Chorus",
            "Pre-Chorus": "Pre-Chorus",
            "Bridge": "Bridge",
        }.get(name, name)
        start = section_starts.get(blueprint_name, 1)
        lines = section["lines"]
        for index, text in enumerate(lines):
            if not complete and index > 0:
                continue
            mappings.append(
                {
                    "section": blueprint_name,
                    "line": text,
                    "start_bar": start + index * 2,
                    "start_beat": 1,
                    "bars": 2,
                    "syllables": count_syllables(text, "id"),
                    "syllables_per_beat": 1.5,
                    "melodic_motion": "motif fragment, mostly stepwise",
                    "contour": "arch" if index % 2 == 0 else "descending",
                    "emphasis": "strong beat on the stressed syllable",
                    "breath_mark": "end of line",
                    "melisma": "none",
                    "stress_conflicts": [],
                    "harmony_support": "chord tone on the downbeat",
                }
            )
    return {
        "song_form": "Intro-V1-PC-C-V2-C-B-C-O",
        "total_bars": 56,
        "total_seconds": 146.1,
        "vocal_entry_bar": 5,
        "hook_bar": 21,
        "mappings": mappings,
        "section_timeline": [],
        "prosody_notes": (
            "Two bars per line at 92 BPM gives roughly 2 syl/beat, which leaves room for the "
            "held syllable at the end of every chorus line."
        ),
        "syllable_report": (
            "Verses sit at 8-12 syllables per line, chorus lines at 9-11, so the same melodic "
            "shape can carry every section."
        ),
        "stress_notes": (
            "Every line ends on an open vowel or a nasal (i, u, a, ai, ang), which sustains well."
        ),
        "breath_plan": (
            "A full bar of breath at the end of each couplet; the bar before the chorus is empty "
            "of words so the downbeat can be taken from silence."
        ),
        "dynamic_plan": (
            "Verse around -20 LUFS short-term, chorus around -14, bridge drops to -24, and the "
            "final line sits alone at -26."
        ),
        "transition_plan": (
            "Last bar of every section removes the drums; the pre-chorus to chorus transition "
            "adds one beat of total silence before the chorus downbeat."
        ),
        "hook_placement": (
            "The hook lands at bar 21 and again at bar 45, both times with the drums stopping for "
            "one beat and returning an octave-doubled guitar underneath."
        ),
        "instrumental_breaks": [
            "Bars 1-4: guitar only",
            "Bars 37-40: bridge, voice and guitar",
            "Bars 53-56: outro, guitar and held cello",
        ],
        "conflicts": (
            []
            if complete
            else ["Only the first line of each section has a grid position."]
        ),
        "resolved_by": []
        if complete
        else [],
    }


def _review_payload(
    stage: str,
    *,
    passed: bool,
    score: float,
    dimensions: dict[str, float],
    restart_from: str = "",
    issues: list[dict[str, Any]] | None = None,
    reasoning: str = "",
    strengths: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "verdict": "pass" if passed else "revise",
        "passed": passed,
        "overall_score": score,
        "dimensions": dimensions,
        "gates_failed": [],
        "issues": issues or [],
        "restart_from": restart_from,
        "reasoning": reasoning,
        "strengths": strengths or [],
    }


def _prompt_payload(exclude: Sequence[str]) -> dict[str, Any]:
    return {
        "title": "Tikungan yang Salah",
        "style": DEMO_STYLE,
        "exclude_styles": list(exclude),
        "lyrics": _suno_lyrics_field(),
        "vocal_gender_tag": "female",
        "meta_notes": DEMO_PROMPT_META,
        "lossless_checklist": list(DEMO_CHECKLIST),
        "model_target": "v5",
        "provider_notes": DEMO_LOSSLESS_NOTE,
    }


def _suno_lyrics_field() -> str:
    """Render the lyric sheet exactly as it should be pasted into Suno."""
    blocks = ["[Intro]"]
    blocks.append("[Verse 1]")
    blocks.append("\n".join(DEMO_LYRICS[0]["lines"]))
    blocks.append("[Pre-Chorus]")
    blocks.append("\n".join(DEMO_LYRICS[1]["lines"]))
    blocks.append("[Chorus]")
    blocks.append("\n".join(DEMO_LYRICS[2]["lines"]))
    blocks.append("[Verse 2]")
    blocks.append("\n".join(DEMO_LYRICS[3]["lines"]))
    blocks.append("[Chorus]")
    blocks.append("\n".join(DEMO_LYRICS[2]["lines"]))
    blocks.append("[Bridge]")
    blocks.append("\n".join(DEMO_LYRICS[4]["lines"]))
    blocks.append("[Chorus]")
    blocks.append("\n".join(DEMO_LYRICS[2]["lines"]))
    blocks.append("[Outro]")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# scripted responses per stage
# --------------------------------------------------------------------------- #


def _responder(payload: Any) -> Callable[[str, list[dict[str, str]]], str]:
    def respond(_system: str, _wire: list[dict[str, str]]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    return respond


def _reference_payload() -> dict[str, Any]:
    return {
        "interpretation": (
            "Tidak ada file referensi yang dikirim, jadi tidak ada angka terukur untuk "
            "ditafsirkan. Arah produksi diambil dari brief: tempo 92 BPM, C major, aransemen "
            "akustik yang lapang, vokal perempuan alto yang dekat dengan mikrofon."
        ),
        "genre_read": ["indie folk", "acoustic ballad"],
        "production_notes": [
            "Pertahankan low end mono; upright bass harus duduk di bawah gitar, bukan di sampingnya.",
            "Rekam gitar akustik dengan satu mikrofon dekat dan satu ruangan untuk lebar stereo yang jujur.",
            "Sisakan minimal 3 dB headroom untuk bagian chorus.",
        ],
        "arrangement_lessons": [
            "Chorus tidak menambah keras, tetapi menambah kepadatan: baris kedua gitar dobel dan "
            "harmoni vokal masuk di baris terakhir.",
            "Kurangi instrumen sebelum chorus agar downbeat terasa seperti pintu yang terbuka.",
            "Biarkan satu bar penuh tanpa kata sebelum chorus; napas adalah bagian dari aransemen.",
        ],
        "transferable_to_new_song": [
            "Kontras verse-chorus lewat kepadatan, bukan volume",
            "Vokal dekat, sedikit vibrato, di belakang ketukan",
        ],
        "do_not_copy": [
            "Tidak ada referensi audio, jadi tidak ada yang boleh disalin.",
        ],
        "lossless_reality_check": (
            "Belum ada file untuk diperiksa. Target akhir tetap 24-bit/48 kHz WAV; mulai selalu "
            "dari render WAV Suno, jangan dari MP3."
        ),
        "quality_warnings": [
            "Tanpa referensi, keputusan loudness harus diambil dari target streaming, bukan dari "
            "kebiasaan.",
        ],
        "confidence": 0.25,
    }


def _intake_payload() -> dict[str, Any]:
    return {
        "reply": (
            "Baik. Aku kunci arahnya: indie folk ballad Indonesia, 92 BPM, C major, vokal "
            "perempuan alto yang dekat dan sedikit serak, gitar akustik sebagai tulang, "
            "drum brushed, upright bass. Chorus dibiarkan lega karena band berhenti sejenak, "
            "bukan karena bertambah keras. Cerita: seseorang yang masih menunggu di tempat yang "
            "sama, tahu jawabannya sudah jelas tapi belum siap dengar. Kalau sudah pas, tekan "
            "\"Bangun Prompt Suno\" dan aku mulai menyusun nadanya."
        ),
        "ready_to_build": True,
        "brief": DEMO_BRIEF_DATA,
        "missing_information": [],
        "suggested_choices": [],
    }


def _stage_script(stage: str) -> list[Callable[[str, list[dict[str, str]]], str]]:
    """Ordered responses for one stage; the last one repeats forever."""
    if stage == "intake":
        return [_responder(_intake_payload())]
    if stage == "audio_analyst":
        return [_responder(_reference_payload())]
    if stage == "composer":
        return [_responder(_blueprint_payload())]
    if stage == "lyricist":
        return [_responder(_lyric_payload())]
    if stage == "lyric_reviewer":
        return [
            _responder(
                _review_payload(
                    "lyric_reviewer",
                    passed=True,
                    score=88.0,
                    dimensions={
                        "originality": 8.5,
                        "imagery": 9.0,
                        "singability": 8.5,
                        "emotional_truth": 9.0,
                        "hook_strength": 8.5,
                        "language_quality": 8.0,
                        "vocal_fit": 9.0,
                        "brief_fidelity": 9.0,
                    },
                    reasoning=(
                        "Citraan konkret dan spesifik (kopi dingin, foto di dompet, tiket yang "
                        "tidak jadi dipakai) tanpa satu pun klise. Hook bisa diulang setelah "
                        "sekali dengar dan tidak mengulang dirinya sendiri. Spesifikasi vokal "
                        "cukup spesifik untuk dipatuhi renderer."
                    ),
                    strengths=[
                        "Citraan benda konkret menggantikan pernyataan perasaan",
                        "Hook dinyatakan sebagai pertanyaan, bukan keluhan",
                        "Chorus tidak mengulang baris yang sama",
                    ],
                )
            )
        ]
    if stage == "arranger":
        return [
            _responder(_arrangement_payload(complete=False)),
            _responder(_arrangement_payload(complete=True)),
        ]
    if stage == "music_reviewer":
        return [
            _responder(
                _review_payload(
                    "music_reviewer",
                    passed=False,
                    score=74.0,
                    dimensions={
                        "structure": 8.0,
                        "harmony": 8.0,
                        "melody_contour": 7.5,
                        "groove_and_rhythm": 7.5,
                        "energy_arc": 8.5,
                        "hook_strength": 8.0,
                        "lyrics_integration": 5.0,
                        "character_and_identity": 8.0,
                        "producibility": 8.0,
                        "lossless_readiness": 8.0,
                    },
                    restart_from="arranger",
                    issues=[
                        {
                            "code": "INCOMPLETE_GRID",
                            "severity": "blocker",
                            "dimension": "lyrics_integration",
                            "message": (
                                "Hanya baris pertama tiap bagian yang punya posisi bar/beat, "
                                "sehingga separuh lirik tidak punya tempat di melodi."
                            ),
                            "fix": "Petakan setiap baris lirik ke bar, beat, dan kontur melodi.",
                            "evidence": "mappings berisi 5 baris, lirik berisi 16 baris",
                            "restart_from": "arranger",
                        }
                    ],
                    reasoning=(
                        "Struktur dan harmoni sudah kuat dan kurva energinya jelas, tetapi peta "
                        "lirik-ke-nada belum lengkap, jadi lagu ini belum bisa dinyanyikan "
                        "sampai selesai."
                    ),
                    strengths=[
                        "Kurva energi 3-5-6-9-5-9-4-10-3 dengan kontras nyata",
                        "Bridge memakai iv pinjaman untuk menggelapkan pertanyaan terakhir",
                    ],
                )
            ),
            _responder(
                _review_payload(
                    "music_reviewer",
                    passed=True,
                    score=88.5,
                    dimensions={
                        "structure": 9.0,
                        "harmony": 8.5,
                        "melody_contour": 9.0,
                        "groove_and_rhythm": 8.5,
                        "energy_arc": 9.0,
                        "hook_strength": 9.0,
                        "lyrics_integration": 8.5,
                        "character_and_identity": 9.0,
                        "producibility": 8.5,
                        "lossless_readiness": 8.5,
                    },
                    reasoning=(
                        "Setiap baris sekarang punya tempat, tekanan kata jatuh di ketukan kuat, "
                        "dan hook-nya timing-nya tepat saat drum berhenti. Karakternya jelas: "
                        "ruang kosong dipakai sebagai instrumen, bukan sebagai kekurangan."
                    ),
                    strengths=[
                        "Penempatan hook memakai keheningan sebagai penekanan",
                        "Kontras verse-chorus lewat kepadatan, bukan volume",
                        "Chorus terakhir mengubah baris penutup, bukan mengulang",
                    ],
                )
            ),
        ]
    if stage == "prompt_engineer":
        return [
            _responder(_prompt_payload(DEMO_EXCLUDE_BAD)),
            _responder(_prompt_payload(DEMO_EXCLUDE_GOOD)),
        ]
    if stage == "prompt_reviewer":
        return [
            _responder(
                _review_payload(
                    "prompt_reviewer",
                    passed=False,
                    score=76.0,
                    dimensions={
                        "prompt_fidelity": 8.5,
                        "style_clarity": 9.0,
                        "descriptor_balance": 8.5,
                        "lyrics_format": 8.5,
                        "exclusion_effectiveness": 4.0,
                        "structural_clarity": 8.5,
                        "vocal_direction": 6.0,
                        "title_hook": 8.5,
                        "render_risk": 7.0,
                        "lossless_readiness": 8.5,
                    },
                    restart_from="prompt_engineer",
                    issues=[
                        {
                            "code": "VOCAL_GENDER_UNLOCKED",
                            "severity": "blocker",
                            "dimension": "exclusion_effectiveness",
                            "message": (
                                "Exclude field tidak mengunci gender vokal, jadi Suno bisa "
                                "menukar penyanyi perempuan menjadi laki-laki."
                            ),
                            "fix": "Tambahkan 'male vocals, male lead' ke exclude_styles.",
                            "evidence": "exclude_styles = [heavy drums, autotune]",
                            "restart_from": "prompt_engineer",
                        }
                    ],
                    reasoning=(
                        "Prompt sudah menggambarkan lagu yang dibangun dengan tepat dan lirik "
                        "sudah bertag rapi, tetapi vokal tidak dikunci dan itu risiko render "
                        "yang paling mahal."
                    ),
                    strengths=[
                        "Style field menempatkan genre dan vokal di 200 karakter pertama",
                        "Tidak ada negasi di style field",
                    ],
                )
            ),
            _responder(
                _review_payload(
                    "prompt_reviewer",
                    passed=True,
                    score=90.5,
                    dimensions={
                        "prompt_fidelity": 9.0,
                        "style_clarity": 9.5,
                        "descriptor_balance": 9.0,
                        "lyrics_format": 9.0,
                        "exclusion_effectiveness": 9.0,
                        "structural_clarity": 9.0,
                        "vocal_direction": 9.0,
                        "title_hook": 9.0,
                        "render_risk": 8.5,
                        "lossless_readiness": 9.0,
                    },
                    reasoning=(
                        "Vokal sekarang terkunci, exclude field tidak menabrak apa pun yang "
                        "diminta style, lirik bertag rapi dengan baris kosong antar bagian, dan "
                        "checklist lossless-nya spesifik untuk aransemen ini."
                    ),
                    strengths=[
                        "Vokal perempuan terkunci lewat exclude field",
                        "Checklist mastering memakai angka nyata untuk lagu ini",
                        "Hook diangkat menjadi judul tanpa dipaksakan",
                    ],
                )
            ),
        ]
    return [_responder({})]


class DemoRunner:
    """Runner that answers every stage from the scripted demo material.

    ``for_stage`` is honoured by the orchestrator, which lets each stage keep its
    own response cursor - that is what makes the two scripted rollbacks work.
    """

    def __init__(self, *, trace=None) -> None:
        self.model = "demo-script"
        #: Recorder shared by every stage's mock, so offline runs produce a log that
        #: looks like a live run (minus the network).
        self.trace = trace
        self._stages: dict[str, MockRunner] = {}

    def for_stage(self, stage: str) -> MockRunner:
        runner = self._stages.get(stage)
        if runner is None:
            runner = MockRunner(
                _stage_script(stage), model=f"demo:{stage}", trace=self.trace
            )
            runner.label = f"{stage}#1"
            self._stages[stage] = runner
        return runner

    # Direct use (intake chat, single calls) falls through to a mixed script.
    def complete(self, *args, **kwargs):
        return self.for_stage("intake").complete(*args, **kwargs)

    def complete_json(self, *args, **kwargs):
        return self.for_stage("intake").complete_json(*args, **kwargs)


def build_demo_runner(trace=None) -> DemoRunner:
    """Fresh scripted runner for a full offline pipeline run.

    Pass a recorder to make the offline run show up in the agent-log tab.
    """
    return DemoRunner(trace=trace)


def demo_brief() -> CreativeBrief:
    return _brief()


def demo_settings() -> AppSettings:
    """Settings pre-filled for the offline demo."""
    settings = AppSettings()
    settings.ensure_defaults()
    settings.offline_demo = True
    return settings


__all__ = [
    "DemoRunner",
    "build_demo_runner",
    "demo_brief",
    "demo_settings",
    "DEMO_STYLE",
    "DEMO_LYRICS",
]

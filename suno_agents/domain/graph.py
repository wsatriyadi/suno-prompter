"""The nine-stage pipeline graph.

This module is the single source of truth for stage order, stage identity, and
- most importantly - *which earlier stage a reviewer is allowed to send work
back to*. The orchestrator never trusts a model's ``restart_from`` blindly: it
runs it through :func:`resolve_restart`, which clamps the request to a legal
earlier stage (or escalates one level when a stage keeps failing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

# --------------------------------------------------------------------------- #
# stage ids
# --------------------------------------------------------------------------- #

INTAKE = "intake"
AUDIO_ANALYST = "audio_analyst"
COMPOSER = "composer"
LYRICIST = "lyricist"
LYRIC_REVIEWER = "lyric_reviewer"
ARRANGER = "arranger"
MUSIC_REVIEWER = "music_reviewer"
PROMPT_ENGINEER = "prompt_engineer"
PROMPT_REVIEWER = "prompt_reviewer"

STAGE_ORDER: tuple[str, ...] = (
    INTAKE,
    AUDIO_ANALYST,
    COMPOSER,
    LYRICIST,
    LYRIC_REVIEWER,
    ARRANGER,
    MUSIC_REVIEWER,
    PROMPT_ENGINEER,
    PROMPT_REVIEWER,
)

STAGE_INDEX: dict[str, int] = {stage: i for i, stage in enumerate(STAGE_ORDER)}

REVIEW_STAGES: tuple[str, ...] = (LYRIC_REVIEWER, MUSIC_REVIEWER, PROMPT_REVIEWER)
PRODUCTION_STAGES: tuple[str, ...] = (COMPOSER, LYRICIST, ARRANGER, PROMPT_ENGINEER)

#: Legal rollback targets per reviewer. A reviewer may only send work back to
#: one of these; anything else is clamped to the closest legal ancestor.
RESTART_TARGETS: dict[str, tuple[str, ...]] = {
    LYRIC_REVIEWER: (LYRICIST, AUDIO_ANALYST, INTAKE),
    MUSIC_REVIEWER: (ARRANGER, LYRICIST, COMPOSER, AUDIO_ANALYST, INTAKE),
    PROMPT_REVIEWER: (
        PROMPT_ENGINEER,
        ARRANGER,
        LYRICIST,
        COMPOSER,
        AUDIO_ANALYST,
        INTAKE,
    ),
}

#: Free-text aliases models invent, mapped onto real stage ids.
RESTART_ALIASES: dict[str, str] = {
    "brief": INTAKE,
    "briefing": INTAKE,
    "intake": INTAKE,
    "conversation": INTAKE,
    "chat": INTAKE,
    "konsultan": INTAKE,
    "audio": AUDIO_ANALYST,
    "audio_analysis": AUDIO_ANALYST,
    "analyst": AUDIO_ANALYST,
    "reference": AUDIO_ANALYST,
    "referensi": AUDIO_ANALYST,
    "compose": COMPOSER,
    "composition": COMPOSER,
    "music": COMPOSER,
    "nada": COMPOSER,
    "melody": COMPOSER,
    "harmony": COMPOSER,
    "structure": COMPOSER,
    "lyric": LYRICIST,
    "lyrics": LYRICIST,
    "lirik": LYRICIST,
    "lyric_writer": LYRICIST,
    "vocal": LYRICIST,
    "arrange": ARRANGER,
    "arrangement": ARRANGER,
    "prosody": ARRANGER,
    "syllable": ARRANGER,
    "syllables": ARRANGER,
    "prompt": PROMPT_ENGINEER,
    "engineer": PROMPT_ENGINEER,
    "suno": PROMPT_ENGINEER,
    "style": PROMPT_ENGINEER,
    "exclude": PROMPT_ENGINEER,
    "none": "",
    "": "",
    "pass": "",
    "no": "",
}


@dataclass(frozen=True)
class StageInfo:
    """Human-facing identity of a stage, used by the UI and exported reports."""

    id: str
    title: str
    role: str
    specialty: str
    icon: str
    accent: str  # theme token name, resolved by the UI layer
    produces: str
    reviews: tuple[str, ...] = ()


STAGE_INFO: dict[str, StageInfo] = {
    INTAKE: StageInfo(
        id=INTAKE,
        title="Konsultan Musik",
        role="Agent 1 - berbicara dengan user",
        specialty=(
            "Menerjemahkan bahasa awam menjadi spesifikasi musik: genre, mood, era, "
            "tempo, tonalitas, vokal, struktur, batasan. Tidak pernah memaksa user "
            "memakai istilah teknis."
        ),
        icon="\u25cf",
        accent="c1",
        produces="CreativeBrief",
    ),
    AUDIO_ANALYST: StageInfo(
        id=AUDIO_ANALYST,
        title="Analis Referensi Audio",
        role="Agent 2 - pembaca file musik",
        specialty=(
            "Mengukur mp3/wav/m4a/flac: tempo, key, loudness, dynamic range, lebar stereo, "
            "keseimbangan band, dan mendeteksi apakah file benar-benar lossless atau hasil "
            "transcode. Menerjemahkan angka menjadi keputusan artistik."
        ),
        icon="\u223f",
        accent="c2",
        produces="ReferenceProfile + catatan produksi",
    ),
    COMPOSER: StageInfo(
        id=COMPOSER,
        title="Pencipta Nada",
        role="Agent 3 - arsitek musik",
        specialty=(
            "Merancang tonalitas, progresi akor fungsional, groove, motif melodik, "
            "pengembangan motif, titik hook, dan kurva tegangan. Berpikir sebagai "
            "komposer yang menulis untuk penyanyi, bukan generator akor acak."
        ),
        icon="\u266b",
        accent="c3",
        produces="MusicBlueprint",
    ),
    LYRICIST: StageInfo(
        id=LYRICIST,
        title="Pencipta Lirik & Vokal",
        role="Agent 4 - penulis lirik dan persona vokal",
        specialty=(
            "Menulis lirik dengan citraan konkret, suku kata yang bisa dinyanyikan, "
            "rima yang tidak dipaksakan, serta menentukan gender penyanyi, tipe suara, "
            "register, timbre, dan cara menyanyi - termasuk arah instrumental."
        ),
        icon="\u270e",
        accent="c4",
        produces="LyricSheet (lirik + vocal spec)",
    ),
    LYRIC_REVIEWER: StageInfo(
        id=LYRIC_REVIEWER,
        title="Reviewer Lirik",
        role="Agent 5 - editor lirik",
        specialty=(
            "Menilai lirik terhadap brief: orisinalitas, citraan, ekonomi kata, "
            "kelayakan nyanyi (suku kata, tekanan, vokal terbuka), kekohesifan makna, "
            "kesesuaian gender/tipe suara, dan klise. Menolak lirik yang terdengar "
            "seperti template AI."
        ),
        icon="\u2696",
        accent="c5",
        produces="ReviewVerdict",
        reviews=(LYRICIST,),
    ),
    ARRANGER: StageInfo(
        id=ARRANGER,
        title="Penyambung Lirik & Nada",
        role="Agent 6 - prosodi dan aransemen",
        specialty=(
            "Menempelkan setiap baris lirik ke grid bar/beat, menyeimbangkan jumlah suku "
            "kata dengan ruang melodi, membetulkan tabrakan aksen, merencanakan napas, "
            "dinamika, transisi, dan penempatan hook. Menghasilkan peta yang bisa "
            "dinyanyikan tanpa tersandung."
        ),
        icon="\u21c4",
        accent="c1",
        produces="Arrangement",
    ),
    MUSIC_REVIEWER: StageInfo(
        id=MUSIC_REVIEWER,
        title="Reviewer Musik",
        role="Agent 7 - kritikus struktur, harmoni, melodi, karakter",
        specialty=(
            "Mengaudit blueprint + lirik + peta aransemen sebagai satu lagu utuh: "
            "logika harmoni, kekuatan hook, kontur melodi, kurva energi, karakter dan "
            "koherensi genre, serta kelayakan produksi. Menentukan mundur ke arranger, "
            "lyricist, composer, analyst, atau intake."
        ),
        icon="\u2699",
        accent="c5",
        produces="ReviewVerdict",
        reviews=(ARRANGER, LYRICIST, COMPOSER),
    ),
    PROMPT_ENGINEER: StageInfo(
        id=PROMPT_ENGINEER,
        title="Penterjemah Prompt Suno",
        role="Agent 8 - penulis prompt final",
        specialty=(
            "Mengubah seluruh hasil kerja menjadi empat kolom Suno: judul, style, "
            "exclude styles, dan lirik bertag. Front-load deskriptor penting di 200 "
            "karakter pertama, disiplin pada batas karakter, tanpa tag di style field, "
            "dan menyertakan checklist master lossless."
        ),
        icon="\u25b6",
        accent="c3",
        produces="SunoPrompt",
    ),
    PROMPT_REVIEWER: StageInfo(
        id=PROMPT_REVIEWER,
        title="Reviewer Prompt Suno",
        role="Agent 9 - QA prompt",
        specialty=(
            "Memeriksa prompt persis seperti validator Suno: batas karakter, higienitas "
            "tag, konflik antar deskriptor, negasi yang tidak sengaja menolak hal yang "
            "diinginkan, vokal lawan jenis, dan risiko kesan 'flat/overcompressed'. "
            "Menentukan mundur sampai tahap paling awal yang benar-benar bermasalah."
        ),
        icon="\u2713",
        accent="c5",
        produces="ReviewVerdict",
        reviews=(PROMPT_ENGINEER,),
    ),
}

#: Hard limits the prompt layers must respect (Suno v4.5+/v5/v5.5).
LIMITS: dict[str, int] = {
    "style_chars": 1000,
    "lyrics_chars": 5000,
    "exclude_chars": 200,
    "title_chars": 80,
    #: Descriptors in the first ~200 chars dominate generation.
    "style_critical_zone": 200,
    "descriptors_min": 5,
    "descriptors_max": 8,
}

#: Quality floor a run must clear before the orchestrator will call it done.
QUALITY_FLOOR = 82.0


def stage_index(stage: str) -> int:
    return STAGE_INDEX.get(stage, -1)


def is_reviewer(stage: str) -> bool:
    return stage in REVIEW_STAGES


def earlier_stages(stage: str) -> tuple[str, ...]:
    """Every stage that runs strictly before ``stage``."""
    index = stage_index(stage)
    if index <= 0:
        return ()
    return STAGE_ORDER[:index]


def legal_restarts(stage: str) -> tuple[str, ...]:
    """Stages a reviewer may legally send work back to."""
    if is_reviewer(stage):
        return RESTART_TARGETS.get(stage, ())
    return earlier_stages(stage)


def normalise_restart_name(raw: str) -> str:
    """Map model-invented names onto real stage ids ('' when unknown)."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in STAGE_INDEX:
        return key
    return RESTART_ALIASES.get(key, "")


def escalate(stage: str, steps: int = 1) -> str:
    """Step one or more production stages earlier, for a stage that keeps failing.

    Reviewers are skipped: sending the arranger's work back to the lyric reviewer
    would only re-approve lyrics that already passed. Escalation therefore walks
    the production spine (intake, audio analyst, composer, lyricist, arranger,
    prompt engineer) and returns '' once there is nothing earlier left.
    """
    spine = [name for name in earlier_stages(stage) if not is_reviewer(name)]
    if not spine:
        return ""
    return spine[max(0, len(spine) - max(1, steps))]


def resolve_restart(stage: str, requested: str | Iterable[str]) -> str:
    """Clamp a reviewer's requested rollback target to something legal.

    ``requested`` may be a single name or a priority list (the model's first
    choice wins when legal). Returns '' when no rollback is available, which the
    orchestrator treats as "cannot fix - accept or abort".
    """
    options = [requested] if isinstance(requested, str) else list(requested)
    legal = legal_restarts(stage)
    for option in options:
        candidate = normalise_restart_name(str(option))
        if candidate and candidate in legal:
            return candidate
    # Unknown/greedy request: fall back to the closest legal ancestor, which is
    # the safest way to honour "reviewer decides where to restart".
    if not legal:
        return ""
    return legal[0]


def restart_depth(stage: str, target: str) -> int:
    """How many stages of work a rollback discards (0 when illegal)."""
    if target not in legal_restarts(stage):
        return 0
    return max(0, stage_index(stage) - stage_index(target))


def stages_between(start: str, end: str) -> tuple[str, ...]:
    """Inclusive range of stages from ``start`` up to ``end``."""
    a, b = stage_index(start), stage_index(end)
    if a < 0 or b < 0 or a > b:
        return ()
    return STAGE_ORDER[a : b + 1]


def describe_graph() -> str:
    """Plain-text dump of the pipeline, used in the About panel and reports."""
    lines = ["Pipeline (9 agen spesialis):"]
    for index, stage in enumerate(STAGE_ORDER, start=1):
        info = STAGE_INFO[stage]
        lines.append(f"  {index}. {info.title} [{stage}] -> {info.produces}")
        restarts = legal_restarts(stage)
        if is_reviewer(stage) and restarts:
            lines.append(f"     mundur ke: {', '.join(STAGE_INFO[r].title for r in restarts)}")
    return "\n".join(lines)


#: Metadata consumed by the settings dialog's per-stage routing tab.
STAGE_TEMPERATURE_HINTS: Mapping[str, float] = {
    INTAKE: 0.6,
    AUDIO_ANALYST: 0.4,
    COMPOSER: 0.75,
    LYRICIST: 0.9,
    LYRIC_REVIEWER: 0.2,
    ARRANGER: 0.5,
    MUSIC_REVIEWER: 0.25,
    PROMPT_ENGINEER: 0.55,
    PROMPT_REVIEWER: 0.2,
}

__all__ = [
    "INTAKE",
    "AUDIO_ANALYST",
    "COMPOSER",
    "LYRICIST",
    "LYRIC_REVIEWER",
    "ARRANGER",
    "MUSIC_REVIEWER",
    "PROMPT_ENGINEER",
    "PROMPT_REVIEWER",
    "STAGE_ORDER",
    "STAGE_INDEX",
    "REVIEW_STAGES",
    "PRODUCTION_STAGES",
    "RESTART_TARGETS",
    "StageInfo",
    "STAGE_INFO",
    "LIMITS",
    "QUALITY_FLOOR",
    "STAGE_TEMPERATURE_HINTS",
    "stage_index",
    "is_reviewer",
    "earlier_stages",
    "legal_restarts",
    "normalise_restart_name",
    "escalate",
    "resolve_restart",
    "restart_depth",
    "stages_between",
    "describe_graph",
]

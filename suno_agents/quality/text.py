"""Text forensics used by the reviewers: descriptors, cliches, negations.

The reviewers are language models, and language models are generous graders.
These helpers are the deterministic half of every review: they find the specific
defects (a bracketed tag in the style field, a forbidden instrument, a banned
cliche, a negation that cancels a requirement) so the verdict does not depend on
the model noticing them.

Everything here is word-boundary aware. Substring matching is a real bug source:
``"male" in "female"`` would flag a perfectly legal prompt as self-contradictory,
and a false blocker is worse than no check at all.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# descriptor handling (Suno style field)
# --------------------------------------------------------------------------- #

_SPLIT_RE = re.compile(r"[,;•·]|\s+-\s+")
_BRACKET_RE = re.compile(r"[\[\]{}<>]")

#: Words that are glue, not descriptors.
STOPWORDS = {
    "a", "an", "and", "the", "with", "of", "in", "on", "for", "to", "at", "by",
    "or", "as", "from", "into", "over", "under", "that", "this", "is", "are",
    "be", "it", "its", "dan", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "itu", "ini", "adalah", "serta", "agar", "biar", "juga",
}

#: Overloaded production promises that Suno cannot deliver from a text prompt.
UNSUPPORTED_PROMISES = (
    "lossless",
    "24-bit",
    "24 bit",
    "96 khz",
    "48 khz",
    "studio master",
    "mastered",
    "lufs",
    "true peak",
    "flac",
    "wav",
    "bitrate",
    "kbps",
)

#: Genre / instrument clusters used to score descriptor coverage.
GENRE_WORDS = {
    "pop", "rock", "metal", "punk", "jazz", "blues", "folk", "country", "reggae",
    "ska", "soul", "funk", "disco", "house", "techno", "trance", "edm", "dubstep",
    "dnb", "drum", "bass", "hip", "hop", "rap", "trap", "rnb", "r&b", "gospel",
    "ambient", "lofi", "lo-fi", "city", "keroncong", "dangdut", "dangdutkoplo",
    "koplo", "campursari", "gamelan", "jaipong", "sunda", "melayu", "ballad",
    "orchestral", "cinematic", "indie", "alternative", "shoegaze", "synthwave",
    "vaporwave", "afrobeat", "latin", "bossa", "samba", "kpop", "jpop", "anime",
    "hyperpop", "drill", "grime", "garage", "phonk", "kawaii",
}

EMOTION_WORDS = {
    "sad", "happy", "melancholic", "melancholy", "hopeful", "nostalgic", "bitter",
    "angry", "tender", "longing", "heartfelt", "euphoric", "dark", "bright",
    "warm", "cold", "dreamy", "gritty", "raw", "elegant", "playful", "solemn",
    "yearning", "defiant", "serene", "anxious", "sedih", "senang", "rindu",
    "marah", "tenang", "gelap", "cerah", "manis", "pahit",
}

PRODUCTION_WORDS = {
    "reverb", "delay", "tape", "analog", "digital", "lo-fi", "lofi", "polished",
    "gritty", "clean", "compressed", "sidechained", "wide", "mono", "punchy",
    "tight", "airy", "muddy", "dry", "wet", "distorted", "saturated", "vocoded",
    "filtered", "sweeping", "sidechain", "intimate",
}

INSTRUMENT_WORDS = {
    "guitar", "piano", "keys", "synth", "bass", "drum", "drums", "kick", "snare",
    "hat", "hi-hat", "hats", "strings", "cello", "violin", "viola", "brass",
    "trumpet", "sax", "saxophone", "flute", "clarinet", "organ", "rhodes",
    "wurlitzer", "moog", "pad", "arp", "arpeggio", "percussion", "shaker",
    "congas", "bongo", "tabla", "kendang", "gamelan", "sitar", "koto", "erhu",
    "harmonium", "banjo", "mandolin", "harp", "bell", "glockenspiel", "marimba",
    "vibraphone", "timpani", "choir", "vocoder", "kalimba", "accordion", "ukulele",
}

VOCAL_WORDS = {
    "male", "female", "duet", "baritone", "tenor", "bass", "alto", "mezzo",
    "soprano", "falsetto", "belt", "belting", "whisper", "whispered", "croon",
    "spoken", "chant", "harmonies", "harmony", "vocals", "vocal", "voice",
    "gang", "choir", "raspy", "breathy", "husky", "clear", "powerful", "delicate",
}

TEMPO_WORDS = {
    "bpm", "slow", "mid", "midtempo", "fast", "uptempo", "downtempo", "driving",
    "laidback", "laid-back", "half", "brisk", "languid", "steady",
}


def _token_re(needle: str) -> re.Pattern[str]:
    """Word-boundary pattern for a keyword or short phrase."""
    escaped = re.escape(needle.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def contains_keyword(text: str, needle: str) -> bool:
    """True when ``needle`` appears in ``text`` as a whole token or phrase."""
    if not needle or not str(needle).strip():
        return False
    return bool(_token_re(str(needle)).search(text or ""))


def split_descriptors(style: str) -> list[str]:
    """Split a Suno style field into its descriptor phrases."""
    if not style:
        return []
    cleaned = style.replace("\n", ", ")
    parts = [p.strip(" .\t") for p in _SPLIT_RE.split(cleaned)]
    return [p for p in parts if p]


def count_descriptors(style: str) -> int:
    return len(split_descriptors(style))


def has_brackets(text: str) -> bool:
    return bool(_BRACKET_RE.search(text or ""))


def descriptor_coverage(style: str) -> dict[str, bool]:
    """Which descriptor families are represented in the style field."""
    lowered = (style or "").lower()
    words = set(re.findall(r"[a-z0-9&\-']+", lowered))
    return {
        "genre": bool(words & GENRE_WORDS),
        "emotion": bool(words & EMOTION_WORDS),
        "production": bool(words & PRODUCTION_WORDS),
        "instrument": bool(words & INSTRUMENT_WORDS),
        "vocal": bool(words & VOCAL_WORDS),
        "tempo": bool(words & TEMPO_WORDS) or bool(re.search(r"\b\d{2,3}\s?bpm\b", lowered)),
    }


def coverage_score(style: str) -> float:
    """Fraction of descriptor families present (0-1)."""
    coverage = descriptor_coverage(style)
    return sum(1 for present in coverage.values() if present) / len(coverage)


def descriptor_families(style: str) -> list[str]:
    """Names of the descriptor families present, in canonical order."""
    coverage = descriptor_coverage(style)
    return [name for name in ("genre", "emotion", "instrument", "vocal", "tempo", "production")
            if coverage[name]]


def unsupported_promises(style: str) -> list[str]:
    """Technical mastering claims that a text prompt cannot deliver."""
    lowered = (style or "").lower()
    return [phrase for phrase in UNSUPPORTED_PROMISES if phrase in lowered]


# --------------------------------------------------------------------------- #
# negation detection
# --------------------------------------------------------------------------- #

_NEGATION_RE = re.compile(
    r"\b(no|not|without|avoid|never|non|tanpa|jangan|bukan)\b[ \-]?([a-z0-9&\-']+(?:[ \-][a-z0-9&\-']+){0,3})",
    re.IGNORECASE,
)


def find_negations(text: str) -> list[str]:
    """Extract what a piece of text is trying to negate.

    Hyphenated compounds are skipped: "no-frills" and "non-stop" are descriptors,
    not negations, and flagging them would produce false blockers.
    """
    out: list[str] = []
    source = text or ""
    for match in _NEGATION_RE.finditer(source):
        separator = source[match.end(1) : match.start(2)]
        if separator == "-":
            continue
        phrase = (match.group(2) or "").strip(" .,;")
        if phrase:
            out.append(phrase.lower())
    return out


def negation_conflicts(text: str, required: Iterable[str]) -> list[str]:
    """Required things that ``text`` accidentally negates.

    Catches the classic failure: the style field says "no drums" while the brief
    and the whole arrangement are built on a drum groove.
    """
    negations = find_negations(text)
    conflicts: list[str] = []
    for item in required:
        needle = " ".join(str(item).split())
        if not needle:
            continue
        for negation in negations:
            if contains_keyword(negation, needle) or contains_keyword(needle, negation):
                conflicts.append(
                    f"'{str(text).strip()[:70]}...' negates required '{needle}'"
                )
                break
    return conflicts


def contradicts_exclusions(style: str, exclude_styles: Sequence[str]) -> list[str]:
    """Requirements stated in the style field that the exclude field kills."""
    problems: list[str] = []
    for tag in exclude_styles:
        needle = " ".join(str(tag).split())
        if len(needle) < 3:
            continue
        if contains_keyword(style, needle):
            problems.append(
                f"Exclude list forbids '{needle}' but the style field requests it."
            )
            continue
        head = needle.split()[0]
        if (
            len(head) >= 4
            and contains_keyword(style, head)
            and head.lower() not in {"lead", "vocal", "vocals", "backing", "heavy"}
        ):
            problems.append(
                f"Exclude list forbids '{head}' but the style field requests it."
            )
    return problems


def vocal_direction_conflict(style: str, exclude_styles: Sequence[str]) -> list[str]:
    """Classic Suno failure: asking for a female vocal while excluding it."""
    problems: list[str] = []
    exclude_blob = " ".join(str(tag) for tag in exclude_styles)
    for gender in ("female", "male"):
        if contains_keyword(style, gender) and contains_keyword(exclude_blob, gender):
            problems.append(
                f"Style asks for a {gender} vocal but the exclude field also lists "
                f"'{gender}' - the render will drop or flip the vocal."
            )
    return problems


def find_cliches(text: str, extra: Iterable[str] = ()) -> list[str]:
    """Cliche phrases present in lyric text, English and Indonesian."""
    if not text:
        return []
    lowered = text.lower()
    found: list[str] = []
    for phrase in CLICHE_PHRASES:
        if phrase in lowered:
            found.append(phrase)
    for phrase in extra:
        needle = str(phrase).strip().lower()
        if needle and needle in lowered:
            found.append(needle)
    return sorted(set(found))


#: Deliberately short and high-signal: these are the tells that make a lyric read
#: as machine-written, not a general banned-word list.
CLICHE_PHRASES: tuple[str, ...] = (
    "in a world where",
    "tapestry of",
    "shadows of my soul",
    "echoes of the past",
    "dancing in the dark",
    "burning like a fire",
    "fire and ice",
    "broken glass",
    "set me free",
    "hold on tight",
    "light at the end of the tunnel",
    "heart of gold",
    "lost in the crowd",
    "midnight city",
    "neon lights",
    "fading away",
    "empty streets",
    "endless rain",
    "tears falling",
    "tears of joy",
    "hujan di malam hari",
    "air mata jatuh",
    "luka yang lama",
    "cinta sejati",
    "takdir telah menulis",
    "bintang di langit",
    "kamu dan aku selamanya",
    "hatiku hancur berkeping",
    "jiwa yang tersiksa",
    "langit kelam",
)


def repetition_ratio(lines: Sequence[str]) -> float:
    """How much of a section is verbatim repetition (1.0 = fully repeated)."""
    cleaned = [re.sub(r"[^\w\s]", "", line).strip().lower() for line in lines if line.strip()]
    if not cleaned:
        return 0.0
    unique = set(cleaned)
    return 1.0 - (len(unique) / len(cleaned))


def reading_level_penalty(text: str) -> float:
    """0-1 score for how bloated a lyric's vocabulary is (0 = tight)."""
    words = re.findall(r"[A-Za-z\u00c0-\u024f]+", text or "")
    if not words:
        return 0.0
    long_words = [w for w in words if len(w) > 9]
    return min(1.0, len(long_words) / max(8, len(words) * 0.15))


def keyword_hits(text: str, keywords: Iterable[str]) -> list[str]:
    """Which keywords appear in ``text``, matched on word boundaries."""
    return [str(k) for k in keywords if contains_keyword(text, str(k))]


def extract_section_tags(lyrics: str) -> list[str]:
    """Ordered ``[Tag]`` names found in a Suno lyrics block."""
    return [m.group(1).strip() for m in re.finditer(r"\[([^\]]+)\]", lyrics or "")]


def syllable_lines(lyrics: str) -> dict[str, list[str]]:
    """Group lyric lines by their section tag, excluding tag lines themselves."""
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in (lyrics or "").splitlines():
        line = raw_line.strip()
        tag = re.fullmatch(r"\[([^\]]+)\]", line)
        if tag:
            current = tag.group(1).strip()
            sections.setdefault(current, [])
            continue
        if line and current:
            sections[current].append(line)
    return sections


__all__ = [
    "CLICHE_PHRASES",
    "GENRE_WORDS",
    "contains_keyword",
    "count_descriptors",
    "coverage_score",
    "descriptor_coverage",
    "descriptor_families",
    "extract_section_tags",
    "find_cliches",
    "find_negations",
    "has_brackets",
    "keyword_hits",
    "negation_conflicts",
    "reading_level_penalty",
    "repetition_ratio",
    "split_descriptors",
    "contradicts_exclusions",
    "unsupported_promises",
    "vocal_direction_conflict",
    "syllable_lines",
]

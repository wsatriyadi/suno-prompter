"""Frozen data contract for the audio reference-analysis subsystem.

Only dataclasses live here: implementations (probe/decode/analysis/profile)
import these types so the rest of the app never depends on ffmpeg details.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

#: Codecs that are bit-exact lossless containers/streams.
LOSSLESS_CODECS = {
    "flac",
    "alac",
    "wavpack",
    "ape",
    "tta",
    "tak",
    "shorten",
    "aiff",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "pcm_f64le",
    "pcm_s16be",
    "pcm_s24be",
    "pcm_s32be",
    "pcm_u8",
    "pcm_s8",
    "dsd_lsbf",
    "dsd_msbf",
}

LOSSY_CODECS = {
    "mp3",
    "aac",
    "libmp3lame",
    "opus",
    "vorbis",
    "ac3",
    "eac3",
    "wmav2",
    "mp2",
    "amr_nb",
    "amr_wb",
}


@dataclass
class AudioProbe:
    """Container/stream facts straight from ffprobe."""

    path: str
    codec: str = "unknown"
    container: str = "unknown"
    sample_rate: int = 0
    channels: int = 0
    bit_rate: int | None = None
    duration: float = 0.0
    bits_per_raw_sample: int | None = None
    tags: dict[str, str] = field(default_factory=dict)
    lossless: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisMetrics:
    """Signal-level measurements for one file.

    All frequencies in Hz, all loudness/level values in dB, energies normalised
    to the loudest band/bucket where noted.
    """

    path: str = ""
    duration: float = 0.0
    analyzed_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0

    # rhythm
    tempo_bpm: float = 0.0
    tempo_confidence: float = 0.0
    beat_strength: float = 0.0
    onset_rate: float = 0.0
    swing_ratio: float = 0.0

    # tonality
    key_root: str = "C"
    key_mode: str = "major"
    key_confidence: float = 0.0
    chroma: list[float] = field(default_factory=list)  # 12 normalised values, index 0 = C

    # level / dynamics
    loudness_lufs: float = -70.0
    peak_dbfs: float = -70.0
    true_peak_dbfs: float = -70.0
    dynamic_range_db: float = 0.0
    crest_factor_db: float = 0.0
    clipped_samples: int = 0
    silence_ratio: float = 0.0

    # stereo image
    stereo_width: float = 0.0
    stereo_correlation: float = 1.0
    mid_side_balance_db: float = 0.0

    # timbre / spectral
    spectral_centroid_hz: float = 0.0
    spectral_rolloff95_hz: float = 0.0
    spectral_flatness: float = 0.0
    zero_crossing_rate: float = 0.0
    lowpass_cutoff_hz: float = 0.0
    cutoff_confidence: float = 0.0
    bands: dict[str, float] = field(default_factory=dict)
    energy_curve: list[float] = field(default_factory=list)  # 32 normalised buckets

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceProfile:
    """Aggregate verdict over every reference file the user attached."""

    probes: list[AudioProbe] = field(default_factory=list)
    metrics: AnalysisMetrics | None = None  # primary = first successfully analysed file
    per_file: list[AnalysisMetrics] = field(default_factory=list)
    lossless_verdict: str = "unknown"  # lossless | lossy | lossy-transcoded | mixed | unknown
    quality_score: int = 0  # 0-100 delivery-quality score
    warnings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    summary: str = ""  # LLM-readable text block; MUST contain every key metric
    analyzed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

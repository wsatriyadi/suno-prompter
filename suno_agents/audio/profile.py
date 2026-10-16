"""Aggregate reference files into a delivery verdict for the prompt pipeline.

The profile answers the questions the prompt writer actually needs: how good is
this reference technically, is it really lossless, and what should it not
promise in style tags.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .analysis import analyze
from .errors import AudioEngineError
from .models import AnalysisMetrics, AudioProbe, ReferenceProfile
from .probe import probe_file

__all__ = ["build_profile"]

_BAND_ORDER = ("sub", "low", "low_mid", "mid", "high_mid", "high", "air")
_BAND_LABELS = {
    "sub": "sub 20-60Hz",
    "low": "low 60-250Hz",
    "low_mid": "low-mid 250-500Hz",
    "mid": "mid 500-2000Hz",
    "high_mid": "high-mid 2-6kHz",
    "high": "high 6-12kHz",
    "air": "air 12-20kHz",
}
_MIN_TRANSPARENT_CUTOFF = 20500.0
_MAX_SUMMARY = 1800

#: (rule key, penalty, strength) — evaluated in table order.
_QUALITY_RULES = (
    ("transcoded", -45, None),
    ("lossy", -35, None),
    ("cutoff16", -20, None),
    ("cutoff12", -15, None),
    ("clipped", -10, None),
    ("dynamics", -15, "Leaves real dynamic contrast for quiet and loud sections."),
    ("hot_peak", -5, None),
    ("loud", -10, "Loudness sits in a comfortable streaming range."),
    (
        "samplerate",
        -15,
        "Sample rate is 44.1 kHz or higher, so the full audible band is available.",
    ),
)

_FALLBACK_WARNINGS = {
    "lossy": (
        "Lossy source: the codec already removed high frequencies and stereo detail, so "
        "the mix cannot be trusted as an exact tonal reference."
    ),
    "transcoded": (
        "Lossless container holding lossy audio: the file was upsampled, so the extra "
        "bandwidth above the original brickwall is empty and misleading."
    ),
    "cutoff16": (
        "High-frequency content stops below 16 kHz — style tags must not promise 'airy "
        "cymbals', 'crisp hi-hats' or 'sparkling top end'."
    ),
    "cutoff12": (
        "High-frequency content stops below 12 kHz — the reference sounds dull with no "
        "air band to copy."
    ),
    "clipped": (
        "Clipped samples (>= 0.9995 full scale) present — transients are already "
        "flattened, do not ask for more loudness."
    ),
    "dynamics": (
        "Dynamic range below 6 dB — the reference is squashed, so contrast has to come "
        "from arrangement, not level."
    ),
    "hot_peak": (
        "Peak level is above -0.1 dBFS — there is no headroom left for lossy encoding "
        "downstream."
    ),
    "loud": (
        "Integrated loudness above -8 LUFS — this is a loudness-war master; expect "
        "compressors and limiters to dominate the mix."
    ),
    "samplerate": (
        "Sample rate below 44.1 kHz — the source band limit is audible and cannot be "
        "recovered by the target model."
    ),
}


def _finite(value: object, default: float = 0.0) -> float:
    """Coerce *value* to a finite float, falling back to *default*."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt_db(value: float) -> str:
    return f"{value:.1f} dB"


# --- verdict ----------------------------------------------------------------
def _verdict(probes: Sequence[AudioProbe], metrics: Sequence[AnalysisMetrics]) -> str:
    """One of lossless | lossy | lossy-transcoded | mixed | unknown."""
    usable = [p for p in probes if p.error is None and p.codec != "unknown"]
    if not usable:
        return "unknown"
    by_path = {m.path: m for m in metrics}
    lossless_files = [p for p in usable if p.lossless]
    lossy_files = [p for p in usable if not p.lossless]

    transparent = True
    for probe in lossless_files:
        m = by_path.get(probe.path)
        if m is None:
            continue  # no spectrum evidence: trust the codec
        cutoff = _finite(m.lowpass_cutoff_hz)
        if cutoff != 0.0 and cutoff < _MIN_TRANSPARENT_CUTOFF:
            transparent = False
            break

    if lossless_files and not lossy_files:
        return "lossless" if transparent else "lossy-transcoded"
    if lossy_files and not lossless_files:
        return "lossy"
    if lossless_files and lossy_files:
        return "mixed"
    return "unknown"


# --- quality score ----------------------------------------------------------
def _aggregate(
    metrics: Sequence[AnalysisMetrics], probes: Sequence[AudioProbe]
) -> dict[str, float]:
    """Worst-case level/bandwidth facts across every analysed file.

    ``min_rate`` comes from the *probes*, not the metrics: metrics are always
    measured at the engine's analysis rate (22050 Hz by default), which says
    nothing about the file's native bandwidth.
    """
    facts = {
        "min_cutoff": math.inf,
        "max_peak": -math.inf,
        "min_dynamics": math.inf,
        "max_loudness": -math.inf,
        "min_rate": math.inf,
        "clipped": 0.0,
        "any_cutoff": 0.0,
    }
    for m in metrics:
        if m.lowpass_cutoff_hz > 0.0:
            facts["any_cutoff"] = 1.0
            facts["min_cutoff"] = min(facts["min_cutoff"], _finite(m.lowpass_cutoff_hz))
        facts["max_peak"] = max(facts["max_peak"], _finite(m.peak_dbfs, -70.0))
        facts["min_dynamics"] = min(facts["min_dynamics"], _finite(m.dynamic_range_db))
        facts["max_loudness"] = max(
            facts["max_loudness"], _finite(m.loudness_lufs, -70.0)
        )
        facts["clipped"] += float(m.clipped_samples or 0)
    for probe in probes:
        if probe.error is None and probe.sample_rate > 0:
            facts["min_rate"] = min(facts["min_rate"], float(probe.sample_rate))
    return facts


def _triggered(facts: dict[str, float], verdict: str) -> dict[str, bool]:
    """Which quality rules apply, collapsed so each issue is reported once."""
    any_cutoff = bool(facts["any_cutoff"])
    min_cutoff = facts["min_cutoff"]
    triggered = {
        "lossy": verdict == "lossy",
        "transcoded": verdict == "lossy-transcoded",
        "clipped": facts["clipped"] > 0,
        "dynamics": math.isfinite(facts["min_dynamics"]) and facts["min_dynamics"] < 6.0,
        "hot_peak": math.isfinite(facts["max_peak"]) and facts["max_peak"] > -0.1,
        "loud": math.isfinite(facts["max_loudness"]) and facts["max_loudness"] > -8.0,
        "samplerate": math.isfinite(facts["min_rate"]) and facts["min_rate"] < 44100,
        "cutoff16": any_cutoff and min_cutoff < 16000.0,
        "cutoff12": any_cutoff and min_cutoff < 12000.0,
    }
    if triggered["cutoff12"]:
        triggered["cutoff16"] = False  # one warning per issue
    if verdict in ("lossy-transcoded", "mixed"):
        # this verdict's own warning already covers the bandwidth loss
        triggered["cutoff16"] = triggered["cutoff12"] = False
    if verdict == "mixed":
        # a lossy file in the set still costs bandwidth and stereo detail
        triggered["lossy"] = True
    return triggered


def _quality(
    probes: Sequence[AudioProbe], verdict: str, metrics: Sequence[AnalysisMetrics]
) -> tuple[int, list[str], list[str]]:
    """Delivery-quality score 0-100 plus warnings and strengths."""
    if not metrics:
        return 0, [], []

    facts = _aggregate(metrics, probes)
    triggered = _triggered(facts, verdict)

    score = 100
    warnings: list[str] = []
    strengths: list[str] = []
    for key, penalty, strength in _QUALITY_RULES:
        if not triggered.get(key):
            continue
        score += penalty
        warnings.append(
            _issue_warning(probes, metrics, key, facts) or _FALLBACK_WARNINGS[key]
        )
        if strength:
            strengths.append(strength)

    if verdict == "lossless":
        strengths.append(
            "Every reference is bit-exact lossless with content up to Nyquist, so the "
            "tonal balance can be trusted as-is."
        )
    if not warnings:
        strengths.append("No technical red flags: clean levels, plenty of headroom.")
    return int(min(max(score, 0), 100)), warnings, strengths


def _offender(
    probes: Sequence[AudioProbe],
    metrics: Sequence[AnalysisMetrics],
    key: str,
    facts: dict[str, float],
) -> tuple[AudioProbe | None, AnalysisMetrics | None]:
    """The file that triggered *key*, used for concrete warning wording."""
    by_path = {p.path: p for p in probes if p.error is None}
    if key == "transcoded":
        for m in metrics:
            probe = by_path.get(m.path)
            if probe is not None and probe.lossless and m.lowpass_cutoff_hz > 0.0:
                return probe, m
    elif key == "lossy":
        for m in metrics:
            probe = by_path.get(m.path)
            if probe is not None and not probe.lossless:
                return probe, m
        for probe in probes:
            if probe.error is None and not probe.lossless:
                return probe, None
    elif key in ("cutoff16", "cutoff12"):
        worst = facts["min_cutoff"]
        best: tuple[AudioProbe | None, AnalysisMetrics | None] = (None, None)
        for m in metrics:
            cutoff = _finite(m.lowpass_cutoff_hz)
            if cutoff > 0.0 and cutoff <= worst + 1.0:
                best = (by_path.get(m.path), m)
        return best
    elif key == "samplerate":
        lowest = int(facts["min_rate"])
        for probe in probes:
            if probe.error is None and probe.sample_rate == lowest:
                return probe, (metrics[0] if metrics else None)
    return (probes[0] if probes else None), (metrics[0] if metrics else None)


def _describe(probe: AudioProbe | None, metrics: AnalysisMetrics | None) -> str:
    """``"track.mp3 (MP3, 128 kbps): "``-style prefix for a warning."""
    if probe is None:
        return ""
    detail = (probe.codec or "unknown").upper()
    if probe.bit_rate:
        detail += f", {int(probe.bit_rate / 1000)} kbps"
    if probe.sample_rate:
        detail += f", {probe.sample_rate} Hz"
    if metrics is not None and metrics.lowpass_cutoff_hz > 0.0:
        detail += f", band ends {metrics.lowpass_cutoff_hz / 1000.0:.1f} kHz"
    return f"{Path(probe.path).name} ({detail}): "


def _issue_warning(
    probes: Sequence[AudioProbe],
    metrics: Sequence[AnalysisMetrics],
    key: str,
    facts: dict[str, float],
) -> str | None:
    """Concrete, file-specific wording for a triggered quality rule."""
    probe, offending = _offender(probes, metrics, key, facts)
    prefix = _describe(probe, offending)
    if not prefix:
        return None
    edge = (
        f"{offending.lowpass_cutoff_hz / 1000.0:.1f} kHz"
        if offending is not None and offending.lowpass_cutoff_hz > 0.0
        else "the codec edge"
    )

    if key == "transcoded":
        return (
            prefix
            + f"lossless container holding upsampled lossy audio — real content stops at "
            f"{edge}, so the empty top octave above it must not be described as air or "
            "sparkle."
        )
    if key == "lossy":
        return (
            prefix
            + "lossy source — the codec already removed high frequencies and stereo "
            "detail, so absolute tonal balance and imaging are approximate."
        )
    if key == "cutoff16":
        return (
            prefix
            + f"high frequencies above ~{edge} are already destroyed — style tags must "
            "not promise 'airy cymbals', 'crisp hi-hats' or 'sparkling top end'."
        )
    if key == "cutoff12":
        return (
            prefix
            + f"content stops near {edge}; the reference has no air band at all to copy."
        )
    if key == "clipped":
        return (
            prefix
            + f"{int(facts['clipped'])} clipped samples at full scale — transients are "
            "already flattened, do not ask the target model for more loudness."
        )
    if key == "dynamics":
        return (
            prefix
            + f"dynamic range is only {facts['min_dynamics']:.1f} dB — the master is "
            "squashed, so contrast has to come from arrangement, not level."
        )
    if key == "hot_peak":
        return (
            prefix
            + f"peak level is {facts['max_peak']:.2f} dBFS, above -0.1 dBFS — there is no "
            "headroom left for lossy encoding downstream."
        )
    if key == "loud":
        return (
            prefix
            + f"integrated loudness is {facts['max_loudness']:.1f} LUFS, above -8 LUFS — "
            "expect compressors and limiters to dominate the mix."
        )
    if key == "samplerate":
        return (
            prefix
            + f"sample rate is {int(facts['min_rate'])} Hz, below 44.1 kHz — the source "
            "band limit is audible and cannot be recovered by the target model."
        )
    return None


# --- summary ----------------------------------------------------------------
def _cutoff_text(metrics: AnalysisMetrics) -> str:
    if metrics.lowpass_cutoff_hz > 0.0:
        return f"lowpass cutoff {metrics.lowpass_cutoff_hz:.0f} Hz"
    return "no lowpass cutoff detected"


def _energy_trend(curve: Sequence[float]) -> str:
    if not curve:
        return "energy curve unavailable"
    values = [_finite(v) for v in curve]
    peak = int(max(range(len(values)), key=lambda i: values[i]))
    half = len(values) // 2
    first = sum(values[:half]) / max(half, 1)
    second = sum(values[half:]) / max(len(values) - half, 1)
    delta = (second - first) * 100.0
    if delta > 15.0:
        shape = "rises to its peak"
    elif delta < -15.0:
        shape = "falls away after its peak"
    else:
        shape = "holds a steady level"
    return (
        f"energy curve 32 buckets (RMS per bucket, max = 1.00): peak at bucket {peak}, "
        f"{shape}; [{', '.join(f'{v:.2f}' for v in values)}]"
    )


def _bands_text(bands: dict[str, float]) -> str:
    if not bands:
        return "band balance unavailable"
    parts = [
        f"{_BAND_LABELS[name]} {_finite(bands[name]):+.1f} dB"
        for name in _BAND_ORDER
        if name in bands
    ]
    return "band balance (dB relative to the loudest band): " + ", ".join(parts)


def _file_line(probe: AudioProbe, metrics: AnalysisMetrics | None) -> str:
    codec = (probe.codec or "unknown").upper()
    bitrate = f"{probe.bit_rate / 1000.0:.0f} kbps" if probe.bit_rate else "bitrate n/a"
    rate = f"{probe.sample_rate} Hz" if probe.sample_rate else "rate unknown"
    channels = f"{probe.channels} ch" if probe.channels else "channels unknown"
    pieces = [
        f"{Path(probe.path).name}: {codec} in {probe.container}",
        bitrate,
        rate,
        channels,
        f"{_finite(probe.duration):.1f}s",
        "lossless" if probe.lossless else "lossy",
        _cutoff_text(metrics) if metrics is not None else "not analysed",
    ]
    return " | ".join(pieces)


def _primary_block(metrics: AnalysisMetrics) -> list[str]:
    chroma = ", ".join(f"{v:.2f}" for v in (metrics.chroma or []))
    cutoff = (
        f"{metrics.lowpass_cutoff_hz:.0f} Hz (confidence {metrics.cutoff_confidence:.2f})"
        if metrics.lowpass_cutoff_hz > 0.0
        else "none detected above 20.5 kHz (full-bandwidth content)"
    )
    return [
        f"PRIMARY ANALYSIS: {Path(metrics.path).name} — "
        f"{metrics.analyzed_seconds:.1f}s analysed at {metrics.sample_rate} Hz",
        f"- tempo: {metrics.tempo_bpm:.1f} BPM (confidence "
        f"{metrics.tempo_confidence:.2f}), beat strength {metrics.beat_strength:.2f}, "
        f"onset rate {metrics.onset_rate:.1f}/s, swing ratio {metrics.swing_ratio:.2f}",
        f"- key: {metrics.key_root} {metrics.key_mode} (confidence "
        f"{metrics.key_confidence:.2f}); chroma C..B [{chroma}]",
        f"- loudness: {metrics.loudness_lufs:.1f} LUFS integrated (BS.1770 K-weighted "
        f"approximation), peak {_fmt_db(metrics.peak_dbfs)}, true peak "
        f"{_fmt_db(metrics.true_peak_dbfs)}",
        f"- dynamics: range {_fmt_db(metrics.dynamic_range_db)} (95th-10th percentile "
        f"short-term RMS), crest factor {_fmt_db(metrics.crest_factor_db)}, clipped "
        f"samples {metrics.clipped_samples}, silence "
        f"{metrics.silence_ratio * 100.0:.1f}%",
        f"- stereo: width {metrics.stereo_width:.2f} (0 = mono, 1 = decorrelated), "
        f"correlation {metrics.stereo_correlation:+.2f}, mid/side "
        f"{_fmt_db(metrics.mid_side_balance_db)}",
        f"- spectral: centroid {metrics.spectral_centroid_hz:.0f} Hz, rolloff95 "
        f"{metrics.spectral_rolloff95_hz:.0f} Hz, flatness "
        f"{metrics.spectral_flatness:.3f}, zero-crossing rate "
        f"{metrics.zero_crossing_rate:.3f}",
        f"- codec cutoff: {cutoff}",
        f"- {_bands_text(metrics.bands)}",
        f"- {_energy_trend(metrics.energy_curve)}",
    ]


def _summary(
    probes: Sequence[AudioProbe],
    metrics: AnalysisMetrics | None,
    per_file: Sequence[AnalysisMetrics],
    verdict: str,
    quality: int,
) -> str:
    by_path = {m.path: m for m in per_file}
    lines = ["REFERENCE AUDIO PROFILE", "", "FILES"]
    for probe in probes:
        if probe.error is not None:
            lines.append(f"- {Path(probe.path).name}: unreadable ({probe.error})")
        else:
            lines.append(f"- {_file_line(probe, by_path.get(probe.path))}")
    if metrics is not None:
        lines.append("")
        lines.extend(_primary_block(metrics))
    lines += ["", f"VERDICT: {verdict}; quality score {quality}/100"]
    text = "\n".join(lines)
    if len(text) > _MAX_SUMMARY:
        text = text[: _MAX_SUMMARY - 20].rstrip() + "\n[summary truncated]"
    return text


def build_profile(paths: Sequence[str | Path]) -> ReferenceProfile:
    """Probe and analyse every reference file, then summarise the set.

    Never raises for bad input: an undecodable file lands in ``probes[i].error``
    and the aggregate verdict degrades to "unknown" with ``error`` set when
    nothing at all could be analysed.
    """
    profile = ReferenceProfile()
    if not paths:
        profile.error = "no reference files supplied"
        profile.summary = "REFERENCE AUDIO PROFILE\n\nNo reference files supplied."
        return profile

    probes: list[AudioProbe] = []
    per_file: list[AnalysisMetrics] = []
    failures: list[str] = []
    for raw in paths:
        probe = probe_file(raw)
        probes.append(probe)
        if probe.error is not None:
            failures.append(f"{Path(probe.path).name}: {probe.error}")
            continue
        try:
            metrics = analyze(probe.path)
        except AudioEngineError as exc:
            failures.append(f"{Path(probe.path).name}: {exc}")
            continue
        per_file.append(metrics)

    profile.probes = probes
    profile.per_file = per_file
    profile.metrics = per_file[0] if per_file else None
    profile.analyzed = bool(per_file)
    profile.lossless_verdict = _verdict(probes, per_file)
    score, warnings, strengths = _quality(probes, profile.lossless_verdict, per_file)
    profile.quality_score = score
    profile.warnings = warnings
    profile.strengths = strengths
    profile.summary = _summary(
        probes, profile.metrics, per_file, profile.lossless_verdict, score
    )
    if not per_file:
        detail = "; ".join(failures) if failures else "no analysable audio streams found"
        profile.error = f"nothing could be analysed: {detail}"
        profile.summary += "\n\nNOT ANALYSED: " + profile.error
    return profile

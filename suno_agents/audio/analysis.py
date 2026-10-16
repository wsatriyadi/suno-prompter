"""Signal-level reference analysis: rhythm, tonality, level, stereo, timbre.

Everything here is numpy/scipy over ffmpeg-decoded PCM. Design notes:

* Rhythm/tonality work on a mono decode at *sr* (default 22050 Hz).
* Stereo metrics use a second decode at the same rate with ``-ac 2``.
* :func:`_analyze_spectrum` runs on a short full-bandwidth decode (44.1/48 kHz,
  30 s max) because band balance and the codec brickwall live above 11 kHz,
  which a 22050 Hz analysis signal cannot see.
* Web ``ffmpeg -i x.wav -af ebur128`` output is deliberately **not** used: the
  engine must stay testable without parsing log lines.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from .decode import decode_mono, decode_stereo
from .errors import AudioEngineError
from .models import AnalysisMetrics
from .probe import probe_file

__all__ = ["analyze"]

# --- tunables ---------------------------------------------------------------
_ON_FPS = 200.0  # onset-envelope frame rate
_ONSET_N_FFT = 2048
_TIMBRE_N_FFT = 1024
_TIMBRE_HOP = 512
_SPECTRUM_N_FFT = 4096
_SPECTRUM_HOP = 2048
_SPECTRUM_SECONDS = 30.0  # full-bandwidth pass window (median spectrum + bands)
_MIN_BPM, _MAX_BPM = 60.0, 200.0
_LOUDNESS_FLOOR = -70.0
_CUTOFF_MIN_HZ = 8000.0
#: Content at or above this counts as full-bandwidth, so no cutoff is reported.
_CUTOFF_NYQUIST_MARGIN = 20500.0
#: Never report an edge closer to Nyquist than this: an encoder's own low-pass at
#: the container rate is indistinguishable from a resampling artefact there.
_CUTOFF_NYQUIST_GUARD_HZ = 600.0
#: Ceiling for the edge scan, independent of the decode rate.
_CUTOFF_SCAN_MAX_HZ = 21500.0
_CUTOFF_WINDOW_HZ = 1000.0
_CUTOFF_AFTER_HZ = 2000.0
_BRICKWALL_MIN_DROP_DB = 18.0
_FLOOR_DB = -120.0
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_BANDS = (
    ("sub", 20.0, 60.0),
    ("low", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("high_mid", 2000.0, 6000.0),
    ("high", 6000.0, 12000.0),
    ("air", 12000.0, 20000.0),
)

#: Krumhansl-Schmuckler key profiles, index 0 = the tonic.
_KS_MAJOR = (
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
)
_KS_MINOR = (
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
)


def _f(value: float, default: float = 0.0) -> float:
    """Coerce *value* to a finite float, never NaN/inf."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip(value: float, lo: float, hi: float) -> float:
    return _f(min(max(_f(value, lo), lo), hi), lo)


def _max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def _peak_dbfs(y: np.ndarray) -> float:
    peak = _max_abs(y)
    if peak <= 0.0:
        return _LOUDNESS_FLOOR
    return _clip(20.0 * math.log10(peak), _FLOOR_DB, 12.0)


def _rms_dbfs(y: np.ndarray) -> float:
    if y.size == 0:
        return _LOUDNESS_FLOOR
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if rms <= 0.0:
        return _LOUDNESS_FLOOR
    return _clip(20.0 * math.log10(rms), _FLOOR_DB, 12.0)


def _frames_view(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Zero-padded strided frame view of *y* (no copy)."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    return np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]


def _magnitudes(frames: np.ndarray, window: np.ndarray, block: int = 256):
    """Yield ``(start_index, magnitude spectra)`` blocks for a frame view."""
    n_frames = frames.shape[0]
    for start in range(0, n_frames, block):
        chunk = np.asarray(frames[start : start + block], dtype=np.float32)
        yield start, np.abs(np.fft.rfft(chunk * window, axis=1))


# --- rhythm -----------------------------------------------------------------
def _onset_envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Spectral-flux onset envelope plus its frame rate (frames/second).

    Flux is the summed positive magnitude growth across the spectrum, batched so
    the FFT never materialises every frame at once.
    """
    hop = max(1, int(round(sr / _ON_FPS)))
    frames = _frames_view(y, _ONSET_N_FFT, hop)
    window = np.hanning(_ONSET_N_FFT).astype(np.float32)

    parts: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for _, mags in _magnitudes(frames, window, block=512):
        if prev is None:
            parts.append(np.zeros(1, dtype=np.float32))
        else:
            parts.append(
                np.asarray([np.maximum(mags[0] - prev, 0.0).sum()], dtype=np.float32)
            )
        if mags.shape[0] > 1:
            parts.append(
                np.maximum(mags[1:] - mags[:-1], 0.0).sum(axis=1, dtype=np.float32)
            )
        prev = mags[-1]
    if not parts:
        return np.zeros(0, dtype=np.float32), _ON_FPS

    flux = np.concatenate(parts).astype(np.float64, copy=False)
    flux = np.maximum(flux - float(flux.mean()), 0.0)
    std = float(flux.std())
    if std > 0.0:
        flux = flux / std
    return np.asarray(flux, dtype=np.float32), float(sr) / float(hop)


def _pick_onsets(env: np.ndarray, fps: float) -> np.ndarray:
    """Adaptive-threshold peak picking on the onset envelope."""
    n = env.size
    if n < 3:
        return np.empty(0, dtype=np.int64)
    win = max(3, int(round(fps * 0.35)) | 1)
    local_mean = ndimage.uniform_filter1d(env.astype(np.float64), win, mode="nearest")
    local_var = ndimage.uniform_filter1d(
        np.square(env.astype(np.float64)), win, mode="nearest"
    ) - np.square(local_mean)
    local_std = np.sqrt(np.maximum(local_var, 0.0))
    threshold = local_mean + 1.5 * local_std
    is_peak = (env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:])
    cand = is_peak & (env[1:-1] > threshold[1:-1]) & (env[1:-1] > 0.30 * float(env.max()))
    idx = np.nonzero(cand)[0] + 1
    if idx.size > 1:
        gap = max(1, int(round(fps * 0.05)))
        keep = [int(idx[0])]
        for pos in idx[1:]:
            if pos - keep[-1] >= gap:
                keep.append(int(pos))
        idx = np.asarray(keep, dtype=np.int64)
    return idx.astype(np.int64)


def _estimate_tempo(env: np.ndarray, fps: float) -> tuple[float, float, float]:
    """Autocorrelation tempo estimate over the 60-200 BPM lag range.

    Every metrical multiple of a tempo correlates with the onset envelope too (a
    120 BPM click train is also periodic at 60 BPM), so picking the single
    strongest autocorrelation peak reports the half tempo. Each candidate lag L
    is therefore scored with its own harmonics::

        score(L) = acf[L] + 0.5 * acf[L/2] + 0.3 * acf[2L]

    Candidates whose own correlation is negligible are dropped (a score can be
    propped up entirely by its half-lag harmonic), then the *fastest* candidate
    within 10% of the best surviving score wins — the slower peak is almost
    always the half-tempo that the beat subdivisions created.

    Returns ``(bpm, confidence, period_frames)``; ``(0, 0, 0)`` when there is no
    usable periodicity. ``confidence`` is the winning peak height divided by the
    best peak more than 35% away in lag, measured *after* the octave correction,
    so genuine ambiguity still reads as low confidence rather than a silently
    wrong number.
    """
    n = env.size
    if n < 8 or fps <= 0.0:
        return 0.0, 0.0, 0.0

    centre = env.astype(np.float64)
    centre = centre - centre.mean()
    if not np.any(centre):
        return 0.0, 0.0, 0.0

    size = 1 << int(math.ceil(math.log2(max(2 * n, 2))))
    spec = np.fft.rfft(centre, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n]
    if ac[0] <= 0.0:
        return 0.0, 0.0, 0.0
    ac = ac / ac[0]

    lag_min = max(2, int(round(fps * 60.0 / _MAX_BPM)))
    lag_max = min(n - 2, int(round(fps * 60.0 / _MIN_BPM)))
    if lag_max - lag_min < 4:
        return 0.0, 0.0, 0.0

    lags = np.arange(lag_min, lag_max + 1)

    def ac_at(lag: float) -> float:
        """Autocorrelation at a fractional lag; 0.0 outside the valid range."""
        if lag < 1.0 or lag >= n:
            return 0.0
        lo = int(lag)
        hi = min(lo + 1, n - 1)
        frac = lag - lo
        return float(ac[lo] * (1.0 - frac) + ac[hi] * frac)

    raw = ac[lags]
    score = (
        raw
        + 0.5 * np.asarray([ac_at(lag / 2.0) for lag in lags])
        + 0.3 * np.asarray([ac_at(lag * 2.0) for lag in lags])
    )

    is_peak = np.ones(lags.size, dtype=bool)
    if score.size > 2:
        is_peak[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:])
    candidates = np.nonzero(is_peak)[0]
    if candidates.size == 0:
        candidates = np.asarray([int(np.argmax(score))], dtype=np.int64)

    raw_peak = float(raw[candidates].max())
    if raw_peak <= 0.0:
        return 0.0, 0.0, 0.0
    # The spec's tie-break: among the candidate lags whose *own* correlation is
    # within 10% of the best one, take the fastest. The harmonic-weighted score
    # deliberately favours the slowest octave (its half-lag term is the true beat
    # period), so it decides only which of the tie group carries the peak, not
    # which octave is reported.
    strong = candidates[raw[candidates] >= 0.9 * raw_peak - 1e-9]
    strong = strong[raw[strong] > 0.0]
    best_i = int(strong.min()) if strong.size else int(np.argmax(score))
    best_score = float(score[best_i])
    if best_score <= 0.0:
        return 0.0, 0.0, 0.0
    best_val = float(raw[best_i])
    if best_val <= 0.0:
        return 0.0, 0.0, 0.0

    delta = 0.0
    if 0 < best_i < lags.size - 1:
        y0, y1, y2 = score[best_i - 1], score[best_i], score[best_i + 1]
        denom = float(y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    lag = float(lags[best_i]) + delta
    if lag <= 0.0:
        return 0.0, 0.0, 0.0
    bpm = _clip(60.0 * fps / lag, _MIN_BPM, _MAX_BPM)

    far = candidates[np.abs(lags[candidates] - lag) > 0.35 * lag]
    if far.size:
        second = max(float(score[far].max()), 1e-9)
    else:
        second = max(float(np.median(score[candidates])), 1e-9)
    confidence = 1.0 if far.size == 0 else _clip(best_score / second, 0.0, 1.0)
    return bpm, confidence, float(60.0 * fps / bpm)


def _snap_to_peaks(
    env: np.ndarray, times: np.ndarray, period: float, fps: float
) -> np.ndarray:
    """Snap beat *times* (frame indices) onto the nearest onset-envelope peak."""
    peaks = _pick_onsets(env, fps)
    if peaks.size == 0 or times.size == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(peaks, times)
    left = np.clip(pos - 1, 0, peaks.size - 1)
    right = np.clip(pos, 0, peaks.size - 1)
    take_left = np.abs(peaks[left] - times) <= np.abs(peaks[right] - times)
    snapped = np.where(take_left, peaks[left], peaks[right])
    keep = np.abs(snapped - times) <= 0.2 * period
    return np.unique(np.asarray(snapped[keep], dtype=np.int64))


def _detect_beats(
    env: np.ndarray, fps: float, bpm: float, subdivisions: int = 1
) -> np.ndarray:
    """Beat positions (frame indices) on the best-fitting periodic grid."""
    if bpm <= 0.0 or env.size < 4 or fps <= 0.0:
        return np.empty(0, dtype=np.int64)
    period = 60.0 * fps / bpm * subdivisions
    if period < 3.0:
        return np.empty(0, dtype=np.int64)
    peaks = _pick_onsets(env, fps)
    if peaks.size < 3:
        return np.empty(0, dtype=np.int64)

    phase = np.mod(peaks.astype(np.float64) / period, 1.0)
    phases = np.linspace(0.0, 1.0, 41)[:-1]
    costs = np.abs(phase[None, :] - phases[:, None])
    costs = np.minimum(costs, 1.0 - costs)
    offset = float(phases[int(np.argmin(costs.mean(axis=1)))])

    grid = (np.arange(-1.0, env.size / period + 2.0) + offset) * period
    grid = grid[(grid >= 0.0) & (grid < env.size)]
    return _snap_to_peaks(env, grid, period, fps)


def _swing_ratio(env: np.ndarray, fps: float, bpm: float, period: float) -> float:
    """Ratio of an off-beat onset's position to the whole beat span.

    0.5 = straight eighths, 0.66 = a full triplet shuffle, 0.25 = sixteenth feel.
    *period* is in onset-envelope frames. Falls back to 0.5 when fewer than four
    usable off-beat onsets exist.
    """
    if bpm <= 0.0 or fps <= 0.0 or period < 6.0 or env.size < 4:
        return 0.5
    peaks = _pick_onsets(env, fps)
    if peaks.size < 5:
        return 0.5
    grid = np.arange(0.0, float(env.size), period)
    beats = _snap_to_peaks(env, grid, period, fps)
    if beats.size < 4:
        return 0.5
    ratios: list[float] = []
    for first, second in zip(beats[:-1], beats[1:]):
        if second - first <= 1:
            continue
        inner = peaks[(peaks > first + 1) & (peaks < second - 1)]
        if inner.size == 0:
            continue
        span = float(second - first)
        ratios.extend(float(step - first) / span for step in inner)
    if len(ratios) < 4:
        return 0.5
    return round(_clip(float(np.median(ratios)), 0.2, 0.85), 4)


def _rhythm(y: np.ndarray, sr: int, analyzed_seconds: float) -> dict[str, float]:
    env, fps = _onset_envelope(y, sr)
    if env.size < 4 or not np.any(env) or fps <= 0.0:
        return {
            "tempo_bpm": 0.0,
            "tempo_confidence": 0.0,
            "beat_strength": 0.0,
            "onset_rate": 0.0,
            "swing_ratio": 0.5,
        }

    bpm, confidence, period = _estimate_tempo(env, fps)
    onsets = _pick_onsets(env, fps)
    env_max = float(env.max())

    beat_strength = 0.0
    if env_max > 0.0 and onsets.size >= 2:
        beat_strength = _clip(float(np.mean(env[onsets])) / env_max, 0.0, 1.0)
    onset_rate = _f(onsets.size / analyzed_seconds) if analyzed_seconds > 0.0 else 0.0
    return {
        "tempo_bpm": _clip(bpm, 0.0, 300.0),
        "tempo_confidence": _clip(confidence, 0.0, 1.0),
        "beat_strength": beat_strength,
        "onset_rate": _clip(onset_rate, 0.0, 100.0),
        "swing_ratio": _swing_ratio(env, fps, bpm, period),
    }


# --- tonality ---------------------------------------------------------------
def _chroma_map(sr: int, n_fft: int, low_hz: float = 55.0, high_hz: float = 5000.0):
    """Bin index + pitch class (0 = C) for the bins inside the mapping band."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sr / 2.0 - 1.0))
    idx = np.nonzero(mask)[0]
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs[idx], 1e-9) / 440.0)
    classes = np.mod(np.rint(midi).astype(np.int64), 12)
    return idx, classes


def _spectral_stats(y: np.ndarray, sr: int) -> dict[str, object]:
    """Per-frame timbre stats (means/medians) plus the averaged chroma vector."""
    frames = _frames_view(y, _TIMBRE_N_FFT, _TIMBRE_HOP)
    window = np.hanning(_TIMBRE_N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(_TIMBRE_N_FFT, 1.0 / sr)
    chroma_idx, chroma_class = _chroma_map(sr, _TIMBRE_N_FFT)

    centroid_sum = 0.0
    flatness_sum = 0.0
    rolloffs: list[np.ndarray] = []
    chroma_acc = np.zeros(12, dtype=np.float64)
    frames_used = 0
    result: dict[str, object] = {}

    for _, mags in _magnitudes(frames, window, block=512):
        totals = mags.sum(axis=1, dtype=np.float64)
        live = totals > 0.0
        if not np.any(live):
            continue
        mags_live = mags[live].astype(np.float64, copy=False)
        totals_live = totals[live]
        count = int(mags_live.shape[0])
        frames_used += count

        centroid_sum += float(np.sum((mags_live * freqs).sum(axis=1) / totals_live))
        geo = np.exp(np.mean(np.log(mags_live + 1e-12), axis=1))
        arith = totals_live / mags_live.shape[1]
        flatness_sum += float(np.sum(geo / np.maximum(arith, 1e-12)))

        cumulative = np.cumsum(mags_live, axis=1)
        target = 0.95 * totals_live
        roll_idx = (cumulative >= target[:, None]).argmax(axis=1)
        rolloffs.append(freqs[roll_idx])
        chroma_acc += np.bincount(
            chroma_class,
            weights=mags_live[:, chroma_idx].sum(axis=0),
            minlength=12,
        )

    if frames_used > 0:
        result["spectral_centroid_hz"] = _clip(
            centroid_sum / frames_used, 0.0, sr / 2.0
        )
        result["spectral_flatness"] = _clip(flatness_sum / frames_used, 0.0, 1.0)
        result["spectral_rolloff95_hz"] = _clip(
            float(np.median(np.concatenate(rolloffs))), 0.0, sr / 2.0
        )
    else:
        result["spectral_centroid_hz"] = 0.0
        result["spectral_flatness"] = 0.0
        result["spectral_rolloff95_hz"] = 0.0

    peak = float(chroma_acc.max()) if chroma_acc.size else 0.0
    chroma = chroma_acc / peak if peak > 0.0 else np.zeros(12, dtype=np.float64)
    result["chroma"] = [round(_clip(v, 0.0, 1.0), 6) for v in chroma]
    return result


def _key_from_chroma(chroma: np.ndarray) -> tuple[str, str, float]:
    """Krumhansl-Schmuckler correlation match against 24 keys.

    ``key_confidence`` is the correlation margin between the winning and the
    runner-up key, scaled so a 0.5 correlation gap reads as full confidence.
    """
    vec = np.asarray(chroma, dtype=np.float64)
    if vec.size != 12 or not np.isfinite(vec).all() or float(vec.std()) <= 0.0:
        return "C", "major", 0.0
    scores: list[tuple[float, int, str]] = []
    for shift in range(12):
        rotated = np.roll(vec, -shift)
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            prof = np.asarray(profile, dtype=np.float64)
            denom = float(np.std(rotated) * np.std(prof))
            corr = float(np.mean((rotated - rotated.mean()) * (prof - prof.mean())))
            scores.append((corr / denom if denom > 1e-12 else 0.0, shift, mode))
    scores.sort(key=lambda item: item[0], reverse=True)
    best, second = scores[0], scores[1]
    margin = _clip((best[0] - second[0]) / 0.5, 0.0, 1.0)
    return _NOTE_NAMES[best[1]], best[2], round(margin, 4)


# --- level / dynamics -------------------------------------------------------
def _k_weight_biquads(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """ITU-R BS.1770 K-weighting pre-filter, designed for *sr*.

    The reference coefficients are specified at 48 kHz; the biquads are re-derived
    at the analysis rate from the same analogue prototypes (high-shelf
    +3.999 dB at 1681.97 Hz with the spec's shelf slope, plus a Q 0.5 high-pass
    at 38.1358 Hz). That re-derivation is the documented approximation of this
    module, and it is what makes the estimate usable at 22050 Hz.
    """
    shelf_f0, shelf_gain = 1681.974450955533, 3.999843853973347
    hp_f0, hp_q = 38.13547087602444, 0.5003270373238773

    gain = 10.0 ** (shelf_gain / 40.0)
    w0 = 2.0 * math.pi * shelf_f0 / sr
    alpha = math.sin(w0) / 2.0 * math.sqrt(2.0)
    cos_w0 = math.cos(w0)
    two_sqrt_gain_alpha = 2.0 * math.sqrt(gain) * alpha
    b_shelf = np.asarray(
        [
            gain * ((gain + 1.0) + (gain - 1.0) * cos_w0 + two_sqrt_gain_alpha),
            -2.0 * gain * ((gain - 1.0) + (gain + 1.0) * cos_w0),
            gain * ((gain + 1.0) + (gain - 1.0) * cos_w0 - two_sqrt_gain_alpha),
        ]
    )
    a_shelf = np.asarray(
        [
            (gain + 1.0) - (gain - 1.0) * cos_w0 + two_sqrt_gain_alpha,
            2.0 * ((gain - 1.0) - (gain + 1.0) * cos_w0),
            (gain + 1.0) - (gain - 1.0) * cos_w0 - two_sqrt_gain_alpha,
        ]
    )

    w0_hp = 2.0 * math.pi * hp_f0 / sr
    alpha_hp = math.sin(w0_hp) / (2.0 * hp_q)
    cos_hp = math.cos(w0_hp)
    b_hp = np.asarray([(1.0 + cos_hp) / 2.0, -(1.0 + cos_hp), (1.0 + cos_hp) / 2.0])
    a_hp = np.asarray([1.0 + alpha_hp, -2.0 * cos_hp, 1.0 - alpha_hp])

    return (
        np.asarray([b_shelf / a_shelf[0], b_hp / a_hp[0]]),
        np.asarray([a_shelf / a_shelf[0], a_hp / a_hp[0]]),
    )


def _k_weighted(y: np.ndarray, sr: int) -> np.ndarray:
    b, a = _k_weight_biquads(sr)
    out = signal.lfilter(b[1], a[1], y.astype(np.float64, copy=False))
    return signal.lfilter(b[0], a[0], out)


def _integrated_loudness_lufs(y: np.ndarray, sr: int) -> float:
    """Approximate integrated loudness (ITU-R BS.1770-4 style, gated).

    Approximation: the K-weighting biquads are re-derived at the analysis rate
    instead of the spec's 48 kHz, the signal is not pre-resampled, and the
    channel sum is a mono mix. For full-band material the error stays well under
    ~1 LU, which is enough to spot crushed or over-hot references.
    """
    if y.size == 0:
        return _LOUDNESS_FLOOR
    filtered = _k_weighted(y, sr)
    win = max(1, int(round(0.4 * sr)))
    step = max(1, int(round(0.1 * sr)))
    if filtered.size < win:
        filtered = np.pad(filtered, (0, win - filtered.size))
    starts = np.arange(0, filtered.size - win + 1, step)
    if starts.size == 0:
        starts = np.asarray([0])
    csum = np.concatenate(([0.0], np.cumsum(np.square(filtered))))
    mean_square = (csum[starts + win] - csum[starts]) / float(win)
    loud = -0.691 + 10.0 * np.log10(np.maximum(mean_square, 1e-12))

    gated = loud > _LOUDNESS_FLOOR
    if not np.any(gated):
        return _LOUDNESS_FLOOR
    relative = (
        -0.691
        + 10.0 * math.log10(max(float(mean_square[gated].mean()), 1e-12))
        - 10.0
    )
    keep = gated & (loud > relative)
    if not np.any(keep):
        keep = gated
    value = -0.691 + 10.0 * math.log10(max(float(mean_square[keep].mean()), 1e-12))
    return _clip(value, _LOUDNESS_FLOOR, 6.0)


def _short_term_rms(y: np.ndarray, sr: int, window_s: float = 0.4) -> np.ndarray:
    win = max(1, int(round(window_s * sr)))
    if y.size == 0:
        return np.zeros(0, dtype=np.float64)
    if y.size < win:
        y = np.pad(y, (0, win - y.size))
    hop = max(1, win // 2)
    frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
    return np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))


def _dynamic_range_db(y: np.ndarray, sr: int) -> float:
    """95th minus 10th percentile short-term RMS over 400 ms windows."""
    rms = _short_term_rms(y, sr)
    rms = rms[rms > 1e-7]
    if rms.size < 2:
        return 0.0
    hi, lo = float(np.percentile(rms, 95)), float(np.percentile(rms, 10))
    if hi <= 0.0 or lo <= 0.0:
        return 0.0
    return _clip(20.0 * math.log10(hi / lo), 0.0, 60.0)


def _true_peak_dbfs(y: np.ndarray, factor: int = 4, chunk: int = 4_000_000) -> float:
    """Peak of a *factor*-times oversampled signal (inter-sample peaks)."""
    if y.size == 0:
        return _LOUDNESS_FLOOR
    data = y.astype(np.float32, copy=False)
    if data.size <= chunk:
        return _peak_dbfs(np.asarray(signal.resample_poly(data, factor, 1), np.float32))
    peak = 0.0
    overlap = 4096
    step = chunk - overlap
    for start in range(0, data.size, step):
        block = data[start : start + chunk]
        if block.size == 0:
            break
        peak = max(
            peak,
            _max_abs(np.asarray(signal.resample_poly(block, factor, 1), np.float32)),
        )
        if start + chunk >= data.size:
            break
    if peak <= 0.0:
        return _LOUDNESS_FLOOR
    return _clip(20.0 * math.log10(peak), _FLOOR_DB, 12.0)


def _silence_ratio(y: np.ndarray, sr: int, floor_db: float = -60.0) -> float:
    """Fraction of 20 ms frames sitting below *floor_db*."""
    frame = max(1, int(round(0.02 * sr)))
    if y.size == 0:
        return 1.0
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::frame]
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    if rms.size == 0:
        return 1.0
    return _clip(float(np.mean(rms <= 10.0 ** (floor_db / 20.0))), 0.0, 1.0)


# --- stereo -----------------------------------------------------------------
def _stereo_metrics(stereo: np.ndarray, sr: int, mono_source: bool) -> dict[str, float]:
    """Width / correlation / mid-side balance from the stereo decode.

    ``stereo_width`` is ``1 - mean(|per-frame correlation|)``, so 0.0 means the
    two channels are identical (always the case for a mono source) and 1.0 means
    fully decorrelated.
    """
    if stereo.ndim != 2 or stereo.shape[0] < 2 or stereo.shape[1] == 0:
        return {
            "stereo_width": 0.0,
            "stereo_correlation": 1.0,
            "mid_side_balance_db": 0.0,
        }
    left = stereo[0].astype(np.float64)
    right = stereo[1].astype(np.float64)

    lc = left - left.mean()
    rc = right - right.mean()
    denom = float(np.sqrt(np.sum(np.square(lc)) * np.sum(np.square(rc))))
    corr = _clip(float(np.sum(lc * rc) / denom) if denom > 1e-12 else 1.0, -1.0, 1.0)

    frame = max(1, int(round(0.02 * sr)))
    n_full = (left.size // frame) * frame
    per_frame = np.zeros(0, dtype=np.float64)
    if n_full >= 2 * frame:
        lf = left[:n_full].reshape(-1, frame)
        rf = right[:n_full].reshape(-1, frame)
        lfc = lf - lf.mean(axis=1, keepdims=True)
        rfc = rf - rf.mean(axis=1, keepdims=True)
        num = np.sum(lfc * rfc, axis=1)
        den = np.sqrt(np.sum(np.square(lfc), axis=1) * np.sum(np.square(rfc), axis=1))
        usable = den > 1e-9
        if np.any(usable):
            per_frame = np.clip(num[usable] / den[usable], -1.0, 1.0)

    width = 0.0 if (mono_source or per_frame.size == 0) else _clip(
        1.0 - float(np.mean(np.abs(per_frame))), 0.0, 1.0
    )

    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    mid_energy = float(np.sum(np.square(mid)))
    side_energy = float(np.sum(np.square(side)))
    if mid_energy <= 1e-12 and side_energy <= 1e-12:
        balance = 0.0
    else:
        balance = 10.0 * math.log10(max(side_energy, 1e-12) / max(mid_energy, 1e-12))
    return {
        "stereo_width": round(width, 6),
        "stereo_correlation": round(corr, 6),
        "mid_side_balance_db": round(_clip(balance, -60.0, 20.0), 3),
    }


# --- full-bandwidth spectrum: bands + codec cutoff --------------------------
def _band_energies(y: np.ndarray, sr: int) -> dict[str, float]:
    """Mean band energy in dB relative to the loudest band (loudest = 0.0 dB)."""
    if y.size < 256:
        return {name: 0.0 for name, _, _ in _BANDS}
    nperseg = min(4096, y.size)
    freqs, power = signal.welch(
        y.astype(np.float64, copy=False),
        fs=float(sr),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
    )
    energies: dict[str, float] = {}
    for name, lo, hi in _BANDS:
        mask = (freqs >= lo) & (freqs <= min(hi, sr / 2.0))
        if mask.sum() >= 2:
            energies[name] = float(np.trapezoid(power[mask], freqs[mask]))
        elif mask.sum() == 1:
            energies[name] = float(power[mask][0])
        else:
            energies[name] = 0.0
    peak = max(energies.values()) if energies else 0.0
    if peak <= 0.0:
        return {name: 0.0 for name in energies}
    return {
        name: round(_clip(10.0 * math.log10(max(value, 1e-12) / peak), _FLOOR_DB, 0.0), 3)
        for name, value in energies.items()
    }


def _median_spectrum(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-median magnitude spectrum over the full-bandwidth window."""
    frames = _frames_view(y, _SPECTRUM_N_FFT, _SPECTRUM_HOP)
    window = np.hanning(_SPECTRUM_N_FFT).astype(np.float32)
    n_bins = _SPECTRUM_N_FFT // 2 + 1
    freqs = np.fft.rfftfreq(_SPECTRUM_N_FFT, 1.0 / sr)
    if frames.shape[0] == 0:
        return np.zeros(n_bins, dtype=np.float64), freqs
    accum = np.empty((frames.shape[0], n_bins), dtype=np.float32)
    for start, mags in _magnitudes(frames, window, block=128):
        accum[start : start + mags.shape[0]] = mags
    return np.median(accum.astype(np.float64), axis=0), freqs


def _lowpass_cutoff(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Detect a codec brickwall from the time-median magnitude spectrum.

    The comparison reference is the *loudest* bin above 8 kHz, not the global
    spectral peak: real music is bass-dominant, so its 10-20 kHz content often
    sits more than 30 dB below the low-frequency peak and a global reference
    would never find a cutoff.

    ``(0.0, 0.0)`` means "no cutoff detected": the strongest collapse anywhere in
    the band failed to reach the brickwall threshold, which is what genuinely
    full-bandwidth material looks like. There is deliberately no separate
    "content exists above 20.5 kHz" early exit: lossy encoders leave decodable
    residue up there, so that test would mask the very bricks it is meant to find.
    Keeping one mechanism (strongest edge) also means the result cannot depend on
    which sample rate the file happened to be decoded at.

    The edge is found by scanning every candidate frequency and keeping the
    *strongest* collapse (>= 18 dB across the following 2 kHz), not merely the
    drop after the highest live bin. That distinction is the whole ballgame: a
    lossy encoder leaves low-level quantisation noise above its cutoff, and when
    the file is decoded at a rate above the encoder's those bins still count as
    "live", so testing only the topmost bin slides the reported cutoff up to the
    new Nyquist and produces the false negative that matters most here - calling
    an upsample lossless. ``cutoff_confidence`` scales with the drop.
    """
    median, freqs = _median_spectrum(y, sr)
    if median.size == 0 or not np.any(median > 0.0):
        return 0.0, 0.0
    band = freqs >= _CUTOFF_MIN_HZ
    if not np.any(band):
        return 0.0, 0.0
    reference = float(np.max(median[band]))
    if reference <= 0.0:
        return 0.0, 0.0
    live = median >= reference * 10.0 ** (-30.0 / 20.0)
    candidates = np.nonzero(live & band)[0]
    if candidates.size == 0:
        return 0.0, 0.0

    nyquist = sr / 2.0
    # Smooth before scanning: a single noisy bin must not look like a step.
    predicted = np.convolve(median, np.ones(9) / 9.0, mode="same")
    scan_upper = min(_CUTOFF_SCAN_MAX_HZ, nyquist - _CUTOFF_NYQUIST_GUARD_HZ)
    best_cutoff = 0.0
    best_drop = 0.0
    for index in candidates:
        cutoff = float(freqs[index])
        if cutoff > scan_upper:
            break
        lo = (freqs >= max(0.0, cutoff - _CUTOFF_WINDOW_HZ)) & (freqs <= cutoff)
        hi = (freqs > cutoff) & (freqs <= min(cutoff + _CUTOFF_AFTER_HZ, nyquist))
        if not np.any(lo) or not np.any(hi):
            continue
        below = float(np.median(predicted[lo]))
        above_level = float(np.median(predicted[hi]))
        drop = 20.0 * math.log10(max(below, 1e-20) / max(above_level, 1e-20))
        if drop > best_drop:
            best_drop = drop
            best_cutoff = cutoff

    if best_drop < _BRICKWALL_MIN_DROP_DB:
        return 0.0, 0.0  # a dip, not a brickwall
    confidence = _clip((best_drop - _BRICKWALL_MIN_DROP_DB) / 40.0, 0.0, 1.0)
    return round(best_cutoff, 2), round(confidence, 4)


# --- misc -------------------------------------------------------------------
def _zero_crossing_rate(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    signs = np.signbit(y)
    return _clip(float(np.mean(signs[1:] != signs[:-1])), 0.0, 1.0)


def _energy_curve(y: np.ndarray, bucket_count: int = 32) -> list[float]:
    """RMS per bucket over the whole analysis window, max bucket normalised to 1."""
    if y.size == 0:
        return [0.0] * bucket_count
    edges = np.linspace(0, y.size, bucket_count + 1).astype(np.int64)
    rms = np.zeros(bucket_count, dtype=np.float64)
    for i in range(bucket_count):
        start, stop = int(edges[i]), int(edges[i + 1])
        if stop <= start:
            continue
        segment = y[start:stop].astype(np.float64, copy=False)
        rms[i] = math.sqrt(float(np.mean(np.square(segment))))
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return [0.0] * bucket_count
    return [round(_clip(v / peak, 0.0, 1.0), 6) for v in rms]


def analyze(
    path: str | Path, sr: int = 22050, max_seconds: float = 300.0
) -> AnalysisMetrics:
    """Measure one audio file: rhythm, key, loudness, stereo image, timbre.

    Rhythm/tonality/loudness come from a mono decode at *sr*; stereo metrics from
    a stereo decode at the same rate; band balance and the codec cutoff from a
    short full-bandwidth decode. At most *max_seconds* of audio is used.

    Raises :class:`AudioEngineError` when the file cannot be probed or decoded.
    Every numeric field of the result is a finite float.
    """
    target = Path(path)
    probe = probe_file(target)
    if probe.error:
        raise AudioEngineError(f"cannot analyse {target.name}: {probe.error}")

    rate = int(sr) if sr and int(sr) > 0 else 22050
    limit = float(max_seconds) if max_seconds and max_seconds > 0 else None

    mono, rate = decode_mono(target, rate, limit)
    stereo, _ = decode_stereo(target, rate, limit)
    if mono.size < _ONSET_N_FFT:
        raise AudioEngineError(
            f"not enough audio in {target.name} to analyse ({mono.size} samples)"
        )

    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0)
    analyzed_seconds = _f(mono.size / rate)
    duration = _f(probe.duration) or analyzed_seconds

    metrics = AnalysisMetrics(
        path=str(target),
        duration=duration,
        analyzed_seconds=analyzed_seconds,
        sample_rate=rate,
        channels=int(probe.channels) or int(stereo.shape[0]),
    )

    for name, value in _rhythm(mono, rate, analyzed_seconds).items():
        setattr(metrics, name, value)

    stats = _spectral_stats(mono, rate)
    metrics.spectral_centroid_hz = _f(stats["spectral_centroid_hz"])
    metrics.spectral_rolloff95_hz = _f(stats["spectral_rolloff95_hz"])
    metrics.spectral_flatness = _f(stats["spectral_flatness"])
    metrics.zero_crossing_rate = _zero_crossing_rate(mono)
    chroma = np.asarray(stats["chroma"], dtype=np.float64)
    metrics.chroma = [round(_f(v), 6) for v in chroma]
    root, mode, key_conf = _key_from_chroma(chroma)
    metrics.key_root, metrics.key_mode, metrics.key_confidence = root, mode, key_conf

    metrics.loudness_lufs = _f(_integrated_loudness_lufs(mono, rate))
    metrics.peak_dbfs = _peak_dbfs(mono)
    metrics.true_peak_dbfs = _true_peak_dbfs(mono)
    metrics.dynamic_range_db = _dynamic_range_db(mono, rate)
    metrics.crest_factor_db = _clip(metrics.peak_dbfs - _rms_dbfs(mono), 0.0, 60.0)
    metrics.clipped_samples = int(np.count_nonzero(np.abs(mono) >= 0.9995))
    metrics.silence_ratio = _silence_ratio(mono, rate)

    for name, value in _stereo_metrics(
        stereo, rate, mono_source=(int(probe.channels) <= 1)
    ).items():
        setattr(metrics, name, value)

    spectrum_rate = 48000 if int(probe.sample_rate or 0) > 44100 else 44100
    spectrum_limit = min(limit, _SPECTRUM_SECONDS) if limit else _SPECTRUM_SECONDS
    try:
        wide, wide_rate = decode_mono(target, spectrum_rate, spectrum_limit)
        wide = np.nan_to_num(wide, nan=0.0, posinf=0.0, neginf=0.0)
    except AudioEngineError:
        wide, wide_rate = mono, rate
    metrics.bands = _band_energies(wide, wide_rate)
    metrics.lowpass_cutoff_hz, metrics.cutoff_confidence = _lowpass_cutoff(wide, wide_rate)
    metrics.energy_curve = _energy_curve(mono)
    return metrics

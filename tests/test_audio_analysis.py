"""Hermetic tests for the audio reference-analysis subsystem.

Every fixture is generated locally with numpy + ``wave``; nothing is downloaded
and nothing is written outside ``tmp_path``. Tests that need ffmpeg are skipped
when it is not on PATH.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import wave
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suno_agents.audio import (  # noqa: E402
    AnalysisMetrics,
    AudioEngineError,
    AudioProbe,
    ReferenceProfile,
    analyze,
    build_profile,
    decode_mono,
    decode_stereo,
    probe_file,
)
from suno_agents.audio import ffbin  # noqa: E402

SR = 44100
CLICK_BPM = 120.0

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


# --- helpers ----------------------------------------------------------------
def _has_ffmpeg() -> bool:
    try:
        ffbin.require_ffmpeg()
    except AudioEngineError:
        return False
    return True


requires_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")


def write_wav(path: Path, data: np.ndarray, sr: int = SR) -> Path:
    """Write *data* (mono or ``(n, channels)``) as 16-bit PCM."""
    array = np.asarray(data, dtype=np.float64)
    channels = 1 if array.ndim == 1 else array.shape[1]
    pcm = (np.clip(array, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def pink_noise(n: int, rng: np.random.Generator, tilt: float = 1.0) -> np.ndarray:
    """Cheap pink-ish noise (``1/f**tilt`` magnitude tilt) of length *n*."""
    spectrum = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    freqs[0] = freqs[1]
    return np.fft.irfft(spectrum / freqs ** (tilt / 2.0), n)


def music_bed(seconds: float = 8.0, seed: int = 11) -> np.ndarray:
    """Full-bandwidth 120 BPM loop: kick, snare, hats, bass, pads, noise bed.

    Built so the time-median spectrum stays within 30 dB from 20 Hz to Nyquist,
    which is what a real lossless master looks like to the cutoff detector.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / CLICK_BPM
    n = int(seconds * SR)
    mix = np.zeros(n)

    def add(start_s: float, segment: np.ndarray, gain: float) -> None:
        start = int(start_s * SR)
        stop = min(n, start + segment.size)
        if stop <= start:
            return
        peak = np.max(np.abs(segment)) + 1e-12
        mix[start:stop] += gain * segment[: stop - start] / peak

    for k in range(int(seconds / beat)):
        tt = np.arange(int(0.18 * SR)) / SR
        sweep = 110.0 * np.exp(-tt / 0.03) + 50.0
        add(k * beat, np.exp(-tt / 0.12) * np.sin(2 * np.pi * np.cumsum(sweep) / SR), 0.35)
    for bar in range(int(seconds / (2 * beat))):
        for offset in (1, 3):
            m = int(0.12 * SR)
            add(
                bar * 2 * beat + offset * beat,
                pink_noise(m, rng, 1.0) * np.exp(-np.arange(m) / (0.035 * SR)),
                0.35,
            )
    for k in range(int(seconds / (beat / 2))):
        m = int(0.04 * SR)
        add(k * beat / 2, np.diff(np.concatenate([[0.0], pink_noise(m, rng, 0.6)])), 0.30)
    for k in range(int(seconds / (2 * beat))):
        f0 = 55.0 if k % 2 == 0 else 82.41
        tt = np.arange(int(2 * beat * SR)) / SR
        add(
            k * 2 * beat,
            np.minimum(1.0, tt / 0.02)
            * np.exp(-tt / 1.2)
            * (np.sin(2 * np.pi * f0 * tt) + 0.3 * np.sin(4 * np.pi * f0 * tt)),
            0.40,
        )
    for k, chord in enumerate(
        [
            (220.0, 277.18, 329.63),
            (293.66, 369.99, 440.0),
            (329.63, 415.30, 493.88),
            (220.0, 277.18, 329.63),
        ]
    ):
        tt = np.arange(int(2.0 * SR)) / SR
        pad = sum(np.sin(2 * np.pi * f * tt * (1 + 0.001 * np.sin(2 * np.pi * 5 * tt))) for f in chord)
        add(k * 2.0, pad, 0.20)
    add(0.0, pink_noise(n, rng, 1.2), 0.9)

    from scipy import signal as sp_signal

    mix = sp_signal.lfilter(*sp_signal.butter(2, 35.0 / (SR / 2), btype="high"), mix)
    return mix / (np.max(np.abs(mix)) + 1e-12) * 0.9


def click_track(bpm: float = CLICK_BPM, seconds: float = 8.0) -> np.ndarray:
    n = int(seconds * SR)
    track = np.zeros(n)
    m = int(0.02 * SR)
    click = np.sin(2 * np.pi * 1000.0 * np.arange(m) / SR) * np.hanning(m) * 0.9
    period = int(round(60.0 / bpm * SR))
    for start in range(0, n - m, period):
        track[start : start + m] += click
    return track


def encode(src: Path, dst: Path, *args: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), *args, str(dst)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return dst


# --- fixtures ---------------------------------------------------------------
@pytest.fixture(scope="session")
def audio(tmp_path_factory) -> dict[str, Path]:
    """Session-wide fixture set; ffmpeg-dependent entries are omitted if absent."""
    directory = tmp_path_factory.mktemp("audio")
    files: dict[str, Path] = {}
    files["bed"] = write_wav(directory / "bed_8s.wav", music_bed())

    sine = 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(6 * SR) / SR)
    files["sine"] = write_wav(directory / "sine440.wav", sine)
    files["clicks"] = write_wav(directory / "clicks_120.wav", click_track())

    half_silent = np.concatenate([music_bed(4.0), np.zeros(4 * SR)])
    files["half_silent"] = write_wav(directory / "half_silent.wav", half_silent)
    files["clipped"] = write_wav(directory / "clipped.wav", np.ones(4 * SR) * 1.0 * np.sin(
        2 * np.pi * 220.0 * np.arange(4 * SR) / SR
    ) / np.maximum(np.abs(np.sin(2 * np.pi * 220.0 * np.arange(4 * SR) / SR)), 1e-9) * 0.9998)

    bed = music_bed()
    delay = int(0.003 * SR)
    right = np.concatenate([np.zeros(delay), bed[:-delay]])
    right = right + 0.25 * np.sin(2 * np.pi * 8000.0 * np.arange(bed.size) / SR)
    stereo = np.stack([bed, right], axis=1)
    files["stereo"] = write_wav(directory / "stereo_8s.wav", stereo / np.max(np.abs(stereo)) * 0.9)

    files["garbage"] = directory / "not_audio.txt"
    files["garbage"].write_text("this is not an audio file\n", encoding="utf-8")

    if _has_ffmpeg():
        files["mp3"] = encode(
            files["bed"], directory / "bed_128k.mp3", "-b:a", "128k"
        )
        files["mp3_upsampled"] = encode(
            files["mp3"], directory / "bed_128k_upsampled_96k.wav", "-ar", "96000"
        )
    return files


# --- package surface --------------------------------------------------------
def test_public_surface_is_reexported() -> None:
    import suno_agents.audio as audio

    expected = {
        "AudioEngineError",
        "AudioProbe",
        "AnalysisMetrics",
        "ReferenceProfile",
        "probe_file",
        "decode_mono",
        "decode_stereo",
        "analyze",
        "build_profile",
    }
    assert expected <= set(audio.__all__)
    assert audio.AudioProbe is AudioProbe
    assert issubclass(AudioEngineError, RuntimeError)


def test_import_is_safe_without_ffmpeg(tmp_path: Path) -> None:
    """Importing the package must not touch ffmpeg, even with an empty PATH."""
    env = dict(os.environ)
    env["PATH"] = str(tmp_path)
    env["FFMPEG_BIN"] = str(tmp_path / "missing-ffmpeg")
    env["FFPROBE_BIN"] = str(tmp_path / "missing-ffprobe")
    result = subprocess.run(
        [sys.executable, "-c", "import suno_agents.audio"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_ffmpeg_raises_single_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("FFMPEG_BIN", str(tmp_path / "nope-ffmpeg"))
    monkeypatch.setenv("FFPROBE_BIN", str(tmp_path / "nope-ffprobe"))
    ffbin.reset_cache()
    try:
        with pytest.raises(AudioEngineError) as excinfo:
            ffbin.require_ffmpeg()
        assert "FFMPEG_BIN" in str(excinfo.value) or "ffmpeg" in str(excinfo.value)
    finally:
        monkeypatch.undo()
        ffbin.reset_cache()


@requires_ffmpeg
def test_probe_failure_does_not_raise(audio: dict[str, Path]) -> None:
    probe = probe_file(audio["garbage"])
    assert isinstance(probe, AudioProbe)
    assert probe.error
    assert isinstance(probe.error, str)

    missing = probe_file(audio["garbage"].parent / "does-not-exist.wav")
    assert missing.error == "file not found"


# --- probe ------------------------------------------------------------------
@requires_ffmpeg
def test_probe_reads_wav_facts(audio: dict[str, Path]) -> None:
    probe = probe_file(audio["bed"])
    assert probe.error is None
    assert probe.codec == "pcm_s16le"
    assert "wav" in probe.container
    assert probe.sample_rate == SR
    assert probe.channels == 1
    assert probe.lossless is True
    assert probe.bit_rate and probe.bit_rate > 100_000
    assert probe.bits_per_raw_sample == 16
    assert probe.duration == pytest.approx(8.0, abs=0.15)
    assert all(key == key.lower() for key in probe.tags)


@requires_ffmpeg
def test_probe_reads_stereo_channels(audio: dict[str, Path]) -> None:
    probe = probe_file(audio["stereo"])
    assert probe.channels == 2
    assert probe.sample_rate == SR


@requires_ffmpeg
def test_probe_accepts_str_path(audio: dict[str, Path]) -> None:
    assert probe_file(str(audio["bed"])).path == str(audio["bed"])


@requires_ffmpeg
def test_probe_flags_lossy_codec(audio: dict[str, Path]) -> None:
    probe = probe_file(audio["mp3"])
    assert probe.error is None
    assert probe.codec == "mp3"
    assert probe.lossless is False


# --- decode -----------------------------------------------------------------
@requires_ffmpeg
def test_decode_mono_native_rate(audio: dict[str, Path]) -> None:
    samples, rate = decode_mono(audio["bed"])
    assert rate == SR
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert samples.size == pytest.approx(8 * SR, rel=0.02)
    assert 0.0 < float(np.max(np.abs(samples))) <= 1.0


@requires_ffmpeg
def test_decode_mono_resamples_and_truncates(audio: dict[str, Path]) -> None:
    samples, rate = decode_mono(audio["bed"], sr=8000, max_seconds=2.0)
    assert rate == 8000
    assert samples.size == pytest.approx(2 * 8000, rel=0.05)


@requires_ffmpeg
def test_decode_stereo_duplicates_mono(audio: dict[str, Path]) -> None:
    stereo, rate = decode_stereo(audio["bed"], sr=22050)
    assert stereo.shape[0] == 2
    assert rate == 22050
    assert np.array_equal(stereo[0], stereo[1])


@requires_ffmpeg
def test_decode_stereo_keeps_real_channels(audio: dict[str, Path]) -> None:
    stereo, _ = decode_stereo(audio["stereo"])
    assert stereo.shape[0] == 2
    assert not np.array_equal(stereo[0], stereo[1])


@requires_ffmpeg
def test_decode_missing_file_raises_engine_error(tmp_path: Path) -> None:
    with pytest.raises(AudioEngineError):
        decode_mono(tmp_path / "nope.wav")


@requires_ffmpeg
def test_decode_garbage_raises_engine_error(audio: dict[str, Path]) -> None:
    with pytest.raises(AudioEngineError):
        decode_mono(audio["garbage"])


# --- analyze: sanity and finiteness ----------------------------------------
def _numeric_values(metrics: AnalysisMetrics):
    for field in fields(metrics):
        value = getattr(metrics, field.name)
        if isinstance(value, float):
            yield field.name, value
        elif isinstance(value, int):
            yield field.name, float(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                yield f"{field.name}.{key}", float(item)
        elif isinstance(value, list) and field.name == "energy_curve":
            for index, item in enumerate(value):
                yield f"{field.name}[{index}]", float(item)


@requires_ffmpeg
def test_analyze_outputs_are_finite(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["bed"])
    assert isinstance(metrics, AnalysisMetrics)
    assert metrics.path == str(audio["bed"])
    assert metrics.sample_rate == 22050
    assert metrics.analyzed_seconds == pytest.approx(8.0, abs=0.2)
    assert metrics.duration == pytest.approx(8.0, abs=0.2)
    for name, value in _numeric_values(metrics):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"


@requires_ffmpeg
def test_bands_and_energy_curve(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["bed"])
    assert set(metrics.bands) == {
        "sub",
        "low",
        "low_mid",
        "mid",
        "high_mid",
        "high",
        "air",
    }
    assert max(metrics.bands.values()) == pytest.approx(0.0, abs=1e-6)
    assert all(value <= 0.0 for value in metrics.bands.values())
    assert len(metrics.energy_curve) == 32
    assert max(metrics.energy_curve) == pytest.approx(1.0, abs=1e-6)
    assert len(metrics.chroma) == 12
    assert max(metrics.chroma) == pytest.approx(1.0, abs=1e-6)


@requires_ffmpeg
def test_analyze_rejects_too_short_or_missing(tmp_path: Path) -> None:
    with pytest.raises(AudioEngineError):
        analyze(tmp_path / "nope.wav")
    tiny = write_wav(tmp_path / "tiny.wav", np.zeros(64))
    with pytest.raises(AudioEngineError):
        analyze(tiny)


@requires_ffmpeg
def test_silence_ratio_and_clipping(audio: dict[str, Path]) -> None:
    quiet = analyze(audio["half_silent"])
    assert quiet.silence_ratio == pytest.approx(0.5, abs=0.1)
    clipped = analyze(audio["clipped"])
    assert clipped.clipped_samples > 0
    assert clipped.peak_dbfs > -0.1


# --- analyze: musical sanity ------------------------------------------------
@requires_ffmpeg
def test_click_track_tempo(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["clicks"])
    assert metrics.tempo_bpm == pytest.approx(CLICK_BPM, abs=2.0)
    assert 0.0 < metrics.tempo_confidence <= 1.0
    assert metrics.beat_strength > 0.0
    assert metrics.onset_rate == pytest.approx(2.0, abs=0.6)


def _beat_track(bpm: float, offbeat: float | None = None, seconds: float = 10.0) -> np.ndarray:
    """Click track at *bpm*, optionally with an off-beat at *offbeat* of the beat."""
    n = int(seconds * SR)
    track = np.zeros(n)
    m = int(0.02 * SR)
    click = np.sin(2 * np.pi * 1000.0 * np.arange(m) / SR) * np.hanning(m) * 0.9
    beat = 60.0 / bpm
    step = 0
    while step * beat < seconds - 0.1:
        offsets = (0.0,) if offbeat is None else (0.0, offbeat * beat)
        for offset in offsets:
            start = int((step * beat + offset) * SR)
            if start + m <= n:
                track[start : start + m] += click
        step += 1
    return track


@requires_ffmpeg
@pytest.mark.parametrize("bpm", [60.0, 90.0, 120.0, 128.0, 160.0, 200.0])
def test_tempo_is_not_the_half_time_octave(tmp_path: Path, bpm: float) -> None:
    """A click train is periodic at half its tempo too; the faster octave must win."""
    metrics = analyze(write_wav(tmp_path / f"click_{bpm:.0f}.wav", _beat_track(bpm)))
    assert metrics.tempo_bpm == pytest.approx(bpm, abs=2.0), metrics.tempo_bpm
    assert metrics.tempo_confidence > 0.5


@requires_ffmpeg
def test_tempo_with_eighth_note_offbeats(tmp_path: Path) -> None:
    """Eighth notes double the event rate; the beat, not the subdivision, is reported."""
    metrics = analyze(write_wav(tmp_path / "eighths.wav", _beat_track(120.0, offbeat=0.5)))
    assert metrics.tempo_bpm == pytest.approx(120.0, abs=2.0), metrics.tempo_bpm


@requires_ffmpeg
@pytest.mark.parametrize(
    ("offbeat", "expected"),
    [(0.5, 0.5), (0.6, 0.6), (2.0 / 3.0, 0.667)],
)
def test_swing_ratio_measures_offbeat_placement(
    tmp_path: Path, offbeat: float, expected: float
) -> None:
    """0.5 = straight eighths, 0.667 = triplet shuffle: the classic swing reading."""
    track = _beat_track(120.0, offbeat=offbeat)
    metrics = analyze(write_wav(tmp_path / f"swing_{offbeat:.2f}.wav", track))
    assert metrics.swing_ratio == pytest.approx(expected, abs=0.05)


@requires_ffmpeg
def test_swing_ratio_defaults_to_straight_without_offbeats(tmp_path: Path) -> None:
    metrics = analyze(write_wav(tmp_path / "plain.wav", click_track(120.0)))
    assert metrics.swing_ratio == pytest.approx(0.5, abs=0.05)


@requires_ffmpeg
def test_pure_sine_is_key_a_and_centred_at_440(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["sine"])
    assert metrics.key_root in {"A", "A#", "G#"}, metrics.key_root
    assert metrics.spectral_centroid_hz == pytest.approx(440.0, rel=0.35)
    assert metrics.peak_dbfs == pytest.approx(-6.02, abs=0.5)
    assert metrics.true_peak_dbfs <= metrics.peak_dbfs + 0.5
    assert metrics.clipped_samples == 0
    assert metrics.silence_ratio == 0.0
    assert metrics.stereo_width == 0.0
    assert metrics.lowpass_cutoff_hz == 0.0


@requires_ffmpeg
def test_swing_ratio_in_range(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["clicks"])
    assert 0.2 <= metrics.swing_ratio <= 0.85


@requires_ffmpeg
def test_stereo_metrics_on_mono_and_stereo(audio: dict[str, Path]) -> None:
    mono = analyze(audio["bed"])
    assert mono.stereo_width == 0.0
    assert mono.stereo_correlation == pytest.approx(1.0, abs=1e-6)

    wide = analyze(audio["stereo"])
    assert wide.stereo_width > 0.1
    assert wide.stereo_correlation < 0.99
    assert -1.0 <= wide.stereo_correlation <= 1.0
    assert math.isfinite(wide.mid_side_balance_db)


# --- codec cutoff / lossless detection --------------------------------------
@requires_ffmpeg
def test_lossless_wav_has_no_cutoff(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["bed"])
    assert metrics.lowpass_cutoff_hz == 0.0


@requires_ffmpeg
def test_mp3_reports_a_brickwall(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["mp3"])
    assert metrics.lowpass_cutoff_hz > 0.0
    assert metrics.lowpass_cutoff_hz <= 20500.0
    assert 0.0 < metrics.cutoff_confidence <= 1.0


@requires_ffmpeg
def test_upsampled_mp3_reports_brickwall_below_nyquist(audio: dict[str, Path]) -> None:
    metrics = analyze(audio["mp3_upsampled"])
    assert metrics.lowpass_cutoff_hz > 0.0
    assert metrics.lowpass_cutoff_hz < 20500.0


# --- build_profile ----------------------------------------------------------
def test_build_profile_without_paths() -> None:
    profile = build_profile([])
    assert isinstance(profile, ReferenceProfile)
    assert profile.analyzed is False
    assert profile.lossless_verdict == "unknown"
    assert profile.quality_score == 0
    assert profile.error


@requires_ffmpeg
def test_build_profile_on_lossless_reference(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["bed"]])
    assert profile.analyzed is True
    assert profile.error is None
    assert profile.lossless_verdict == "lossless"
    assert profile.metrics is not None
    assert profile.metrics.path == str(audio["bed"])
    assert len(profile.per_file) == 1
    assert 0 <= profile.quality_score <= 100
    assert profile.quality_score >= 85
    assert profile.strengths


@requires_ffmpeg
def test_build_profile_flags_lossy(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["mp3"]])
    assert profile.lossless_verdict == "lossy"
    assert profile.warnings
    assert any("lossy" in warning.lower() for warning in profile.warnings)
    assert profile.quality_score <= 65


@requires_ffmpeg
def test_build_profile_flags_transcoded_lossless_container(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["mp3_upsampled"]])
    assert profile.probes[0].lossless is True
    assert profile.probes[0].sample_rate == 96000
    assert profile.lossless_verdict == "lossy-transcoded"
    assert any("upsampled" in warning.lower() for warning in profile.warnings)
    assert profile.quality_score <= 60


@requires_ffmpeg
def test_build_profile_mixed_and_ordering(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["mp3"], audio["bed"]])
    assert [m.path for m in profile.per_file] == [str(audio["mp3"]), str(audio["bed"])]
    assert profile.metrics is not None
    assert profile.metrics.path == str(audio["mp3"])
    assert profile.lossless_verdict == "mixed"
    assert profile.quality_score <= 65


@requires_ffmpeg
def test_build_profile_survives_unreadable_input(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["garbage"], audio["bed"]])
    assert profile.lossless_verdict == "lossless"  # garbage is not a usable probe
    assert profile.probes[0].error is not None
    assert profile.analyzed is True
    assert len(profile.per_file) == 1


@requires_ffmpeg
def test_build_profile_without_any_audio(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["garbage"]])
    assert profile.analyzed is False
    assert profile.metrics is None
    assert profile.error
    assert "not_audio.txt" in profile.error or "nothing could be analysed" in profile.error


@requires_ffmpeg
def test_summary_contains_every_required_fact(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["bed"]])
    summary = profile.summary
    assert summary
    assert 500 < len(summary) <= 1800

    lowered = summary.lower()
    required = [
        "pcm_s16le",  # codec
        "wav",  # container
        "kbps",  # bitrate
        "44100 hz",  # sample rate
        " ch",  # channels
        "8.0s",  # duration
        "bpm",
        "confidence",
        "key:",
        "major" if profile.metrics and profile.metrics.key_mode == "major" else "minor",
        "lufs",
        "peak",
        "crest",
        "stereo",
        "width",
        "correlation",
        "centroid",
        "rolloff95",
        "cutoff",
        "band balance",
        "energy curve",
        "verdict",
        "quality score",
        "lossless",
    ]
    for token in required:
        assert token in lowered, f"summary is missing {token!r}"
    for band in ("sub", "low", "low-mid", "mid", "high-mid", "high", "air"):
        assert band in lowered


@requires_ffmpeg
def test_profile_round_trips_to_dict(audio: dict[str, Path]) -> None:
    profile = build_profile([audio["bed"]])
    data = profile.to_dict()
    assert isinstance(data, dict)
    assert data["lossless_verdict"] == "lossless"
    assert data["metrics"]["path"] == str(audio["bed"])
    assert len(data["per_file"]) == 1
    assert isinstance(data["probes"][0]["tags"], dict)


# --- misc -------------------------------------------------------------------
def test_expected_number_of_dataclass_fields_unchanged() -> None:
    """Guards against silently redefining the frozen contract."""
    assert {f.name for f in fields(AudioProbe)} == {
        "path",
        "codec",
        "container",
        "sample_rate",
        "channels",
        "bit_rate",
        "duration",
        "bits_per_raw_sample",
        "tags",
        "lossless",
        "error",
    }
    assert {f.name for f in fields(AnalysisMetrics)} == {
        "path",
        "duration",
        "analyzed_seconds",
        "sample_rate",
        "channels",
        "tempo_bpm",
        "tempo_confidence",
        "beat_strength",
        "onset_rate",
        "swing_ratio",
        "key_root",
        "key_mode",
        "key_confidence",
        "chroma",
        "loudness_lufs",
        "peak_dbfs",
        "true_peak_dbfs",
        "dynamic_range_db",
        "crest_factor_db",
        "clipped_samples",
        "silence_ratio",
        "stereo_width",
        "stereo_correlation",
        "mid_side_balance_db",
        "spectral_centroid_hz",
        "spectral_rolloff95_hz",
        "spectral_flatness",
        "zero_crossing_rate",
        "lowpass_cutoff_hz",
        "cutoff_confidence",
        "bands",
        "energy_curve",
    }
    assert is_dataclass(ReferenceProfile)


# --- codec cutoff: edge selection and rate independence ---------------------
def test_cutoff_is_independent_of_the_analysis_sample_rate(audio: dict[str, Path]) -> None:
    """The reported brickwall must be a property of the file, not of our decoder.

    A lossy encoder leaves low-level quantisation noise above its cutoff. If the
    edge were chosen as "just after the highest surviving bin", decoding at a
    higher rate would raise the reported cutoff - and eventually hide it
    entirely, which is exactly how an upsample gets mistaken for a lossless
    master. The strongest-collapse scan must yield the same answer at either rate.
    """
    import numpy as np

    from suno_agents.audio.analysis import _lowpass_cutoff
    from suno_agents.audio.decode import decode_mono

    for key in ("mp3", "mp3_upsampled"):
        path = audio.get(key)
        if path is None:
            continue  # ffmpeg unavailable
        results = []
        for rate in (44100, 48000):
            samples, actual = decode_mono(path, sr=rate, max_seconds=20)
            assert actual == rate
            results.append(_lowpass_cutoff(np.asarray(samples), rate))
        (cut_a, conf_a), (cut_b, conf_b) = results
        assert cut_a > 0.0 and cut_b > 0.0, results
        assert abs(cut_a - cut_b) <= 300.0, results
        assert abs(conf_a - conf_b) <= 0.25, results


@requires_ffmpeg
def test_a_real_mp3_reports_a_brickwall_that_is_not_the_container_limit(
    audio: dict[str, Path],
) -> None:
    """A real 128 kbps MP3 must report a brickwall, and not the decode ceiling.

    Deliberately loose: the encoder build decides whether the edge lands at 16 kHz
    (older LAME) or just under 20.1 kHz (current LAME). What must never happen is
    the reading that would make the app trust a lossy file - no edge at all, or an
    edge sitting on our own scan ceiling.
    """
    metrics = analyze(audio["mp3"])
    assert metrics.lowpass_cutoff_hz > 0.0, "128 kbps MP3 must show a brickwall"
    assert metrics.lowpass_cutoff_hz <= 20500.0
    assert metrics.cutoff_confidence >= 0.5
    assert build_profile([audio["mp3"]]).lossless_verdict == "lossy"


def brickwalled_noise(sr: int, seconds: float, cutoff: float, seed: int) -> np.ndarray:
    """Dense noise with a hard low-pass exactly at ``cutoff``.

    Dense, because the cutoff is measured from the *median* magnitude per band: a
    sparse signal (a few sine tones) would show only its noise floor between the
    tone bins. And cut in the frequency domain rather than with a Butterworth
    filter, so the edge is a genuine brickwall and the expected answer is exact
    instead of depending on a filter's transition width. The result is normalised
    so the pass band sits near full scale, like real material.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    spectrum = np.fft.rfft(rng.normal(0.0, 1.0, n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum[freqs > cutoff] = 0.0
    out = np.fft.irfft(spectrum, n)
    return out / (np.max(np.abs(out)) + 1e-12) * 0.7


def test_a_known_brickwall_is_located_exactly(tmp_path: Path) -> None:
    """The edge detector must find a brickwall where one actually is.

    Deterministic and encoder-independent: 44.1 kHz content that genuinely stops
    at 12 kHz, with a whisper of noise above it exactly the way an encoder leaves
    residue, must report ~12 kHz rather than sliding up toward Nyquist.
    """
    mix = brickwalled_noise(44100, 8.0, 12000.0, seed=11)
    mix = mix + 1e-6 * np.random.default_rng(12).normal(0.0, 1.0, mix.size)
    path = write_wav(tmp_path / "cut12k.wav", mix)
    metrics = analyze(path)
    assert 11500.0 <= metrics.lowpass_cutoff_hz <= 12300.0, metrics.lowpass_cutoff_hz
    assert metrics.cutoff_confidence > 0.5


def test_a_brickwalled_file_in_a_lossless_container_is_flagged(tmp_path: Path) -> None:
    """The real-world trap, built by hand: 96 kHz WAV holding 16 kHz content.

    Nothing here depends on ffmpeg or on an encoder build. This is the exact file
    shape a user produces by "converting my MP3 to WAV for better quality", and
    the app must refuse to call it lossless.
    """
    sr = 96000
    mix = brickwalled_noise(sr, 6.0, 16000.0, seed=5)
    mix = mix + 1e-6 * np.random.default_rng(6).normal(0.0, 1.0, mix.size)
    path = write_wav(tmp_path / "upsampled96.wav", mix, sr=sr)

    profile = build_profile([path])
    metrics = profile.per_file[0]
    assert metrics.lowpass_cutoff_hz > 0.0, "the 16 kHz wall must be found"
    assert 14500.0 <= metrics.lowpass_cutoff_hz <= 16500.0, metrics.lowpass_cutoff_hz
    assert profile.lossless_verdict == "lossy-transcoded"
    assert profile.quality_score < 70
    assert profile.warnings
    assert "16" in profile.summary or "kHz" in profile.summary


@requires_ffmpeg
def test_upsampled_mp3_is_not_mistaken_for_a_lossless_master(audio: dict[str, Path]) -> None:
    """The 96 kHz upsampled MP3 is the exact trap this detector exists to catch."""
    from suno_agents.audio.profile import build_profile

    profile = build_profile([audio["mp3_upsampled"]])
    assert profile.lossless_verdict == "lossy-transcoded"
    assert profile.quality_score < 70
    primary = profile.metrics
    assert primary is not None
    assert 14000.0 <= primary.lowpass_cutoff_hz <= 20500.0, primary.lowpass_cutoff_hz
    assert profile.warnings, "a transcoded lossless file must warn the user"


def test_genuinely_full_bandwidth_material_reports_no_cutoff(audio: dict[str, Path]) -> None:
    """No brickwall must be invented for material that really does reach Nyquist."""
    metrics = analyze(audio["bed"])
    assert metrics.lowpass_cutoff_hz == 0.0
    assert metrics.cutoff_confidence == 0.0

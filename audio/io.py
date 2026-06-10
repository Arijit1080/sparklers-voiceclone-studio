"""
Audio I/O for the voiceclone studio.

Records from the USB codec mic via sounddevice, plays back via ALSA
through `aplay` (no python audio output dependency — aplay is rock solid
on the Jetson with the Waveshare codec).

Sample rate is fixed at 16000 mono — OpenVoice's reference encoder and
MeloTTS both work in 16k internally.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

# ---------- constants ----------

SAMPLE_RATE = 16000          # Hz — what OpenVoice/MeloTTS use internally
CHANNELS    = 1
DTYPE       = "int16"
DEFAULT_DEV = int(os.environ.get("SPARKLERS_AUDIO_IN", "0"))   # USB codec index
APLAY_DEV   = os.environ.get("SPARKLERS_APLAY_DEV", "default") # ALSA device

# ---------- helpers ----------

def list_input_devices() -> list[dict]:
    """Return a list of input-capable devices for the dashboard / debug page."""
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({
                "index": i,
                "name": d["name"],
                "default_samplerate": int(d.get("default_samplerate", 0)),
                "max_input_channels": int(d["max_input_channels"]),
            })
    return out


def db_rms(samples: np.ndarray) -> float:
    """Return RMS energy in dBFS for an int16 array."""
    if samples.size == 0:
        return -120.0
    x = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    return 20.0 * float(np.log10(rms))


# ---------- recording ----------

def record_clip(
    seconds: float,
    *,
    device: Optional[int] = None,
    samplerate: int = SAMPLE_RATE,
    on_progress: Optional[Callable[[float, float], None]] = None,
) -> np.ndarray:
    """
    Block-record `seconds` of mono audio.

    Some USB codecs only accept the channel count they natively support
    (e.g. the Waveshare codec exposes 2 channels and rejects 1-ch open).
    To stay portable we open the device at its native channel count and
    down-mix to mono in software.

    on_progress(t_seconds, rms_dbfs) is called ~10x/sec while the
    capture is running so the UI can draw a live VU meter.
    """
    if device is None:
        device = DEFAULT_DEV

    # discover the device's native input channel count and pick the
    # smallest valid open we can do (1 if the device allows, else its
    # native max — typically 2 for USB codecs).
    try:
        info = sd.query_devices(device)
        native_in = int(info.get("max_input_channels", 1))
    except Exception:
        native_in = 1
    open_ch = 1 if native_in <= 1 else min(2, native_in)

    n_total = int(round(seconds * samplerate))
    buf = np.zeros((n_total,), dtype=np.int16)
    cursor = 0
    chunk_n = max(1, samplerate // 10)  # ~100ms blocks

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=open_ch,
        dtype=DTYPE,
        device=device,
        blocksize=chunk_n,
    )
    stream.start()
    t0 = time.monotonic()
    try:
        while cursor < n_total:
            block, _ = stream.read(min(chunk_n, n_total - cursor))
            # block shape: (frames, open_ch); mix to mono by mean across channels
            if open_ch > 1:
                block = block.astype(np.int32).mean(axis=1).astype(np.int16)
            else:
                block = block.reshape(-1)
            buf[cursor:cursor + len(block)] = block
            cursor += len(block)
            if on_progress is not None:
                rms = db_rms(block)
                on_progress(time.monotonic() - t0, rms)
    finally:
        stream.stop()
        stream.close()
    return buf


def save_wav(path: Path, samples: np.ndarray, samplerate: int = SAMPLE_RATE) -> None:
    """Write an int16 mono WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(samples.astype(np.int16).tobytes())


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as int16 mono. Returns (samples, sample_rate).

    Fast path is the stdlib `wave` module (standard PCM RIFF WAV). When
    that fails — a non-PCM WAV (float / WAVE_FORMAT_EXTENSIBLE), or a
    compressed upload like MP3/FLAC/OGG/M4A — we fall back to soundfile
    (libsndfile) and then to ffmpeg, so any format the upload form
    accepts actually decodes instead of raising "file does not start
    with a RIFF id".
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n  = w.getnframes()
            ch = w.getnchannels()
            raw = w.readframes(n)
        samples = np.frombuffer(raw, dtype=np.int16)
        if ch > 1:
            samples = samples.reshape(-1, ch)[:, 0].copy()
        return samples, sr
    except Exception:
        pass  # not a plain PCM WAV — try the robust decoders below

    # soundfile handles WAV variants (float/extensible), FLAC, OGG, MP3
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="int16", always_2d=False)
        if data.ndim > 1:
            data = data[:, 0].copy()
        if data.size:
            return data.astype(np.int16), int(sr)
    except Exception:
        pass

    # ffmpeg as the universal fallback (M4A/AAC and anything else)
    samples, sr = _decode_with_ffmpeg(path)
    return samples, sr


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode any audio file to int16 mono 16 kHz via ffmpeg → raw PCM."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"could not decode {path.name}: unsupported audio format and "
            "ffmpeg is not installed"
        )
    cmd = [
        ffmpeg, "-nostdin", "-v", "error",
        "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "ignore").strip()[:400]
        raise RuntimeError(f"ffmpeg failed to decode {path.name}: {err}")
    samples = np.frombuffer(proc.stdout, dtype=np.int16)
    return samples, SAMPLE_RATE


# ---------- playback ----------

def play_wav(path: Path, device: str | None = None) -> None:
    """Play a WAV through aplay on the host's default ALSA device."""
    aplay = shutil.which("aplay")
    if aplay is None:
        raise RuntimeError("aplay not found — install alsa-utils")
    cmd = [aplay, "-q"]
    dev = device or APLAY_DEV
    if dev and dev != "default":
        cmd += ["-D", dev]
    cmd.append(str(path))
    subprocess.run(cmd, check=True)


def beep(freq_hz: int = 880, ms: int = 200, gain: float = 0.3) -> None:
    """Play a short sine beep as a recording cue."""
    n = int(round(ms / 1000.0 * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    s = (gain * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    # 5 ms fade in/out to avoid clicks
    fade = int(0.005 * SAMPLE_RATE)
    if n > 2 * fade:
        env = np.ones(n, dtype=np.float32)
        env[:fade]  = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        s *= env
    s16 = np.clip(s * 32767.0, -32768, 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_wav(tmp_path, s16)
        play_wav(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

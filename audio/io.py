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

    on_progress(t_seconds, rms_dbfs) is called ~10x/sec while the
    capture is running so the UI can draw a live VU meter.
    """
    if device is None:
        device = DEFAULT_DEV

    n_total = int(round(seconds * samplerate))
    buf = np.zeros((n_total,), dtype=np.int16)
    cursor = 0
    chunk_n = max(1, samplerate // 10)  # ~100ms blocks

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=CHANNELS,
        dtype=DTYPE,
        device=device,
        blocksize=chunk_n,
    )
    stream.start()
    t0 = time.monotonic()
    try:
        while cursor < n_total:
            block, _ = stream.read(min(chunk_n, n_total - cursor))
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
    """Load an int16 mono WAV (or stereo → first channel)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n  = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        samples = samples.reshape(-1, ch)[:, 0].copy()
    return samples, sr


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

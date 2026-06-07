"""
Synthesis pipeline — F5-TTS edition.

F5-TTS is a flow-matching, DiT-based TTS that conditions on a reference
WAV + its transcript at every synth call. Unlike OpenVoice (which
extracts a fixed embedding at enroll time and re-uses it), F5-TTS
re-encodes the reference each time — that's why it sounds dramatically
more like the speaker, but also why the model handle stays warm-loaded
while the per-call work is the inference itself.

Module-level cache holds one F5TTS instance.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from .paths import OUT_AUDIO
from .registry import embedding_path, enroll_wav_path, load_meta

log = logging.getLogger("voice.synth")

# ---------- lazy globals ----------

_F5 = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation knobs the F5-TTS team recommend in their README.
#   nfe_step  number of flow-matching steps. Higher = closer to model's
#             prior but slower. 32 is the default and a good RTF/quality
#             trade-off on the Orin Nano.
#   cfg_strength  classifier-free guidance scale. Higher = more
#             obedient to text + reference, lower = freer prosody.
#   sway_sampling_coef  flow-step sampler bias. -1.0 is the F5
#             default; produces best similarity in their A/B tests.
# Best-results settings on Jetson Orin Nano (chosen via A/B listening):
#   nfe_step=16      sub-realtime (RTF 0.76) with quality very close to 32
#   cfg_strength=3.0 stronger conditioning → closer to enrolled voice
DEFAULT_NFE         = 16
DEFAULT_CFG         = 3.0
DEFAULT_SWAY        = -1.0
DEFAULT_SPEED       = 1.0
TARGET_RMS          = 0.1   # reference loudness normalization

# When a voice's "transcript" sidecar is missing or empty we fall back
# to this string; F5-TTS will still produce audio but quality drops.
FALLBACK_REF_TEXT = "Hello, this is the reference audio for cloning."


def _get_f5():
    global _F5
    if _F5 is not None:
        return _F5
    from f5_tts.api import F5TTS
    log.info("loading F5-TTS on %s", _DEVICE)
    _F5 = F5TTS(device=_DEVICE)
    return _F5


# ---------- synth ----------

@dataclass
class SynthResult:
    voice_key: str
    text: str
    wav_path: Path
    seconds: float
    elapsed: float
    rtf: float
    base_speaker: str        # kept for web-layer compat; unused by F5
    tau: float               # ditto — we expose cfg_strength in its place


def _voice_ref_text(voice_key: str) -> str:
    """Read the saved transcript for this voice, or fall back."""
    meta = load_meta(voice_key)
    if meta is None:
        return FALLBACK_REF_TEXT
    txt = (getattr(meta, "ref_text", "") or "").strip()
    return txt if txt else FALLBACK_REF_TEXT


def speak(
    voice_key: str,
    text: str,
    *,
    speed: float = DEFAULT_SPEED,
    base_speaker: str = "F5-Base",       # unused; signature compat
    tau: float = DEFAULT_CFG,            # repurposed → cfg_strength
    nfe_step: int = DEFAULT_NFE,
    output: Optional[Path] = None,
) -> SynthResult:
    """Generate `text` in the enrolled voice using F5-TTS."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")

    meta = load_meta(voice_key)
    if meta is None:
        raise FileNotFoundError(f"voice '{voice_key}' is not enrolled")

    ref_wav = enroll_wav_path(voice_key)
    if not ref_wav.exists():
        raise FileNotFoundError(f"reference WAV missing: {ref_wav}")

    ref_text = _voice_ref_text(voice_key)

    f5 = _get_f5()

    if output is None:
        output = OUT_AUDIO / f"{voice_key}-{int(time.time()*1000)}.wav"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    # F5TTS.infer() writes the WAV when file_wave is given.
    f5.infer(
        ref_file=str(ref_wav),
        ref_text=ref_text,
        gen_text=text,
        file_wave=str(output),
        speed=speed,
        nfe_step=nfe_step,
        cfg_strength=tau,
        sway_sampling_coef=DEFAULT_SWAY,
        target_rms=TARGET_RMS,
        # F5's own silence-removal is too conservative — we do a stricter
        # RMS-based trim ourselves below.
        remove_silence=False,
        cross_fade_duration=0.15,
    )
    # Hard-trim silence from both ends so chunks abut cleanly.
    _hard_trim_silence(output)
    elapsed = time.monotonic() - t0

    # Read duration from header
    info = sf.info(str(output))
    seconds = info.frames / float(info.samplerate)

    rtf = elapsed / max(0.01, seconds)
    log.info("synth %s nfe=%d cfg=%.1f '%s' → %s (%.2fs audio, %.2fs wall, rtf=%.2f)",
             voice_key, nfe_step, tau, text[:40],
             output.name, seconds, elapsed, rtf)

    return SynthResult(
        voice_key=voice_key,
        text=text,
        wav_path=output,
        seconds=round(seconds, 2),
        elapsed=round(elapsed, 2),
        rtf=round(rtf, 2),
        base_speaker="F5-Base",
        tau=tau,
    )


def warmup() -> None:
    """Force-load F5-TTS so the first /speak call is hot."""
    _get_f5()


def _hard_trim_silence(
    wav_path: Path,
    head_db: float = -42.0,
    tail_db: float = -42.0,
    fade_ms: int = 15,
) -> None:
    """In-place trim leading/trailing silence on a WAV.

    F5-TTS's `remove_silence=True` uses pydub's silence-detect which is
    too conservative — it leaves ~150ms of low-amplitude breath/noise at
    both ends. For seamless multi-chunk playback we want a much tighter
    crop, then a tiny fade so the cut isn't a click.

    Uses RMS-window scanning over 20ms windows. Strict but reliable.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.size == 0:
        return

    win = max(1, int(sr * 0.020))  # 20ms windows
    # Compute per-window RMS
    n_full = (len(audio) // win) * win
    if n_full < win:
        return  # too short to bother
    rms = np.sqrt(np.mean(audio[:n_full].reshape(-1, win).astype(np.float64) ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)

    # find first window above head_db
    voiced = np.where(rms_db > head_db)[0]
    if voiced.size == 0:
        # whole clip is silent — leave it alone
        return
    start_w = voiced[0]
    # find last window above tail_db
    voiced_tail = np.where(rms_db > tail_db)[0]
    end_w = voiced_tail[-1] + 1

    start = start_w * win
    end = min(end_w * win, len(audio))
    cropped = audio[start:end]

    # short fade in / out so the boundary cut isn't audible as a click
    fade_samples = max(1, int(sr * fade_ms / 1000.0))
    if cropped.size > 2 * fade_samples:
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        cropped[:fade_samples] *= ramp
        cropped[-fade_samples:] *= ramp[::-1]

    sf.write(str(wav_path), cropped, sr)


# ---------- sentence-level streaming ----------

import re

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str, target_chars: int = 220, max_chars: int = 320) -> list[str]:
    """
    Split text into chunks that are roughly `target_chars` long and never
    longer than `max_chars`. Three goals:

    1. **Avoid tiny chunks.** "Hello." rendered alone is a 0.5 s audio
       clip the server takes ~0.8 s to produce — playback ends before
       the next chunk is ready and the listener hears a gap. Greedy
       merging keeps each chunk's audio long enough to mask the next
       chunk's render time.
    2. **Avoid giant chunks.** A 30-second chunk blocks playback start
       and feels like the streaming isn't streaming.
    3. **Keep cuts on natural pause points.** Always try sentence-level
       (.!?), only fall back to comma/semicolon if a single sentence
       blows past max_chars.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1. coarse-split on sentence punctuation
    raw = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]

    # 2. break any oversized sentence on commas/semicolons
    pieces: list[str] = []
    for sent in raw:
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        sub = [s.strip() for s in re.split(r'(?<=[,;:])\s+', sent) if s.strip()]
        buf = ""
        for s in sub:
            if len(buf) + 1 + len(s) <= max_chars:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    pieces.append(buf)
                buf = s
        if buf:
            pieces.append(buf)

    # 3. greedily merge adjacent pieces until we hit target_chars
    out: list[str] = []
    buf = ""
    for p in pieces:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= target_chars:
            buf = buf + " " + p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def speak_stream(
    voice_key: str,
    text: str,
    *,
    speed: float = DEFAULT_SPEED,
    base_speaker: str = "F5-Base",
    tau: float = DEFAULT_CFG,
    nfe_step: int = DEFAULT_NFE,
):
    """
    Generator that yields one SynthResult per sentence-sized chunk.

    The web layer streams these out as SSE events so the browser can
    start playback the moment the first chunk lands while the rest
    render in the background.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")

    chunks = split_sentences(text)
    log.info("streaming %d chunks for voice=%s", len(chunks), voice_key)

    for i, chunk in enumerate(chunks):
        result = speak(
            voice_key, chunk,
            speed=speed, base_speaker=base_speaker,
            tau=tau, nfe_step=nfe_step,
            output=OUT_AUDIO / f"{voice_key}-{int(time.time()*1000)}-{i:02d}.wav",
        )
        log.info("  chunk %d/%d: %r (%.2fs audio, rtf %.2f)",
                 i + 1, len(chunks), chunk[:48], result.seconds, result.rtf)
        yield result


def available_base_speakers() -> list[str]:
    """F5-TTS doesn't use base speakers — return a sentinel."""
    return ["F5-Base"]

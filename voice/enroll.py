"""
Enrollment pipeline — F5-TTS edition.

Unlike OpenVoice (which pre-extracts a fixed embedding at enroll time),
F5-TTS conditions on the reference audio + its transcript at every
synth call. So "enrollment" here means:

  1. record from the USB codec mic
  2. trim silence with webrtcvad
  3. crop to the centered 6–10 s window (F5-TTS quality plateaus there)
  4. auto-transcribe with HF Whisper tiny.en (~75 MB, on CUDA)
  5. persist the cleaned reference WAV + transcript on disk

The .pth "embedding" file we still emit is a 1-byte placeholder so the
registry's `embedding_path().exists()` check keeps working.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from audio import (
    SAMPLE_RATE,
    record_clip,
    save_wav,
    load_wav,
    beep,
    db_rms,
    trim_for_enrollment,
)
from .registry import (
    VoiceMeta,
    enroll_wav_path,
    embedding_path,
    save_meta,
    fresh_timestamp,
)

log = logging.getLogger("voice.enroll")

# ---------- shared models ----------

_WHISPER = None      # transformers pipeline


def _get_whisper():
    """HF transformers Whisper small.en — 244M params, far better than
    tiny.en for casual speech. Loaded on CPU so it doesn't compete with
    F5-TTS for Tegra's unified GPU memory. Transcribing 30 s of audio on
    CPU takes ~5-10 s, which is fine for a one-time enrollment step.
    """
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    from transformers import pipeline
    log.info("loading openai/whisper-small.en on cpu")
    _WHISPER = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",
        device=-1,
        chunk_length_s=30,
    )
    return _WHISPER


def _transcribe(wav_path: Path) -> str:
    """Auto-transcribe the reference clip. Returns "" on failure.

    We read the WAV via soundfile and pass {"array", "sampling_rate"}
    directly to the HF pipeline so we don't need ffmpeg.
    """
    try:
        import soundfile as sf
        asr = _get_whisper()
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        # HF Whisper wants 16 kHz mono; our enrollment audio is 16 kHz
        # already (SAMPLE_RATE=16000), but resample defensively.
        if sr != 16000:
            import numpy as np
            ratio = 16000 / sr
            n_new = int(round(len(audio) * ratio))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_new),
                np.arange(len(audio)),
                audio,
            ).astype("float32")
            sr = 16000
        out = asr({"array": audio, "sampling_rate": sr})
        text = (out.get("text") or "").strip() if isinstance(out, dict) else str(out).strip()
        return " ".join(text.split())
    except Exception:
        log.exception("whisper transcription failed")
        return ""


# ---------- enrollment ----------

@dataclass
class EnrollmentResult:
    key: str
    meta: VoiceMeta
    embedding_path: Path     # placeholder file — F5-TTS doesn't precompute
    seconds_kept: float


REF_TARGET_SEC = 3.0    # F5-TTS reprocesses the WHOLE reference at every
                        # denoising step, so synth time scales with
                        # reference length.  3s keeps time-to-first-audio
                        # under 2s on the Jetson Orin Nano.  Selectable in
                        # the UI (3 / 5.5 / 8 s) — longer = better speaker
                        # fidelity but slower start.

# Allowed reference lengths exposed in the UI.
REF_LENGTH_CHOICES = (3.0, 5.5, 8.0)


def _clean_onset_crop(samples: np.ndarray, sr: int, target_sec: float,
                      pre_pause_ms: int = 80) -> Optional[np.ndarray]:
    """Find a `target_sec` window that STARTS right after a natural pause.

    A centered crop usually begins mid-word at full energy; F5-TTS then
    reproduces that abrupt onset as an "eii"/"ahh" artifact at the start
    of EVERY generation.  Cropping so the reference begins silence→speech
    fixes it at the source.  Returns the int16 window, or None if no clean
    onset exists (caller falls back to a centered crop + leading silence).
    """
    a = samples.astype(np.float32)
    win = int(sr * 0.020)
    nf = (a.size // win) * win
    if nf < win:
        return None
    db = 20.0 * np.log10(
        np.sqrt(np.mean((a[:nf] / 32768.0).reshape(-1, win) ** 2, axis=1) + 1e-12) + 1e-12)
    sil = db < -45.0
    need_v = int(target_sec / 0.020)
    pre = max(1, int(pre_pause_ms / 20))
    best = None
    i = pre
    while i < len(sil) - need_v:
        if bool(np.all(sil[i - pre:i])):          # a pause just before i
            voiced_ratio = float((~sil[i:i + need_v]).mean())
            if voiced_ratio > 0.72 and (best is None or voiced_ratio > best[1]):
                best = (i, voiced_ratio)
            i += need_v
            continue
        i += 1
    if best is None:
        return None
    # start a few frames into the pause so the window opens on a little
    # leading silence, then the speech
    start = max(0, (best[0] - 3)) * win
    target_n = int(target_sec * sr)
    crop = a[start:start + target_n]
    if crop.size < target_n:
        return None
    return crop.astype(np.int16)


def enroll_from_wav(
    key: str,
    display_name: str,
    raw_wav_path: Path,
    *,
    ref_seconds: float = REF_TARGET_SEC,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> EnrollmentResult:
    """
    Clean-onset crop to `ref_seconds` → transcribe → persist meta + WAV.

    `on_progress(message, ratio_0_1)` is called at a few checkpoints.
    """
    if on_progress: on_progress("loading reference clip", 0.05)
    samples, sr = load_wav(raw_wav_path)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        n_new = int(round(len(samples) * ratio))
        samples = np.interp(
            np.linspace(0, len(samples) - 1, n_new),
            np.arange(len(samples)),
            samples.astype(np.float32),
        ).astype(np.int16)
        sr = SAMPLE_RATE

    if on_progress: on_progress("trimming silence (VAD)", 0.20)
    vad = trim_for_enrollment(samples, sr, aggressiveness=2)
    if vad.samples.size < int(0.5 * SAMPLE_RATE):
        raise ValueError(
            f"enrollment audio too short after VAD ({vad.samples.size / SAMPLE_RATE:.2f}s). "
            "Speak continuously for the full window."
        )

    target_n = int(ref_seconds * SAMPLE_RATE)
    # Prefer a CLEAN-ONSET crop from the raw (pre-VAD) audio so the
    # reference starts silence→speech and F5 never emits the leading
    # "eii"/"ahh" artifact.
    cropped = _clean_onset_crop(samples, sr, ref_seconds)
    if cropped is not None:
        if on_progress:
            on_progress(f"clean-onset {ref_seconds:.1f}s reference", 0.30)
    else:
        # fallback: centered crop of the VAD'd audio with a short leading
        # silence prepended so the onset is still clean
        c = vad.samples
        if c.size > target_n:
            start = (c.size - target_n) // 2
            c = c[start:start + target_n]
        lead = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.int16)
        cropped = np.concatenate([lead, c])[:target_n] if c.size else c
        if on_progress:
            on_progress(f"centered {ref_seconds:.1f}s reference (no pause found)", 0.30)

    cleaned_wav = enroll_wav_path(key)
    save_wav(cleaned_wav, cropped, SAMPLE_RATE)

    if on_progress: on_progress("transcribing reference (whisper-tiny.en)", 0.55)
    ref_text = _transcribe(cleaned_wav)
    if not ref_text:
        log.warning("[%s] empty transcript — F5-TTS will fall back to placeholder", key)
        ref_text = ""

    if on_progress: on_progress("writing voice metadata", 0.85)
    meta = VoiceMeta(
        key=key,
        display_name=display_name,
        enrolled_at=fresh_timestamp(),
        seconds=round(len(cropped) / SAMPLE_RATE, 2),
        rms_dbfs=round(vad.rms_dbfs, 1),
        voiced_ratio=round(vad.voiced_ratio, 2),
        source_wav=cleaned_wav.name,
        ref_text=ref_text,
    )
    save_meta(meta)

    # Placeholder so registry.embedding_path().exists() stays True.
    emb_path = embedding_path(key)
    if not emb_path.exists():
        emb_path.write_bytes(b"\x00")

    if on_progress: on_progress("done", 1.0)
    log.info("[%s] enrolled — %.2fs ref, transcript=%r",
             key, meta.seconds, ref_text[:80])

    return EnrollmentResult(
        key=key,
        meta=meta,
        embedding_path=emb_path,
        seconds_kept=meta.seconds,
    )


# ---------- reduce existing references (one-time migration) ----------

def set_reference_length(key: str, target_sec: float) -> Optional[float]:
    """Re-crop an enrolled voice's reference to `target_sec` with a CLEAN
    onset, preferring the original raw recording (which still has the
    pauses needed to find a silence→speech start).  Re-transcribes so the
    saved ref_text matches the new audio.  Returns the new duration, or
    None on failure.
    """
    from .registry import load_meta, save_meta

    wav = enroll_wav_path(key)
    raw = wav.with_suffix(".raw.wav")
    src = raw if raw.exists() else wav
    if not src.exists():
        log.warning("set_reference_length: no source WAV for voice=%s", key)
        return None

    samples, sr = load_wav(src)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        n_new = int(round(len(samples) * ratio))
        samples = np.interp(
            np.linspace(0, len(samples) - 1, n_new),
            np.arange(len(samples)), samples.astype(np.float32),
        ).astype(np.int16)
        sr = SAMPLE_RATE

    target_n = int(target_sec * sr)
    cropped = _clean_onset_crop(samples, sr, target_sec)
    if cropped is None:
        # fallback: centered crop + leading silence
        c = samples
        if c.size > target_n:
            st = (c.size - target_n) // 2
            c = c[st:st + target_n]
        lead = np.zeros(int(sr * 0.12), dtype=np.int16)
        cropped = np.concatenate([lead, c])[:target_n] if c.size else c
    save_wav(wav, cropped, sr)

    new_text = _transcribe(wav)
    meta = load_meta(key)
    if meta is not None:
        meta.seconds = round(len(cropped) / sr, 2)
        if new_text:
            meta.ref_text = new_text
        save_meta(meta)
    log.info("[%s] reference set to %.1fs (clean onset), re-transcribed=%r",
             key, len(cropped) / sr, (new_text or "")[:60])
    return len(cropped) / sr


# backwards-compat alias
def reduce_reference(key: str, target_sec: float = REF_TARGET_SEC,
                     ) -> Optional[float]:
    return set_reference_length(key, target_sec)


# ---------- live-mic enrollment ----------

def enroll_from_mic(
    key: str,
    display_name: str,
    *,
    seconds: float = 15.0,         # default down from 30 — F5 wants shorter clean refs
    ref_seconds: float = REF_TARGET_SEC,
    device: Optional[int] = None,
    play_cues: bool = True,
    on_progress: Optional[Callable[[str, float], None]] = None,
    on_meter: Optional[Callable[[float, float], None]] = None,
) -> EnrollmentResult:
    """
    Beep-start → record `seconds` → beep-stop → enroll.

    `on_meter(t_sec, rms_dbfs)` is forwarded from the recorder so the UI
    can render a live VU bar.
    """
    if play_cues:
        if on_progress: on_progress("get ready…", 0.01)
        beep(880, 180)
    if on_progress: on_progress(f"recording {seconds:.0f}s", 0.03)
    samples = record_clip(seconds, device=device, on_progress=on_meter)
    if play_cues:
        beep(440, 240)

    raw_path = enroll_wav_path(key).with_suffix(".raw.wav")
    save_wav(raw_path, samples, SAMPLE_RATE)
    if on_progress:
        on_progress(
            f"captured {len(samples) / SAMPLE_RATE:.1f}s "
            f"(peak {db_rms(samples):.1f} dBFS)",
            0.10,
        )

    return enroll_from_wav(key, display_name, raw_path,
                           ref_seconds=ref_seconds, on_progress=on_progress)

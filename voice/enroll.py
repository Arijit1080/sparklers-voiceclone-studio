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
    clean_reference,
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


def _transcribe_array(audio: np.ndarray, sr: int) -> str:
    """Transcribe a mono audio array (int16 or float32). Returns "" on failure.

    Passing the {"array", "sampling_rate"} dict to the HF pipeline avoids any
    file I/O, so the enroll-time window ranking can score several candidate
    crops cheaply.
    """
    try:
        asr = _get_whisper()
        a = audio[:, 0] if audio.ndim > 1 else audio
        a = a.astype(np.float32)
        if np.issubdtype(audio.dtype, np.integer):
            a = a / 32768.0
        # HF Whisper wants 16 kHz mono; our enrollment audio is 16 kHz already
        # (SAMPLE_RATE=16000), but resample defensively.
        if sr != 16000:
            n_new = int(round(len(a) * 16000 / sr))
            a = np.interp(np.linspace(0, len(a) - 1, n_new),
                          np.arange(len(a)), a).astype(np.float32)
            sr = 16000
        out = asr({"array": a, "sampling_rate": sr})
        text = (out.get("text") or "").strip() if isinstance(out, dict) else str(out).strip()
        return " ".join(text.split())
    except Exception:
        log.exception("whisper transcription failed")
        return ""


def _transcribe(wav_path: Path) -> str:
    """Auto-transcribe a reference WAV. Returns "" on failure."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        return _transcribe_array(audio, sr)
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

# Voiced-ratio band for picking the reference window (see _onset_candidates).
MIN_VOICED_RATIO = 0.72   # below this the window is mostly silence — unusable
MAX_VOICED_RATIO = 0.94   # above this the window is wall-to-wall speech with no
                          # inter-word pauses — i.e. the fastest-talking part of
                          # the clip.  F5 derives clone speed from the reference,
                          # so a crammed window makes the voice sound rushed.

# Target speech rate for a reference, in transcript-characters per second.
# Natural English speech sits around 12–15 ch/s.  The enroll-time selector
# picks the candidate window whose measured rate is closest to this, and the
# synth-time normalizer nudges playback toward it (see voice/synth.py).
TARGET_CH_PER_SEC   = 13.0
MIN_NATURAL_RATE    = 7.0    # below this a window is silence/garbled, not speech
MAX_RANK_CANDIDATES = 6      # cap Whisper calls during enroll-time ranking


def _onset_candidates(samples: np.ndarray, sr: int, target_sec: float,
                      pre_pause_ms: int = 80) -> list[tuple[np.ndarray, float]]:
    """Every `target_sec` window that starts right after a natural pause and is
    mostly voiced.  Each is (crop_int16, voiced_ratio).

    Starting silence→speech is what keeps F5 from emitting the "eii"/"ahh"
    onset artifact.  Naturally-paced windows (voiced_ratio ≤ MAX_VOICED_RATIO)
    are returned first so callers that only sample a few still see the good
    ones; crammed windows come last as a fallback.
    """
    a = samples.astype(np.float32)
    win = int(sr * 0.020)
    nf = (a.size // win) * win
    if nf < win:
        return []
    db = 20.0 * np.log10(
        np.sqrt(np.mean((a[:nf] / 32768.0).reshape(-1, win) ** 2, axis=1) + 1e-12) + 1e-12)
    sil = db < -45.0
    need_v = int(target_sec / 0.020)
    pre = max(1, int(pre_pause_ms / 20))
    target_n = int(target_sec * sr)
    out: list[tuple[np.ndarray, float]] = []
    i = pre
    while i < len(sil) - need_v:
        if bool(np.all(sil[i - pre:i])):          # a pause just before i
            vr = float((~sil[i:i + need_v]).mean())
            if vr >= MIN_VOICED_RATIO:
                # open a few frames into the pause for a little leading silence
                start = max(0, (i - 3)) * win
                crop = a[start:start + target_n]
                if crop.size >= target_n:
                    out.append((crop.astype(np.int16), vr))
            i += need_v
            continue
        i += 1
    # stable sort: naturally-paced (not crammed) windows first, in position order
    out.sort(key=lambda c: c[1] > MAX_VOICED_RATIO)
    return out


def _clean_onset_crop(samples: np.ndarray, sr: int, target_sec: float,
                      pre_pause_ms: int = 80) -> Optional[np.ndarray]:
    """Single best clean-onset window by the voiced-ratio heuristic (no
    transcription).  Used where a quick crop is enough; the enroll path uses
    the rate-aware `_select_natural_window` instead.  Returns None if no clean
    onset exists (caller falls back to a centered crop + leading silence).
    """
    cands = _onset_candidates(samples, sr, target_sec, pre_pause_ms)
    if not cands:
        return None
    natural = [c for c in cands if c[1] <= MAX_VOICED_RATIO]
    pool = natural or cands
    return max(pool, key=lambda c: c[1])[0]


def _select_natural_window(samples: np.ndarray, sr: int, target_sec: float,
                           max_rank: int = MAX_RANK_CANDIDATES,
                           ) -> tuple[Optional[np.ndarray], str, float]:
    """Pick the reference window whose *measured* speech rate is closest to
    TARGET_CH_PER_SEC.  This is the guarantee that a clip never lands on its
    fastest-talking segment again: we transcribe the top candidate windows and
    choose by chars-per-second directly, not by the voiced-ratio proxy.

    Returns (crop_int16, ref_text, rate_ch_per_sec); crop is None when there is
    no clean onset at all (caller falls back to a centered crop).
    """
    cands = _onset_candidates(samples, sr, target_sec)
    if not cands:
        return None, "", 0.0
    scored: list[tuple[np.ndarray, str, float, float]] = []
    for crop, vr in cands[:max_rank]:
        txt = _transcribe_array(crop, sr)
        rate = (len(txt) / target_sec) if txt else 0.0
        scored.append((crop, txt, rate, vr))
    # prefer windows that actually transcribed to real speech
    valid = [s for s in scored if s[2] >= MIN_NATURAL_RATE]
    pool = valid or scored
    best = min(pool, key=lambda s: abs(s[2] - TARGET_CH_PER_SEC))
    log.info("window selection: %s -> chose %.1f ch/s (vr=%.2f) from %d candidate(s)",
             [round(s[2], 1) for s in scored], best[2], best[3], len(cands))
    return best[0], best[1], best[2]


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
    # Pick the clean-onset window whose MEASURED speech rate is most natural.
    # This is the guarantee against the "rushed clone" bug: rather than trust a
    # voiced-ratio proxy, we transcribe the top candidate windows and choose by
    # chars-per-second, so the reference never lands on the clip's
    # fastest-talking 3 seconds.  (Clean onset also avoids the "eii" artifact.)
    if on_progress: on_progress("choosing a naturally-paced window", 0.30)
    cropped, ref_text, rate = _select_natural_window(samples, sr, ref_seconds)
    if cropped is not None:
        log.info("[%s] selected reference window: %.1f ch/s", key, rate)
        if on_progress:
            on_progress(f"clean-onset {ref_seconds:.1f}s reference ({rate:.0f} ch/s)", 0.45)
    else:
        # fallback: centered crop of the VAD'd audio with a short leading
        # silence prepended so the onset is still clean
        c = vad.samples
        if c.size > target_n:
            start = (c.size - target_n) // 2
            c = c[start:start + target_n]
        lead = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.int16)
        cropped = np.concatenate([lead, c])[:target_n] if c.size else c
        ref_text = ""    # transcribed after cleanup below
        if on_progress:
            on_progress(f"centered {ref_seconds:.1f}s reference (no pause found)", 0.45)

    # High-pass always; gentle spectral gate only if the clip is noisy.
    # F5-TTS clones whatever it hears, so a clean reference => clean clone.
    cropped, ns = clean_reference(cropped, SAMPLE_RATE)
    log.info("[%s] reference cleanup: hp=%.0fHz snr=%.1fdB denoised=%s",
             key, ns.get("highpass_hz", 0), ns.get("snr_db", 0.0), ns.get("denoised"))

    cleaned_wav = enroll_wav_path(key)
    save_wav(cleaned_wav, cropped, SAMPLE_RATE)

    # The window selector already transcribed the chosen crop, and the high-pass
    # doesn't change the words — reuse it.  Only re-transcribe on the fallback
    # path (or if the selector somehow came back empty).
    if not ref_text:
        if on_progress: on_progress("transcribing reference", 0.65)
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
    # Same rate-aware selection as fresh enrollment — re-cropping must never
    # land on the fastest-talking window either.
    cropped, new_text, rate = _select_natural_window(samples, sr, target_sec)
    if cropped is not None:
        log.info("[%s] re-crop selected window: %.1f ch/s", key, rate)
    else:
        # fallback: centered crop + leading silence
        c = samples
        if c.size > target_n:
            st = (c.size - target_n) // 2
            c = c[st:st + target_n]
        lead = np.zeros(int(sr * 0.12), dtype=np.int16)
        cropped = np.concatenate([lead, c])[:target_n] if c.size else c
        new_text = ""

    cropped, ns = clean_reference(cropped, sr)
    log.info("[%s] re-crop cleanup: hp=%.0fHz snr=%.1fdB denoised=%s",
             key, ns.get("highpass_hz", 0), ns.get("snr_db", 0.0), ns.get("denoised"))
    save_wav(wav, cropped, sr)

    if not new_text:                 # fallback path → transcribe the saved clip
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

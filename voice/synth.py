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

# ---------- one-shot CUDA performance knobs ----------
# These have to land before any module that touches CUDA runs (which
# basically means: as a side effect of importing voice.synth, before any
# F5-TTS code).
#
#   cudnn.benchmark    auto-pick the fastest cuDNN algo per input shape
#                      (one-time per shape; cached thereafter)
#   TF32 matmuls       Tensor Cores in TF32 mode for fp32 paths — F5 is
#                      mostly fp16 but text encoder + a few ops stay fp32
#   allow_tf32         legacy alias, set both to be safe
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    log.exception("could not enable cudnn perf knobs")

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

# ---------- automatic speech-rate normalization ----------
# F5 sets the clone's speech rate to (reference chars/sec) × speed.  Most
# references already sit in the natural 10–16 ch/s range and sound fine, so we
# DON'T retune them — we only rein in genuine outliers: a reference faster than
# NORM_BAND_HI or slower than NORM_BAND_LO is nudged just back to the band edge
# (the gentlest correction), then hard-clamped so F5 never stretches enough to
# sound artificial.  The enroll-time selector keeps new voices inside the band,
# so this is a backstop for odd clips / older voices.  Disable with
# SPARKLERS_AUTO_SPEED=0.
AUTO_SPEED_NORMALIZE = os.environ.get("SPARKLERS_AUTO_SPEED", "1") != "0"
NORM_BAND_LO         = 10.0   # ch/s — below this, gently speed up to the edge
NORM_BAND_HI         = 16.0   # ch/s — above this, gently slow down to the edge
NORM_SPEED_MIN       = 0.80
NORM_SPEED_MAX       = 1.15

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

    # ---------- torch.compile the hot path (OPT-IN) ----------
    # Default off because on Tegra Orin Nano (torch 2.8 + cu126 +
    # jetson-ai-lab wheel), inductor either hangs forever on variable
    # input shapes (reduce-overhead/CUDA Graphs) or recompiles per
    # shape until it timing-outs (default mode). Eager mode + cudnn
    # autotune already gets us most of the way.
    #
    # Set SPARKLERS_TRY_TORCH_COMPILE=1 to opt in if a future torch
    # wheel fixes the Tegra compile path.
    import os
    if (os.environ.get("SPARKLERS_TRY_TORCH_COMPILE", "") == "1"
            and hasattr(torch, "compile") and _DEVICE.startswith("cuda")):
        try:
            log.info("torch.compile(ema_model.transformer, mode='default')")
            _F5.ema_model.transformer = torch.compile(
                _F5.ema_model.transformer,
                mode="default", dynamic=True, fullgraph=False,
            )
        except Exception:
            log.exception("torch.compile failed — falling back to eager")

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


def _auto_speed_factor(meta) -> float:
    """Gentle playback-speed multiplier for out-of-band references.  Returns
    1.0 for any reference already in the natural [NORM_BAND_LO, NORM_BAND_HI]
    range (so good voices are never retuned); otherwise nudges just to the band
    edge, hard-clamped to [NORM_SPEED_MIN, NORM_SPEED_MAX].
    """
    if not AUTO_SPEED_NORMALIZE or meta is None:
        return 1.0
    txt = (getattr(meta, "ref_text", "") or "").strip()
    secs = float(getattr(meta, "seconds", 0.0) or 0.0)
    if not txt or secs <= 0.0:
        return 1.0
    ref_rate = len(txt) / secs
    if ref_rate < 1.0:
        return 1.0
    if ref_rate > NORM_BAND_HI:
        factor = NORM_BAND_HI / ref_rate        # too fast → slow toward edge
    elif ref_rate < NORM_BAND_LO:
        factor = NORM_BAND_LO / ref_rate        # too slow → speed toward edge
    else:
        return 1.0                              # already natural — leave alone
    return float(min(NORM_SPEED_MAX, max(NORM_SPEED_MIN, factor)))


def speak(
    voice_key: str,
    text: str,
    *,
    speed: float = DEFAULT_SPEED,
    base_speaker: str = "F5-Base",       # unused; signature compat
    tau: float = DEFAULT_CFG,            # repurposed → cfg_strength
    nfe_step: int = DEFAULT_NFE,
    output: Optional[Path] = None,
    trailing_pad_ms: int = 0,            # silence appended after trim —
                                         # used by streaming to put a
                                         # natural pause after a sentence
    tail_fade_ms: int = 0,               # >0 → longer fade-out for a
                                         # graceful end (last chunk only)
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

    # Normalize speech rate: nudge a fast/slow reference toward a natural pace.
    # `speed` from the caller stays a relative control on top of this.
    norm = _auto_speed_factor(meta)
    eff_speed = speed * norm

    f5 = _get_f5()

    if output is None:
        output = OUT_AUDIO / f"{voice_key}-{int(time.time()*1000)}.wav"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    # F5TTS.infer() returns (wav, sr, spectrogram) AND writes the WAV.
    # Capturing the return saves a second file open for duration calc.
    wav, sr, _ = f5.infer(
        ref_file=str(ref_wav),
        ref_text=ref_text,
        gen_text=text,
        file_wave=str(output),
        speed=eff_speed,
        nfe_step=nfe_step,
        cfg_strength=tau,
        sway_sampling_coef=DEFAULT_SWAY,
        target_rms=TARGET_RMS,
        # F5's own silence-removal is too conservative — we strip head/
        # tail silence ourselves in-memory and rewrite only if needed.
        remove_silence=False,
        cross_fade_duration=0.15,
    )
    # Cheap in-memory silence trim (no re-read; rewrites only when it
    # actually crops > ~30 ms).  trailing_pad_ms appends a controlled
    # pause after the trim so a sentence-final chunk gets a natural beat
    # before the next chunk butt-joins onto it.
    n_out = _hard_trim_silence_inplace(
        wav, sr, output, trailing_pad_ms=trailing_pad_ms,
        tail_fade_ms=tail_fade_ms,
    )
    elapsed = time.monotonic() - t0
    seconds = n_out / float(sr)

    rtf = elapsed / max(0.01, seconds)
    log.info("synth %s nfe=%d cfg=%.1f speed=%.2f(x%.2f) '%s' → %s (%.2fs audio, %.2fs wall, rtf=%.2f)",
             voice_key, nfe_step, tau, eff_speed, norm, text[:40],
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
    """Heavy-warm F5-TTS so the very first user-visible /speak is hot.

    Three things happen here:
      1. Load the F5TTS model (~10-20 s cold).
      2. Pre-cache reference-audio preprocessing for every enrolled
         voice (F5-TTS caches by audio bytes in-process).
      3. Run ONE dummy synth so cudnn.benchmark picks fastest kernels
         per shape and the inductor caches are filled. This is the
         single biggest perceived-speed win — the first real user
         synth then runs in steady-state RTF instead of paying a
         first-call cudnn-autotune tax (~2-3× slower).
    """
    f5 = _get_f5()

    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    from .registry import list_voices

    voices = list_voices()

    # 2) pre-cache reference preprocessing for each voice
    for v in voices:
        ref = enroll_wav_path(v.key)
        if not ref.exists():
            continue
        try:
            preprocess_ref_audio_text(
                str(ref), v.ref_text or "Hello.",
                show_info=lambda *_a, **_k: None,
            )
            log.info("ref-cache warmed for voice=%s", v.key)
        except Exception:
            log.exception("ref-cache warm failed for voice=%s", v.key)

    # 3) dummy synths across two text lengths to fill cuDNN kernel cache
    #    for short AND long shapes (cuDNN autotunes per shape).
    if voices:
        import time as _t
        warm_voice = voices[0].key
        warmup_texts = [
            "Hi.",                                              # short shape
            "This is a longer warmup so cuDNN picks the right "
            "kernels for paragraph-length input shapes too.",   # long shape
        ]
        for txt in warmup_texts:
            try:
                t0 = _t.monotonic()
                out_path = OUT_AUDIO / "_warmup_dummy.wav"
                speak(
                    warm_voice, txt,
                    nfe_step=DEFAULT_NFE, tau=DEFAULT_CFG,
                    output=out_path,
                )
                try: out_path.unlink()
                except OSError: pass
                log.info("dummy-synth (%d chars) warm complete in %.2fs",
                         len(txt), _t.monotonic() - t0)
            except Exception:
                log.exception("dummy-synth warm failed (text=%r)", txt[:30])


def _hard_trim_silence_inplace(
    audio,
    sr: int,
    wav_path: Path,
    head_db: float = -42.0,
    tail_db: float = -47.0,  # keep the word's real decay (above -47) but
                             # cut the low-level F5 noise-floor tail
                             # (-48..-54 dB) — left in, that faint garbled
                             # tail is audible as "distortion" in the
                             # quiet pause before the next chunk
    sil_db: float = -45.0,
    fade_ms: int = 8,        # short — just enough to kill edge clicks
    min_trim_ms: int = 30,
    keep_pause_ms: int = 280,
    max_internal_sil_ms: int = 320,
    trailing_pad_ms: int = 0,   # silence appended AFTER the fade-out, to
                                # give a sentence-final chunk a natural
                                # beat before the next chunk butt-joins on
    tail_fade_ms: int = 0,      # if >0, use this (longer) fade on the END
                                # only — a graceful taper for the last
                                # chunk instead of the abrupt 8ms cut
) -> int:
    """Clean the waveform F5-TTS just returned:

      1. Trim leading/trailing silence (head/tail).
      2. Compress any INTERNAL silence longer than `max_internal_sil_ms`
         down to `keep_pause_ms`.  F5-TTS intermittently emits 0.5-1.5 s
         dead-air gaps mid-utterance (a known flow-matching duration-
         prediction artifact); left in, they sound like the audio froze.
         We keep natural sentence pauses (<= max_internal_sil_ms) intact
         and only collapse the glitchy long ones.

    Rewrites the on-disk WAV only if something actually changed.  Returns
    the resulting length in samples.
    """
    import numpy as _np
    a = audio if hasattr(audio, "shape") else _np.asarray(audio, dtype=_np.float32)
    if a.ndim > 1:
        a = a[:, 0]
    a = a.astype(_np.float32, copy=True)
    if a.size == 0:
        return 0

    win = max(1, int(sr * 0.020))
    n_full = (a.size // win) * win
    if n_full < win:
        return a.size

    rms = _np.sqrt(_np.mean(a[:n_full].reshape(-1, win).astype(_np.float64) ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * _np.log10(rms + 1e-12)

    # ---- 1. head trim — remove a leading "ahh"/breath artifact ----
    # F5 sometimes emits a brief, QUIET vocalization ("ahh"/breath) then a
    # gap BEFORE the first real word — audible at every chunk start.  We
    # only strip it when its signature is unmistakable, so we NEVER clip a
    # real first word:
    #   (a) the initial voiced run is short  (<= 90 ms),
    #   (b) it is quiet — peak < -28 dBFS    (real words are louder),
    #   (c) it is followed by a >= 40 ms gap, and
    #   (d) there is real speech after that gap.
    # Otherwise we fall back to a plain leading-silence trim.
    voiced_mask = rms_db > head_db
    fv = _np.where(voiced_mask)[0]
    if fv.size == 0:
        return a.size
    s0 = int(fv[0])
    # The artifact is quieter than the chunk's REAL speech.  Use a
    # threshold relative to this chunk's own speech level (70th pct of
    # voiced frames) so it adapts to loud and quiet scripts alike — an
    # absolute dB cutoff misses louder "ahh"s in some texts.
    sp = rms_db[voiced_mask]
    typ = float(_np.percentile(sp, 70)) if sp.size else -28.0
    ahh_ceil = typ - 8.0          # >= 8 dB below real speech = artifact
    # walk the initial voiced run until a >= 2-frame (40 ms) gap
    j = s0
    gap_run = 0
    while j < len(voiced_mask):
        if voiced_mask[j]:
            gap_run = 0
        else:
            gap_run += 1
            if gap_run >= 2:
                break
        j += 1
    run_end = j - gap_run
    run_len_ms = (run_end - s0 + 1) * (win * 1000.0 / sr)
    run_peak = float(_np.max(rms_db[s0:run_end + 1])) if run_end >= s0 else -120.0
    after = _np.where(voiced_mask[j:])[0]
    # it's the artifact only if: short run, quiet vs real speech, a gap
    # after it, and real speech following.  Real first words are at full
    # speech level (>= ahh_ceil) so they're never trimmed.
    if (run_len_ms <= 110.0 and run_peak < ahh_ceil
            and gap_run >= 2 and after.size > 0):
        start_frame = j + int(after[0])         # skip the artifact + gap
    else:
        start_frame = s0                        # keep the real onset
    start = max(0, start_frame - 1) * win       # ~20 ms lead

    voiced_tail = _np.where(rms_db > tail_db)[0]
    if voiced_tail.size == 0:
        return a.size
    end = min((int(voiced_tail[-1]) + 1) * win, a.size)
    if end <= start:
        return a.size
    core = a[start:end]

    # ---- 2. compress internal silences ----
    # Re-frame the trimmed core and find silence runs.
    cn_full = (core.size // win) * win
    keep_pause = int(sr * keep_pause_ms / 1000.0)
    max_sil = int(sr * max_internal_sil_ms / 1000.0)
    rebuilt = core
    if cn_full >= win:
        crms = _np.sqrt(_np.mean(core[:cn_full].reshape(-1, win).astype(_np.float64) ** 2, axis=1) + 1e-12)
        cdb = 20.0 * _np.log10(crms + 1e-12)
        is_sil = cdb < sil_db
        # build list of (start_sample, end_sample, is_silence) segments
        segs = []
        k = 0
        while k < len(is_sil):
            j = k
            while j < len(is_sil) and is_sil[j] == is_sil[k]:
                j += 1
            segs.append((k * win, j * win, bool(is_sil[k])))
            k = j
        # any internal silence longer than max_sil → shrink to keep_pause
        if any(sil and (e - s) > max_sil for s, e, sil in segs):
            parts = []
            for idx, (s, e, sil) in enumerate(segs):
                seg = core[s:e]
                if sil and (e - s) > max_sil:
                    # leading/trailing silence inside core shouldn't
                    # happen after head/tail trim, but guard anyway
                    seg = seg[:keep_pause]
                parts.append(seg)
            rebuilt = _np.concatenate(parts) if parts else core

    if rebuilt is core:
        cut_total_ms = (start + (a.size - end)) * 1000 / sr
        # rewrite if we trimmed something OR we need to append a pad
        if cut_total_ms < min_trim_ms and trailing_pad_ms <= 0:
            return a.size                       # nothing worth rewriting
        out = core.copy()
    else:
        out = rebuilt.copy()

    fade = max(1, int(sr * fade_ms / 1000.0))
    if out.size > 2 * fade:
        ramp = _np.linspace(0.0, 1.0, fade, dtype=_np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]

    # graceful end-of-utterance taper: a longer cosine fade on the tail
    # so the last chunk doesn't cut off abruptly
    if tail_fade_ms > 0:
        tf = min(out.size, max(1, int(sr * tail_fade_ms / 1000.0)))
        cos_ramp = (0.5 * (1.0 + _np.cos(
            _np.linspace(0.0, _np.pi, tf, dtype=_np.float32))))
        out[-tf:] *= cos_ramp

    # append the trailing pause AFTER the fade-out so the chunk ends at
    # silence and the next chunk starts cleanly after a natural beat
    if trailing_pad_ms > 0:
        pad = _np.zeros(int(sr * trailing_pad_ms / 1000.0), dtype=out.dtype)
        out = _np.concatenate([out, pad])

    sf.write(str(wav_path), out, sr)
    return out.size


# ---------- sentence-level streaming ----------

import re

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_by_fullstop(text: str) -> list[str]:
    """One chunk per sentence — split only on . ! ? boundaries.

    Natural sentence chunking.  Each chunk is a whole sentence, so the
    prosody is never cut mid-thought.  Downside vs the smart splitter:
    a long sentence becomes one big chunk that renders slowly, so there
    can be a gap before it during streaming.  This is the 'per fullstop'
    option exposed in the UI.
    """
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]


def split_sentences(
    text: str,
    target_chars: int = 46,
    max_chars: int = 62,
    first_chunk_min: int = 44,
    first_chunk_max: int = 54,
    min_chunk_chars: int = 36,
) -> list[str]:
    """
    Split text into chunks for smooth streaming playback.

    The constraint that drives everything: on the Jetson, a chunk takes
    ~0.9× its own audio-duration to render (nfe=16).  So for playback to
    be gap-free, each chunk's render time must be hidden behind the
    PREVIOUS chunk's playback.  That means two things:

    1. **First chunk: small but not tiny.**  We want the first audio in
       ~2 s, so the first chunk targets `first_chunk_min..first_chunk_max`
       chars (~1.2-1.8 s of audio, renders ~1.8-2.2 s at nfe=12).  It
       must NOT be a bare "Hello." (0.3 s audio) — that finishes playing
       seconds before chunk 2 is ready and the listener hears a gap.  A
       too-short leading sentence is merged forward into the next.
    2. **Later chunks: ~target_chars.**  Big enough that their own
       playback (~1.5 s) masks the next chunk's render, small enough
       that no single chunk blocks the stream.
    3. **Cuts on natural pauses.**  Sentence (.!?) first, then
       comma/semicolon, then word boundary.
    """
    text = (text or "").strip()
    if not text:
        return []

    def _wrap_words(s: str, limit: int) -> list[str]:
        """Split `s` into BALANCED pieces of roughly `limit` chars on word
        boundaries.  Greedy wrapping leaves an uneven tail (e.g. a 90-char
        piece next to an 18-char one); balanced wrapping makes every piece
        ~equal so no single chunk is oversized — an oversized chunk renders
        slower than the previous chunk plays and intermittently underruns
        the streaming playback buffer."""
        words = s.split()
        if not words:
            return []
        total = sum(len(w) for w in words) + (len(words) - 1)
        n = max(1, round(total / max(1, limit)))
        if n <= 1:
            return [s]
        per = total / n            # fair share per piece
        out_w: list[str] = []
        buf: list[str] = []
        buflen = 0
        for w in words:
            add = len(w) + (1 if buf else 0)
            # close the current piece if it has reached its fair share and
            # we still owe more pieces (keep at least 1 word per piece)
            if buf and (buflen + add) > per * 1.12 and len(out_w) < n - 1:
                out_w.append(" ".join(buf))
                buf = [w]; buflen = len(w)
            else:
                buf.append(w); buflen += add
        if buf:
            out_w.append(" ".join(buf))
        return out_w

    # 1. coarse-split on sentence punctuation
    raw = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]

    # 2. break any oversized sentence on commas/semicolons, then on word
    #    boundaries — so even a long comma-less sentence becomes several
    #    small chunks (a single big chunk renders slower than the prior
    #    chunk plays → a multi-second gap in streaming playback).
    pieces: list[str] = []
    for sent in raw:
        if len(sent) <= target_chars:
            pieces.append(sent)
            continue
        sub = [s.strip() for s in re.split(r'(?<=[,;:])\s+', sent) if s.strip()]
        buf = ""
        for s in sub:
            # a comma-clause that's itself too long → word-wrap it so no
            # single piece exceeds target_chars
            parts = [s] if len(s) <= target_chars else _wrap_words(s, target_chars)
            for part in parts:
                if buf and len(buf) + 1 + len(part) <= target_chars:
                    buf = (buf + " " + part).strip()
                else:
                    if buf:
                        pieces.append(buf)
                    buf = part
        if buf:
            pieces.append(buf)

    # 3. greedy merge with a per-position cap.  chunk[0] caps at
    #    first_chunk_max; the rest at target_chars.
    out: list[str] = []
    buf = ""
    for p in pieces:
        cap = first_chunk_max if not out else target_chars
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= cap:
            buf = buf + " " + p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)

    # 4. if chunk[0] is too SHORT (e.g. a bare "Hello."), pull words
    #    from chunk[1] forward until it reaches first_chunk_min, cutting
    #    chunk[1] on a word boundary.  A 0.3 s first chunk followed by a
    #    4 s render gap is the worst failure mode; a slightly bigger
    #    first chunk that starts ~0.3 s later is far better UX.
    if len(out) >= 2 and len(out[0]) < first_chunk_min:
        head, nxt = out[0], out[1]
        want = first_chunk_min - len(head)
        # take whole words from nxt until head is long enough
        words = nxt.split()
        take = []
        taken_len = 0
        for w in words:
            take.append(w)
            taken_len += len(w) + 1
            if len(head) + 1 + taken_len >= first_chunk_min:
                break
        moved = " ".join(take)
        remainder = nxt[len(moved):].strip()
        new_head = (head + " " + moved).strip()
        if remainder:
            out = [new_head, remainder] + out[2:]
        else:
            out = [new_head] + out[2:]

    # 5. if chunk[0] is too LONG (one long sentence with no early
    #    break), split it under first_chunk_max at the nearest natural
    #    break: comma/semicolon → em-dash → word boundary.  Keep a
    #    single short sentence intact if it's only modestly over.
    INTACT_MAX = first_chunk_max + 20
    if (out and len(out[0]) > first_chunk_max
            and not (len(out) == 1 and len(out[0]) <= INTACT_MAX)):
        first = out[0]
        cut_at = -1
        for sep in (", ", "; ", " — ", " - "):
            idx = first.rfind(sep, 0, first_chunk_max)
            if idx >= first_chunk_min:
                cut_at = idx + len(sep)
                break
        if cut_at < 0:
            idx = first.rfind(" ", 0, first_chunk_max)
            if idx >= first_chunk_min:
                cut_at = idx + 1
        if cut_at > 0:
            head = first[:cut_at].strip()
            tail = first[cut_at:].strip()
            # only split off a SUBSTANTIAL tail.  A tiny orphan tail
            # (e.g. "physicist") can't merge cleanly into the next body
            # chunk without making it oversized, so it survives as a
            # buffer-draining short chunk.  If the tail would be tiny,
            # leave the first chunk whole (slightly higher TTfA, but it
            # banks a big initial playback buffer → robust gaplessness).
            if head and tail and len(tail) >= 18:
                out = [head, tail] + out[1:]

    # 6. CONSOLIDATE the BODY chunks — every chunk after the first must
    #    be >= min_chunk_chars.  A short body chunk's audio drains the
    #    playback buffer faster than the next chunk can render (~2.4-2.8s
    #    wall on the Jetson), so a too-short chunk → intermittent gap.
    #    The FIRST chunk is exempt: it renders at the lower first-chunk
    #    nfe (faster) AND it builds the initial buffer, so it can stay
    #    small for a fast time-to-first-audio.  Absorb short body chunks
    #    forward; fold a short trailing chunk back into the previous one.
    if len(out) >= 3:
        first = out[0]
        merged: list[str] = []
        for piece in out[1:]:
            if merged and len(merged[-1]) < min_chunk_chars:
                combined = (merged[-1] + " " + piece).strip()
                if len(combined) <= max_chars:
                    # absorb the short chunk forward
                    merged[-1] = combined
                else:
                    # too big for one chunk, too small as two uneven ones
                    # → re-split the pair into BALANCED halves so neither
                    # is short (drains buffer) nor oversized (renders slow)
                    halves = _wrap_words(combined, max(1, len(combined) // 2))
                    merged[-1] = halves[0]
                    merged.extend(halves[1:])
            else:
                merged.append(piece)
        # NOTE: do NOT fold a short trailing chunk back into the previous
        # one — a short LAST chunk is harmless (nothing follows it to
        # starve), whereas folding it makes the last chunk oversized,
        # which renders slower than its predecessor plays → a gap BEFORE
        # it.  Leave the last chunk as-is.
        out = [first] + merged
    return out


FIRST_CHUNK_NFE = 12   # lower nfe for the very first chunk only —
                       # 1.7-2.0 s render time at ~5-8 words instead of
                       # ~2.5-3 s at the default nfe=16, with quality
                       # drop limited to the first ~1.5 s of audio

# Natural pause appended after a chunk (except the last) so consecutive
# chunks don't run together when the client butt-joins them.
SENTENCE_PAUSE_MS = 20     # minimal beat between chunks
CLAUSE_PAUSE_MS   = 15
END_FADE_MS       = 18     # tiny declick on the very last chunk — short
                           # enough it never swallows the final word
LAST_CHUNK_PAD_MS = 70     # silence after the last word so its natural
                           # release has room to finish (no abrupt stop)
MID_CHUNK_FADE_MS = 28     # smooth tail fade on every non-last chunk so
                           # it eases into the inter-chunk pause and masks
                           # any residual low-level tail artifact


def speak_stream(
    voice_key: str,
    text: str,
    *,
    speed: float = DEFAULT_SPEED,
    base_speaker: str = "F5-Base",
    tau: float = DEFAULT_CFG,
    nfe_step: int = DEFAULT_NFE,
    first_chunk_nfe: int = FIRST_CHUNK_NFE,
    chunk_mode: str = "smart",
):
    """
    Generator that yields one SynthResult per chunk.

    The web layer streams these out as SSE events so the browser can
    start playback the moment the first chunk lands while the rest
    render in the background.

    chunk_mode:
      "smart"    — size-optimized chunks tuned so each renders before
                   the previous finishes playing → gapless, ~2.6s start.
      "sentence" — one chunk per sentence (split on . ! ?).  Natural
                   sentence prosody, but a long sentence renders slowly
                   so there can be a gap before it.

    First chunk is rendered at `first_chunk_nfe` (default 12) for a
    faster start; remaining chunks use `nfe_step` (default 16).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")

    if chunk_mode == "sentence":
        chunks = split_by_fullstop(text)
    else:
        chunks = split_sentences(text)
    log.info("streaming %d chunks for voice=%s (mode=%s first_nfe=%d rest_nfe=%d)",
             len(chunks), voice_key, chunk_mode, first_chunk_nfe, nfe_step)

    n = len(chunks)
    for i, chunk in enumerate(chunks):
        # Render EVERY streaming chunk at the fast nfe.  This is required,
        # not just preferred: to keep time-to-first-audio under 2s the
        # first chunk must be small, and for gap-free playback every
        # later chunk must render faster than the previous one plays.  At
        # nfe=16 a chunk only becomes self-sustaining at ~58 chars, which
        # would push the first chunk (and TTfA) over budget; at the fast
        # nfe ~36-char chunks already keep up.  Using one nfe throughout
        # also removes the quality discontinuity between chunk 0 and the
        # rest.  (`nfe_step` is ignored in smart streaming for this
        # reason; the non-streaming /api/speak path still honours it.)
        nfe_for_chunk = first_chunk_nfe
        # Add a natural beat after this chunk unless it's the last one.
        # A sentence-final chunk (ends with . ! ?) gets a fuller pause;
        # a mid-sentence clause split (comma) gets a short one.  Without
        # this the next chunk butt-joins instantly and sentences run
        # together unnaturally.
        is_last = (i == n - 1)
        if is_last:
            pad_ms = LAST_CHUNK_PAD_MS   # room for the final word to finish
        elif chunk.rstrip().endswith((".", "!", "?", "…")):
            pad_ms = SENTENCE_PAUSE_MS
        else:
            pad_ms = CLAUSE_PAUSE_MS
        result = speak(
            voice_key, chunk,
            speed=speed, base_speaker=base_speaker,
            tau=tau, nfe_step=nfe_for_chunk,
            output=OUT_AUDIO / f"{voice_key}-{int(time.time()*1000)}-{i:02d}.wav",
            trailing_pad_ms=pad_ms,
            tail_fade_ms=END_FADE_MS if is_last else MID_CHUNK_FADE_MS,
        )
        log.info("  chunk %d/%d nfe=%d pad=%dms: %r (%.2fs audio, rtf %.2f)",
                 i + 1, n, nfe_for_chunk, pad_ms, chunk[:48],
                 result.seconds, result.rtf)
        yield result


def available_base_speakers() -> list[str]:
    """F5-TTS doesn't use base speakers — return a sentinel."""
    return ["F5-Base"]

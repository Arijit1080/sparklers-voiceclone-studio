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
    # F5TTS.infer() returns (wav, sr, spectrogram) AND writes the WAV.
    # Capturing the return saves a second file open for duration calc.
    wav, sr, _ = f5.infer(
        ref_file=str(ref_wav),
        ref_text=ref_text,
        gen_text=text,
        file_wave=str(output),
        speed=speed,
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
    # actually crops > ~30 ms).
    n_out = _hard_trim_silence_inplace(wav, sr, output)
    elapsed = time.monotonic() - t0
    seconds = n_out / float(sr)

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
    tail_db: float = -42.0,
    fade_ms: int = 15,
    min_trim_ms: int = 30,
) -> int:
    """Trim leading/trailing silence from the in-memory waveform F5-TTS
    just returned, then overwrite the on-disk WAV ONLY if the crop is
    larger than `min_trim_ms` total. Returns the (possibly cropped) length.
    """
    import numpy as _np
    a = audio if hasattr(audio, "shape") else _np.asarray(audio, dtype=_np.float32)
    if a.ndim > 1:
        a = a[:, 0]
    if a.size == 0:
        return 0

    win = max(1, int(sr * 0.020))
    n_full = (a.size // win) * win
    if n_full < win:
        return a.size

    rms = _np.sqrt(_np.mean(a[:n_full].reshape(-1, win).astype(_np.float64) ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * _np.log10(rms + 1e-12)
    voiced = _np.where(rms_db > head_db)[0]
    if voiced.size == 0:
        return a.size

    start = int(voiced[0]) * win
    voiced_tail = _np.where(rms_db > tail_db)[0]
    end = min((int(voiced_tail[-1]) + 1) * win, a.size)

    cut_total_ms = (start + (a.size - end)) * 1000 / sr
    if cut_total_ms < min_trim_ms:
        return a.size                       # not worth a rewrite

    cropped = a[start:end].copy()
    fade = max(1, int(sr * fade_ms / 1000.0))
    if cropped.size > 2 * fade:
        ramp = _np.linspace(0.0, 1.0, fade, dtype=_np.float32)
        cropped[:fade] *= ramp
        cropped[-fade:] *= ramp[::-1]

    sf.write(str(wav_path), cropped, sr)
    return cropped.size


# ---------- sentence-level streaming ----------

import re

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(
    text: str,
    target_chars: int = 220,
    max_chars: int = 320,
    first_chunk_max: int = 30,
) -> list[str]:
    """
    Split text into chunks that are roughly `target_chars` long and never
    longer than `max_chars`. Four goals:

    1. **Tiny FIRST chunk for fast time-to-first-audio.** The first
       chunk is capped at `first_chunk_max` (~60 chars = roughly 1.5 s
       of audio) so playback can start ~1.5-2 s after the user hits
       Speak.  Subsequent chunks are full-sized.
    2. **Avoid tiny later chunks.** "Hello." rendered alone is a 0.5 s
       audio clip the server takes ~0.8 s to produce — playback ends
       before the next chunk is ready and the listener hears a gap.
       Greedy merging keeps each later chunk long enough to mask the
       next chunk's render time.
    3. **Avoid giant chunks.** A 30-second chunk blocks playback start
       and feels like the streaming isn't streaming.
    4. **Keep cuts on natural pause points.** Always try sentence-level
       (.!?), then comma/semicolon, then word boundary as a last
       resort if a single sentence is longer than first_chunk_max.
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

    # 3. greedily merge — but cap chunk[0] at first_chunk_max so the
    #    first audio fires fast.
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

    # 4. if the first chunk is STILL longer than first_chunk_max (one
    #    sentence with no comma is bigger than the cap), split it at
    #    the nearest natural break under the cap, falling through:
    #    comma/semicolon → em-dash → word boundary.  We require at
    #    least 5 chars before the cut so we never produce a "Hi"-only
    #    first chunk.  But if the input is a single short sentence
    #    that the user could reasonably hear as one phrase
    #    (~ first_chunk_max + 25 chars total), keep it intact — the
    #    tail-chunk would be too tiny to render efficiently AND
    #    leave a long gap before chunk[1] arrives.
    INTACT_MAX = first_chunk_max + 25
    if (out and len(out[0]) > first_chunk_max
            and not (len(out) == 1 and len(out[0]) <= INTACT_MAX)):
        first = out[0]
        cut_at = -1
        for sep in (", ", "; ", " — ", " - "):
            idx = first.rfind(sep, 0, first_chunk_max)
            if idx >= 5:
                cut_at = idx + len(sep)
                break
        if cut_at < 0:
            idx = first.rfind(" ", 0, first_chunk_max)
            if idx >= 5:
                cut_at = idx + 1
        if cut_at > 0:
            head = first[:cut_at].strip()
            tail = first[cut_at:].strip()
            # Refuse to create a tail tinier than 12 chars — playback
            # gap is worse than a slightly slower TTfA.
            if head and tail and len(tail) >= 12:
                out = [head, tail] + out[1:]
    return out


FIRST_CHUNK_NFE = 12   # lower nfe for the very first chunk only —
                       # 1.7-2.0 s render time at ~5-8 words instead of
                       # ~2.5-3 s at the default nfe=16, with quality
                       # drop limited to the first ~1.5 s of audio


def speak_stream(
    voice_key: str,
    text: str,
    *,
    speed: float = DEFAULT_SPEED,
    base_speaker: str = "F5-Base",
    tau: float = DEFAULT_CFG,
    nfe_step: int = DEFAULT_NFE,
    first_chunk_nfe: int = FIRST_CHUNK_NFE,
):
    """
    Generator that yields one SynthResult per sentence-sized chunk.

    The web layer streams these out as SSE events so the browser can
    start playback the moment the first chunk lands while the rest
    render in the background.

    First chunk is rendered at `first_chunk_nfe` (default 12) so
    time-to-first-audio stays around 1.7-2.0 s on a Jetson Orin Nano.
    Remaining chunks use `nfe_step` (default 16) for full quality.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")

    chunks = split_sentences(text)
    log.info("streaming %d chunks for voice=%s (first_nfe=%d, rest_nfe=%d)",
             len(chunks), voice_key, first_chunk_nfe, nfe_step)

    for i, chunk in enumerate(chunks):
        nfe_for_chunk = first_chunk_nfe if i == 0 else nfe_step
        result = speak(
            voice_key, chunk,
            speed=speed, base_speaker=base_speaker,
            tau=tau, nfe_step=nfe_for_chunk,
            output=OUT_AUDIO / f"{voice_key}-{int(time.time()*1000)}-{i:02d}.wav",
        )
        log.info("  chunk %d/%d nfe=%d: %r (%.2fs audio, rtf %.2f)",
                 i + 1, len(chunks), nfe_for_chunk, chunk[:48],
                 result.seconds, result.rtf)
        yield result


def available_base_speakers() -> list[str]:
    """F5-TTS doesn't use base speakers — return a sentinel."""
    return ["F5-Base"]

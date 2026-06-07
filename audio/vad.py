"""
Speech-activity trimming for enrollment clips.

Given a raw mic recording of ~30 s, we want to keep just the parts the
user was actually speaking — silence and intake breaths confuse
OpenVoice's reference encoder. We use webrtcvad (rock solid, no model)
to make a per-frame voiced/unvoiced decision, then return a cleaned
clip with silences collapsed but the natural rhythm preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import webrtcvad


@dataclass
class VadResult:
    """Output of `trim_for_enrollment` — useful for debug overlays."""
    samples: np.ndarray
    n_voiced_frames: int
    n_total_frames: int
    rms_dbfs: float

    @property
    def voiced_ratio(self) -> float:
        return self.n_voiced_frames / max(1, self.n_total_frames)


# webrtcvad only accepts 10/20/30 ms frames at 8k/16k/32k/48k.
FRAME_MS = 30


def _frames(samples: np.ndarray, sr: int, ms: int = FRAME_MS) -> Iterable[bytes]:
    n = int(sr * ms / 1000)
    for i in range(0, len(samples) - n + 1, n):
        yield samples[i:i + n].astype(np.int16).tobytes()


def trim_for_enrollment(
    samples: np.ndarray,
    sr: int = 16000,
    *,
    aggressiveness: int = 2,
    pad_ms: int = 90,
    keep_silence_ms: int = 200,
) -> VadResult:
    """
    Drop leading/trailing silence and collapse long internal pauses.

    Parameters
    ----------
    aggressiveness  webrtcvad mode 0–3 (3 = strictest, drops more)
    pad_ms          padding added before each voiced run (prevents chops)
    keep_silence_ms max silence kept between two voiced runs (else collapsed)
    """
    if samples.ndim > 1:
        samples = samples.reshape(-1)

    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = int(sr * FRAME_MS / 1000)
    pad_frames    = max(1, pad_ms // FRAME_MS)
    keep_frames   = max(0, keep_silence_ms // FRAME_MS)

    decisions: list[bool] = []
    for buf in _frames(samples, sr):
        try:
            decisions.append(vad.is_speech(buf, sr))
        except Exception:
            decisions.append(False)

    # Identify voiced runs, then re-emit them with `pad_frames` of context
    # on either side and at most `keep_frames` of silence joining adjacent
    # runs. Anything else (long pauses, intake breath, room noise) is dropped.
    keep = [False] * len(decisions)
    run_start: int | None = None
    for i, v in enumerate(decisions):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            for j in range(max(0, run_start - pad_frames),
                           min(len(decisions), i + pad_frames)):
                keep[j] = True
            run_start = None
    if run_start is not None:
        for j in range(max(0, run_start - pad_frames), len(decisions)):
            keep[j] = True

    # Collapse: keep True spans intact; between them, allow `keep_frames` of False
    out_frames: list[bool] = []
    silence_run = 0
    seen_voiced = False
    for k in keep:
        if k:
            out_frames.append(True)
            silence_run = 0
            seen_voiced = True
        else:
            if not seen_voiced:
                continue                # drop leading silence entirely
            if silence_run < keep_frames:
                out_frames.append(False)
            silence_run += 1
    # drop trailing False
    while out_frames and out_frames[-1] is False:
        out_frames.pop()

    pieces: list[np.ndarray] = []
    for idx, k in enumerate(out_frames):
        start = idx * frame_samples
        end   = start + frame_samples
        if start >= len(samples):
            break
        if k:
            pieces.append(samples[start:end])
    cleaned = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)

    rms = 20 * np.log10(np.sqrt(np.mean(
        (cleaned.astype(np.float32) / 32768.0) ** 2 + 1e-12)) + 1e-12) if cleaned.size else -120.0

    return VadResult(
        samples=cleaned,
        n_voiced_frames=sum(1 for k in keep if k),
        n_total_frames=len(decisions),
        rms_dbfs=float(rms),
    )

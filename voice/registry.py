"""
Tiny on-disk voice registry.

Each enrolled voice = one .pth (tone-color embedding) + one sidecar
.json describing display name, source WAV, enrollment time, and
average reference RMS. The .pth is what OpenVoice's
ToneColorConverter needs at synth time.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .paths import VOICES_DIR, ENROLL_AUDIO


_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(name: str) -> str:
    s = name.strip().lower().replace(" ", "_").replace("-", "_")
    s = _SLUG_RE.sub("", s)
    return s.strip("_") or "voice"


@dataclass
class VoiceMeta:
    key: str
    display_name: str
    enrolled_at: float            # epoch seconds
    seconds: float                # length of trimmed reference
    rms_dbfs: float
    voiced_ratio: float
    source_wav: str               # filename inside ENROLL_AUDIO/
    ref_text: str = ""            # whisper-tiny transcript of the reference

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _meta_path(key: str) -> Path:
    return VOICES_DIR / f"{key}.json"


def embedding_path(key: str) -> Path:
    return VOICES_DIR / f"{key}.pth"


def enroll_wav_path(key: str) -> Path:
    return ENROLL_AUDIO / f"{key}.wav"


def save_meta(meta: VoiceMeta) -> None:
    _meta_path(meta.key).write_text(meta.to_json())


def load_meta(key: str) -> VoiceMeta | None:
    p = _meta_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        # Backward-compat with v1 (OpenVoice) meta that lacked ref_text
        data.setdefault("ref_text", "")
        return VoiceMeta(**data)
    except Exception:
        return None


def list_voices() -> list[VoiceMeta]:
    out: list[VoiceMeta] = []
    for p in sorted(VOICES_DIR.glob("*.json")):
        m = load_meta(p.stem)
        if m and embedding_path(m.key).exists():
            out.append(m)
    return out


def delete_voice(key: str) -> bool:
    """Remove embedding + meta + source WAV. Returns True if something was removed."""
    removed = False
    for p in (embedding_path(key), _meta_path(key), enroll_wav_path(key)):
        if p.exists():
            try:
                p.unlink()
                removed = True
            except OSError:
                pass
    return removed


def new_voice_key(display_name: str) -> str:
    """Build a unique slug, suffixing -2, -3… if needed."""
    base = slugify(display_name)
    if not embedding_path(base).exists() and not _meta_path(base).exists():
        return base
    n = 2
    while True:
        candidate = f"{base}_{n}"
        if not embedding_path(candidate).exists() and not _meta_path(candidate).exists():
            return candidate
        n += 1


def fresh_timestamp() -> float:
    return time.time()

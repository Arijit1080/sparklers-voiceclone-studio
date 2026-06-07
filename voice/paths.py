"""Filesystem layout for the voiceclone studio."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = Path(os.environ.get("SPARKLERS_DATA_DIR",        PROJECT_ROOT / "data"))
MODELS_DIR      = Path(os.environ.get("SPARKLERS_MODELS_DIR",      PROJECT_ROOT / "models"))
VOICES_DIR      = MODELS_DIR / "voices"            # one .pth per enrolled voice
ENROLL_AUDIO    = DATA_DIR   / "enrollments"       # the raw 30s WAV per voice
OUT_AUDIO       = DATA_DIR   / "out"               # synthesized audio cache
OPENVOICE_DIR   = Path(os.environ.get("SPARKLERS_OPENVOICE_DIR",   MODELS_DIR / "openvoice_v2"))
OPENVOICE_CKPTS = OPENVOICE_DIR / "checkpoints_v2"

for p in (DATA_DIR, MODELS_DIR, VOICES_DIR, ENROLL_AUDIO, OUT_AUDIO):
    p.mkdir(parents=True, exist_ok=True)

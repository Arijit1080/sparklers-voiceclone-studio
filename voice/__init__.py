"""Voice cloning pipeline — OpenVoice v2 + MeloTTS (English)."""

from .registry import (
    VoiceMeta,
    list_voices,
    delete_voice,
    load_meta,
    new_voice_key,
    embedding_path,
    enroll_wav_path,
)
from .enroll import enroll_from_mic, enroll_from_wav, EnrollmentResult
from .synth import speak, speak_stream, warmup, SynthResult, available_base_speakers, split_sentences

__all__ = [
    "VoiceMeta",
    "list_voices",
    "delete_voice",
    "load_meta",
    "new_voice_key",
    "embedding_path",
    "enroll_wav_path",
    "enroll_from_mic",
    "enroll_from_wav",
    "EnrollmentResult",
    "speak",
    "speak_stream",
    "warmup",
    "SynthResult",
    "available_base_speakers",
    "split_sentences",
]

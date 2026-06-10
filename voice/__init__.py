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
from .enroll import (
    enroll_from_mic, enroll_from_wav, EnrollmentResult,
    set_reference_length, reduce_reference, REF_LENGTH_CHOICES,
)
from .synth import speak, speak_stream, warmup, SynthResult, available_base_speakers, split_sentences, split_by_fullstop

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
    "set_reference_length",
    "reduce_reference",
    "REF_LENGTH_CHOICES",
    "EnrollmentResult",
    "speak",
    "speak_stream",
    "warmup",
    "SynthResult",
    "available_base_speakers",
    "split_sentences",
    "split_by_fullstop",
]

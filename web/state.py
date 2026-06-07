"""
Service singleton: holds OpenVoice/MeloTTS, enrollment job state, and
the broadcast queues that drive the SSE stream.

Three job slots — at most one of each runs at a time:
  - enrollment (the recorder + SE extractor)
  - synthesis  (MeloTTS + tone-color convert, sub-second so not really a job)
  - warmup     (load both models on the GPU)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import voice
from voice.paths import OUT_AUDIO

log = logging.getLogger("web.state")


@dataclass
class JobState:
    job_id: str
    kind: str                 # "enroll" | "synth" | "warmup"
    started_at: float
    state: str = "running"    # "running" | "done" | "error"
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)


class VoiceCloneService:
    """One per process. Owns the model handles and current job state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy_with: str | None = None
        self.current_job: JobState | None = None
        self._listeners: set[asyncio.Queue] = set()
        self.last_meter: tuple[float, float] = (0.0, -120.0)  # (t, db)
        self.models_loaded = False

    # ---------- subscriber bookkeeping ----------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners.discard(q)

    def _publish(self, event: str, payload: dict[str, Any]) -> None:
        """Schedule an SSE broadcast on the running event loop."""
        loop = getattr(self, "_loop", None)
        if loop is None:
            return
        msg = {"event": event, "data": payload, "ts": time.time()}
        for q in list(self._listeners):
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ---------- busy state ----------

    def busy_with(self) -> str | None:
        with self._lock:
            return self._busy_with

    def _claim(self, kind: str) -> JobState:
        with self._lock:
            if self._busy_with is not None:
                raise RuntimeError(f"service is busy: {self._busy_with}")
            self._busy_with = kind
            job = JobState(
                job_id=uuid.uuid4().hex[:8],
                kind=kind,
                started_at=time.time(),
            )
            self.current_job = job
        return job

    def _release(self) -> None:
        with self._lock:
            self._busy_with = None

    # ---------- enrollment ----------

    def start_enrollment(self, display_name: str, seconds: float = 30.0) -> JobState:
        if not display_name.strip():
            raise ValueError("voice name is required")
        job = self._claim("enroll")
        threading.Thread(
            target=self._enroll_worker,
            args=(job, display_name, seconds),
            daemon=True,
            name=f"enroll-{job.job_id}",
        ).start()
        return job

    def start_enrollment_from_wav(self, display_name: str, wav_path: Path) -> JobState:
        if not display_name.strip():
            raise ValueError("voice name is required")
        if not wav_path.exists():
            raise FileNotFoundError(str(wav_path))
        job = self._claim("enroll")
        threading.Thread(
            target=self._enroll_from_wav_worker,
            args=(job, display_name, wav_path),
            daemon=True,
            name=f"enroll-upload-{job.job_id}",
        ).start()
        return job

    def _enroll_from_wav_worker(self, job: JobState, display_name: str,
                                 wav_path: Path) -> None:
        try:
            key = voice.new_voice_key(display_name)
            log.info("[%s] enroll-from-wav name=%s key=%s wav=%s",
                     job.job_id, display_name, key, wav_path)
            result = voice.enroll_from_wav(
                key, display_name, wav_path,
                on_progress=lambda m, r: self._on_progress(job, m, r),
            )
            job.state = "done"
            job.progress = 1.0
            job.result = {
                "key": key,
                "display_name": display_name,
                "seconds_kept": result.seconds_kept,
                "rms_dbfs": result.meta.rms_dbfs,
            }
            self._publish("done", {"job_id": job.job_id, "result": job.result})
        except Exception as e:
            log.exception("enroll-from-wav failed")
            job.state = "error"
            job.error = str(e)
            self._publish("error", {"job_id": job.job_id, "error": str(e)})
        finally:
            self._release()

    def _on_progress(self, job: JobState, msg: str, ratio: float) -> None:
        job.message = msg
        job.progress = float(ratio)
        self._publish("progress", {
            "job_id": job.job_id, "kind": job.kind,
            "state": job.state, "message": msg, "progress": ratio,
        })

    def _on_meter(self, job: JobState, t: float, db: float) -> None:
        self.last_meter = (t, db)
        self._publish("meter", {
            "job_id": job.job_id, "t": round(t, 2), "db": round(db, 1),
        })

    def _enroll_worker(self, job: JobState, display_name: str, seconds: float) -> None:
        try:
            key = voice.new_voice_key(display_name)
            log.info("[%s] enroll start name=%s key=%s seconds=%.1f",
                     job.job_id, display_name, key, seconds)
            result = voice.enroll_from_mic(
                key, display_name,
                seconds=seconds,
                play_cues=True,
                on_progress=lambda m, r: self._on_progress(job, m, r),
                on_meter=lambda t, d: self._on_meter(job, t, d),
            )
            job.state = "done"
            job.progress = 1.0
            job.result = {
                "key": key,
                "display_name": display_name,
                "seconds_kept": result.seconds_kept,
                "rms_dbfs": result.meta.rms_dbfs,
            }
            self._publish("done", {"job_id": job.job_id, "result": job.result})
        except Exception as e:
            log.exception("enroll failed")
            job.state = "error"
            job.error = str(e)
            self._publish("error", {"job_id": job.job_id, "error": str(e)})
        finally:
            self._release()

    # ---------- synthesis ----------

    def speak(self, voice_key: str, text: str, *, speed: float = 1.0,
              base_speaker: str = "F5-Base", tau: float = 3.0,
              nfe_step: int = 32) -> dict[str, Any]:
        """Synchronous — finishes in <10 s on the Orin Nano (warm)."""
        with self._lock:
            if self._busy_with not in (None, "synth"):
                raise RuntimeError(f"service is busy: {self._busy_with}")
            self._busy_with = "synth"
        try:
            log.info("synth voice=%s tau=%.1f nfe=%d len=%d text=%r",
                     voice_key, tau, nfe_step, len(text), text[:40])
            res = voice.speak(voice_key, text, speed=speed,
                              base_speaker=base_speaker, tau=tau,
                              nfe_step=nfe_step)
            payload = {
                "voice_key": res.voice_key,
                "text": res.text,
                "wav_url": f"/audio/{res.wav_path.name}",
                "seconds": res.seconds,
                "elapsed": res.elapsed,
                "rtf": res.rtf,
                "base_speaker": res.base_speaker,
                "tau": res.tau,
            }
            self._publish("synth", payload)
            return payload
        finally:
            self._release()

    # ---------- warmup ----------

    def warmup_async(self) -> JobState:
        job = self._claim("warmup")
        threading.Thread(
            target=self._warmup_worker, args=(job,),
            daemon=True, name="warmup",
        ).start()
        return job

    def _warmup_worker(self, job: JobState) -> None:
        try:
            self._on_progress(job, "loading MeloTTS …", 0.10)
            voice.warmup()
            self.models_loaded = True
            job.state = "done"
            self._on_progress(job, "models loaded", 1.0)
            self._publish("done", {"job_id": job.job_id, "result": {"loaded": True}})
        except Exception as e:
            log.exception("warmup failed")
            job.state = "error"
            job.error = str(e)
            self._publish("error", {"job_id": job.job_id, "error": str(e)})
        finally:
            self._release()


# Module-level singleton — imported by web.app
SERVICE = VoiceCloneService()


def list_voices_dict() -> list[dict[str, Any]]:
    """Used by the home template and /api/voices."""
    return [
        {
            "key": v.key,
            "display_name": v.display_name,
            "enrolled_at": v.enrolled_at,
            "seconds": v.seconds,
            "rms_dbfs": v.rms_dbfs,
            "voiced_ratio": v.voiced_ratio,
            "ref_text": v.ref_text,
        }
        for v in voice.list_voices()
    ]

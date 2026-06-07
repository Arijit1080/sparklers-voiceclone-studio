"""
FastAPI app for Sparklers VoiceClone Studio.

Routes
------
GET  /                       home — voice library
GET  /enroll                 enroll a new voice (live mic)
GET  /speak                  type text → synth in chosen voice
GET  /dashboard              system monitor

POST /api/enroll/start       kick off live-mic enrollment
POST /api/speak              synth text → returns audio URL
POST /api/voice/delete       delete an enrolled voice
POST /api/warmup             preload MeloTTS + converter
GET  /api/voices             JSON list of enrolled voices
GET  /api/status             snapshot of service state

GET  /events                 SSE — progress, meter, sysstats, synth, error
GET  /audio/<name>           serve a synthesized WAV from data/out
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from voice.paths import OUT_AUDIO, ENROLL_AUDIO
from web.state import SERVICE, list_voices_dict
from web.sysmon import SystemMonitor

log = logging.getLogger("web.app")
logging.basicConfig(
    level=os.environ.get("SPARKLERS_LOG", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR    = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------- lifespan ----------

_sysmon: SystemMonitor | None = None
_last_sys: dict[str, Any] = {}


def _on_sysstats(stats: dict[str, Any]) -> None:
    _last_sys.update(stats)
    SERVICE._publish("sysstats", stats)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bind the asyncio event loop to the service so background threads
    # can publish events thread-safely.
    SERVICE.bind_loop(asyncio.get_running_loop())

    # Start tegrastats reader. Best-effort — missing binary just means
    # the dashboard shows an empty system card.
    global _sysmon
    _sysmon = SystemMonitor(on_update=_on_sysstats)
    started = _sysmon.start()
    log.info("sysmon started=%s", started)

    yield

    if _sysmon is not None:
        _sysmon.stop()


app = FastAPI(title="Sparklers VoiceClone Studio", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- pages ----------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request, "home.html",
        {"voices": list_voices_dict(), "active": "home"},
    )


@app.get("/enroll")
async def enroll_page(request: Request):
    return templates.TemplateResponse(
        request, "enroll.html", {"active": "enroll"},
    )


@app.get("/speak")
async def speak_page(request: Request):
    return templates.TemplateResponse(
        request, "speak.html",
        {"voices": list_voices_dict(), "active": "speak"},
    )


@app.get("/dashboard")
async def dashboard_page(request: Request):
    return templates.TemplateResponse(
        request, "dashboard.html",
        {"voices": list_voices_dict(), "active": "dashboard",
         "models_loaded": SERVICE.models_loaded},
    )


# ---------- API ----------

class EnrollReq(BaseModel):
    display_name: str = Field(min_length=1, max_length=64)
    seconds: float = Field(default=30.0, ge=3.0, le=120.0)


class SpeakReq(BaseModel):
    voice_key: str
    text: str = Field(min_length=1, max_length=1000)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    base_speaker: str = Field(default="F5-Base")    # unused under F5-TTS
    tau: float = Field(default=3.0, ge=0.0, le=10.0)  # F5-TTS cfg_strength
    nfe_step: int = Field(default=16, ge=8, le=128)   # F5-TTS flow steps


class DeleteReq(BaseModel):
    voice_key: str


@app.post("/api/enroll/start")
async def api_enroll_start(req: EnrollReq):
    if SERVICE.busy_with() is not None:
        raise HTTPException(409, f"service is busy: {SERVICE.busy_with()}")
    try:
        job = SERVICE.start_enrollment(req.display_name, req.seconds)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"job_id": job.job_id, "started_at": job.started_at}


@app.post("/api/enroll/upload")
async def api_enroll_upload(
    display_name: str = Form(..., min_length=1, max_length=64),
    file: UploadFile = File(...),
):
    """Enroll from an uploaded WAV (or any soundfile-readable audio).

    The file is saved verbatim, then voice.enroll_from_wav handles
    resampling to 16 kHz mono, VAD trimming, centered 8 s crop, and
    Whisper transcription. Result mirrors /api/enroll/start.
    """
    if SERVICE.busy_with() is not None:
        raise HTTPException(409, f"service is busy: {SERVICE.busy_with()}")

    # cheap content-type check; we mostly trust soundfile to barf if it
    # can't read the bytes.
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            raise HTTPException(400, f"unsupported audio extension: {ext}")

    # Stage the upload under data/uploads/ so it's persisted with the
    # other enrollment artifacts.
    from voice.paths import DATA_DIR
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in (file.filename or "upload.wav")
                        if c.isalnum() or c in "._-") or "upload.wav"
    staged = upload_dir / f"{int(time.time()*1000)}_{safe_name}"

    body = await file.read()
    if not body:
        raise HTTPException(400, "uploaded file is empty")
    staged.write_bytes(body)
    log.info("staged upload %s (%d bytes) for enrollment '%s'",
             staged.name, len(body), display_name)

    try:
        job = SERVICE.start_enrollment_from_wav(display_name, staged)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"job_id": job.job_id, "started_at": job.started_at,
            "source_file": staged.name}


@app.post("/api/speak")
async def api_speak(req: SpeakReq):
    if SERVICE.busy_with() not in (None, "synth"):
        raise HTTPException(409, f"service is busy: {SERVICE.busy_with()}")
    try:
        # Run synth in a worker thread so the event loop stays responsive
        result = await asyncio.to_thread(
            SERVICE.speak, req.voice_key, req.text,
            speed=req.speed, base_speaker=req.base_speaker,
            tau=req.tau, nfe_step=req.nfe_step,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        log.exception("speak failed")
        raise HTTPException(500, str(e))
    return result


@app.get("/api/speak/stream")
async def api_speak_stream(
    request: Request,
    voice_key: str,
    text: str,
    tau: float = 3.0,
    nfe_step: int = 16,
    speed: float = 1.0,
):
    """Sentence-level streaming synth.

    Client opens an EventSource here. Server splits text into sentence
    chunks, synths each one, and emits an SSE `chunk` event the moment
    each WAV lands. Browser queues + chains audio elements so playback
    starts the moment the FIRST chunk is ready while the rest render
    in the background.
    """
    import voice as voice_pkg
    import threading

    if SERVICE.busy_with() not in (None, "synth"):
        raise HTTPException(409, f"service is busy: {SERVICE.busy_with()}")

    # Reserve the synth slot for the duration of the stream
    with SERVICE._lock:
        if SERVICE._busy_with is not None and SERVICE._busy_with != "synth":
            raise HTTPException(409, f"service is busy: {SERVICE._busy_with}")
        SERVICE._busy_with = "synth"

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    SENTINEL = object()

    def worker():
        try:
            chunks = voice_pkg.split_sentences(text)
            loop.call_soon_threadsafe(queue.put_nowait,
                {"event": "plan", "data": {"chunks": len(chunks), "text": chunks}})
            for i, result in enumerate(voice_pkg.speak_stream(
                voice_key, text, speed=speed, tau=tau, nfe_step=nfe_step
            )):
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "event": "chunk",
                    "data": {
                        "index": i,
                        "total": len(chunks),
                        "wav_url": f"/audio/{result.wav_path.name}",
                        "text": result.text,
                        "seconds": result.seconds,
                        "elapsed": result.elapsed,
                        "rtf": result.rtf,
                    },
                })
        except Exception as e:
            log.exception("stream worker failed")
            loop.call_soon_threadsafe(queue.put_nowait,
                {"event": "error", "data": {"error": str(e)}})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    threading.Thread(target=worker, daemon=True,
                     name=f"stream-{voice_key}").start()

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                item = await asyncio.wait_for(queue.get(), timeout=120.0)
                if item is SENTINEL:
                    yield {"event": "done", "data": "{}"}
                    break
                yield {"event": item["event"], "data": json.dumps(item["data"])}
        finally:
            SERVICE._release()

    return EventSourceResponse(gen())


class PlayLocalReq(BaseModel):
    wav_name: str = Field(min_length=1, max_length=256)


@app.post("/api/play/local")
async def api_play_local(req: PlayLocalReq):
    """Play a synthesized WAV through the Jetson's local ALSA speaker.

    Returns immediately (fire-and-forget) — playback runs in a background
    thread so the HTTP call doesn't block while the audio plays.
    """
    from audio import play_wav
    import threading

    # only allow files that live in our output dir, defensive against
    # path traversal via wav_name
    target = (OUT_AUDIO / Path(req.wav_name).name).resolve()
    try:
        target.relative_to(OUT_AUDIO.resolve())
    except ValueError:
        raise HTTPException(400, "invalid wav_name")
    if not target.exists():
        raise HTTPException(404, f"wav not found: {req.wav_name}")

    def _play():
        try:
            play_wav(target)
            log.info("local-playback finished: %s", target.name)
        except Exception as e:
            log.exception("local-playback failed: %s", e)

    threading.Thread(target=_play, daemon=True,
                     name=f"play-{target.name}").start()
    return {"playing": target.name}


@app.post("/api/voice/delete")
async def api_voice_delete(req: DeleteReq):
    import voice as voice_pkg
    if SERVICE.busy_with() is not None:
        raise HTTPException(409, f"service is busy: {SERVICE.busy_with()}")
    ok = voice_pkg.delete_voice(req.voice_key)
    if not ok:
        raise HTTPException(404, f"no such voice: {req.voice_key}")
    return {"deleted": req.voice_key}


@app.post("/api/warmup")
async def api_warmup():
    if SERVICE.busy_with() is not None:
        raise HTTPException(409, f"service is busy: {SERVICE.busy_with()}")
    job = SERVICE.warmup_async()
    return {"job_id": job.job_id}


@app.get("/api/voices")
async def api_voices():
    return list_voices_dict()


@app.get("/api/status")
async def api_status():
    job = SERVICE.current_job
    return {
        "busy_with": SERVICE.busy_with(),
        "models_loaded": SERVICE.models_loaded,
        "job": None if job is None else {
            "job_id": job.job_id,
            "kind": job.kind,
            "state": job.state,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
            "result": job.result,
        },
        "sysstats": _last_sys,
        "voices": len(list_voices_dict()),
    }


# ---------- SSE ----------

@app.get("/events")
async def events(request: Request):
    queue = SERVICE.subscribe()

    async def gen():
        # initial snapshot
        snap = await api_status()
        yield {"event": "snapshot", "data": json.dumps(snap)}
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {"event": msg["event"], "data": json.dumps(msg["data"])}
        finally:
            SERVICE.unsubscribe(queue)

    return EventSourceResponse(gen())


# ---------- audio serving ----------

@app.get("/audio/{name:path}")
async def audio(name: str):
    # Allow either generated outputs or enrollment recordings to be played.
    candidates = [OUT_AUDIO / name, ENROLL_AUDIO / name]
    for p in candidates:
        if p.exists() and p.is_file():
            return FileResponse(str(p), media_type="audio/wav")
    raise HTTPException(404, f"audio not found: {name}")


# ---------- root health ----------

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time()}

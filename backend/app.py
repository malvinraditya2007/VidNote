"""FastAPI app VidNote (Fase 1). Hanya untuk localhost (bind 127.0.0.1).

Endpoint (plan bagian 6):
  GET  /system                 info perangkat (CUDA, mode tersedia) + token sesi
  POST /jobs/url               {url, device, lang?, num_speakers?} -> job
  POST /jobs/upload            multipart file + field -> job
  GET  /jobs/{id}              status ringkas
  GET  /jobs/{id}/events       progres via SSE
  POST /jobs/{id}/cancel       batalkan
  GET  /jobs/{id}/result       JSON hasil (ringkasan, transkrip, info)
  GET  /jobs/{id}/video|vtt|srt  file hasil (video dukung Range)

Semua endpoint lewat LocalhostSecurity (Host/Origin/token).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from backend import config, runner, security
from backend.errors import InputRejected, VidNoteError
from backend.jobqueue import JobManager

FRONTEND_DIR = config.ROOT_DIR / "frontend"

RESULT_FILES = {
    "video": ("video_hardsub.mp4", "video/mp4"),
    "vtt": ("subs.vtt", "text/vtt; charset=utf-8"),
    "srt": ("subs.srt", "application/x-subrip; charset=utf-8"),
    "ass": ("subs.ass", "text/plain; charset=utf-8"),
}


def create_app(manager: Optional[JobManager] = None, port: int = 8765) -> FastAPI:
    app = FastAPI(title="VidNote", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager = manager or JobManager()
    app.state.port = port
    app.add_middleware(security.LocalhostSecurity, port=port)

    # Sapu sisa folder job & file upload staging dari sesi yang mati sebelumnya.
    import time as _t

    from backend import jobs as _jobs

    _jobs.sweep_stale()
    if config.WORK_DIR.is_dir():
        cutoff = _t.time() - config.STALE_JOB_HOURS * 3600
        for f in config.WORK_DIR.glob("upload-*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)

    def mgr() -> JobManager:
        return app.state.manager

    def get_job_or_404(job_id: str):
        job = mgr().get(job_id)
        if not job:
            raise HTTPException(404, "job tidak ditemukan")
        return job

    def parse_device(v: str) -> str:
        from backend import device as device_mod

        if v not in ("gpu", "cpu"):
            raise HTTPException(422, "device harus 'gpu' atau 'cpu'")
        if v == "gpu" and not device_mod.gpu_available():
            raise HTTPException(422, "GPU tidak tersedia di mesin ini")
        return v

    def build_options(device: str, lang: str, num_speakers: Optional[int],
                      no_summary: bool, no_burn: bool, diarizer: str = config.DIAR_DEFAULT_BACKEND
                      ) -> runner.RunOptions:
        if lang not in ("auto", "id", "en"):
            raise HTTPException(422, "lang harus auto/id/en")
        if num_speakers is not None and not (1 <= num_speakers <= config.DIAR_MAX_SPEAKERS * 2):
            raise HTTPException(422, "num_speakers di luar rentang")
        if diarizer not in config.DIAR_BACKENDS:
            raise HTTPException(422, "diarizer harus sherpa/pyannote")
        return runner.RunOptions(device=device, lang=lang, num_speakers=num_speakers,
                                 skip_summary=no_summary, skip_burn=no_burn, diarizer=diarizer)

    def submit_or_409(source: str, src_arg: str, opts: runner.RunOptions):
        try:
            job = mgr().submit(source, src_arg, opts)
        except VidNoteError as e:
            raise HTTPException(409, str(e))
        return {"job_id": job.id, "status": job.status}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        # Suntik token sesi ke halaman (same-origin). Situs lain tak bisa membaca ini
        # karena request mereka sudah ditolak Host/Origin lebih dulu.
        html = html.replace("__VIDNOTE_TOKEN__", security.SESSION_TOKEN)
        return HTMLResponse(html)

    @app.get("/style.css")
    def style():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @app.get("/app.js")
    def appjs():
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/system")
    def system():
        from backend import device as device_mod

        info = device_mod.query_nvidia_smi()
        gpu = device_mod.gpu_available()
        return {
            "token": security.SESSION_TOKEN,
            "gpu_available": gpu,
            "gpu_name": info.name if info else None,
            "vram_mb": info.vram_mb if info else None,
            "devices": (["gpu", "cpu"] if gpu else ["cpu"]),
            "max_duration_sec": config.MAX_DURATION_SEC,
            "max_upload_bytes": config.MAX_UPLOAD_BYTES,
            "languages": ["auto", *config.SUPPORTED_LANGS],
            "busy": mgr().is_busy(),
        }

    @app.post("/jobs/url")
    async def jobs_url(request: Request):
        body = await request.json()
        url = str(body.get("url") or "").strip()
        if not url:
            raise HTTPException(422, "url wajib diisi")
        from backend.pipeline.fetch_url import normalize_url

        try:
            normalize_url(url)  # validasi awal sebelum job dibuat
        except InputRejected as e:
            raise HTTPException(422, str(e))
        device = parse_device(str(body.get("device", "")))
        opts = build_options(device, str(body.get("lang", "auto")), body.get("num_speakers"),
                             bool(body.get("no_summary")), bool(body.get("no_burn")),
                             str(body.get("diarizer", config.DIAR_DEFAULT_BACKEND)))
        cb = body.get("cookies_from_browser")
        if cb:
            from backend.pipeline.fetch_url import SUPPORTED_BROWSERS

            if str(cb).lower() not in SUPPORTED_BROWSERS:
                raise HTTPException(422, "browser cookie tidak didukung")
            opts.cookies_from_browser = str(cb).lower()
        return submit_or_409(f"url:{url}", url, opts)

    @app.post("/jobs/upload")
    async def jobs_upload(
        file: UploadFile,
        device: str = Form(...),
        lang: str = Form("auto"),
        num_speakers: Optional[int] = Form(None),
        no_summary: bool = Form(False),
        no_burn: bool = Form(False),
    ):
        dev = parse_device(device)
        opts = build_options(dev, lang, num_speakers, no_summary, no_burn)
        ext = Path(file.filename or "").suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(422, f"ekstensi {ext or '(kosong)'} tidak didukung")
        # Simpan ke staging (nama asli diabaikan); runner mem-probe & memvalidasi lagi.
        config.WORK_DIR.mkdir(parents=True, exist_ok=True)
        import uuid as _uuid

        staged = config.WORK_DIR / f"upload-{_uuid.uuid4().hex}{ext}"
        size = 0
        try:
            with open(staged, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > config.MAX_UPLOAD_BYTES:
                        out.close()
                        staged.unlink(missing_ok=True)
                        raise HTTPException(413, "file melebihi batas ukuran")
                    out.write(chunk)
        finally:
            await file.close()
        if size == 0:
            staged.unlink(missing_ok=True)
            raise HTTPException(422, "file kosong")
        opts.delete_source = True   # runner hapus staging setelah disalin ke folder job
        try:
            return submit_or_409(f"upload:{file.filename}", str(staged), opts)
        except HTTPException:
            staged.unlink(missing_ok=True)   # job ditolak (409) -> jangan tinggalkan staging
            raise

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = get_job_or_404(job_id)
        return {"job_id": job.id, "status": job.status, "error": job.error,
                "events": len(job.events)}

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        job = get_job_or_404(job_id)

        async def stream():
            import anyio

            seq = 0
            yield f"retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    return
                events = await anyio.to_thread.run_sync(lambda: job.events_since(seq, 20.0))
                for ev in events:
                    seq = ev.seq + 1
                    yield f"event: {ev.type}\ndata: {json.dumps(ev.data, ensure_ascii=False)}\n\n"
                if job.finished and seq >= len(job.events):
                    return
                if not events:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        get_job_or_404(job_id)
        return {"cancelled": mgr().cancel(job_id)}

    def _require_done(job_id: str):
        job = get_job_or_404(job_id)
        if job.status != "done" or not job.out_dir:
            raise HTTPException(409, f"job belum selesai (status: {job.status})")
        return job

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = _require_done(job_id)
        tj = job.out_dir / "transcript.json"
        sj = job.out_dir / "summary.json"
        transcript = json.loads(tj.read_text(encoding="utf-8")) if tj.is_file() else None
        summary = json.loads(sj.read_text(encoding="utf-8")) if sj.is_file() else None
        return {"job_id": job.id, "title": job.result.get("title"),
                "language": job.result.get("language"),
                "num_speakers": job.result.get("num_speakers"),
                "models": job.result.get("models"), "warnings": job.result.get("warnings"),
                "available": sorted(p.name for p in job.out_dir.iterdir()),
                "transcript": transcript, "summary": summary}

    @app.get("/jobs/{job_id}/{kind}")
    def job_file(job_id: str, kind: str, request: Request):
        if kind not in RESULT_FILES:
            raise HTTPException(404, "tidak ditemukan")
        job = _require_done(job_id)
        name, media = RESULT_FILES[kind]
        path = job.out_dir / name
        if not path.is_file():
            raise HTTPException(404, f"{name} tidak tersedia")
        # FileResponse mendukung Range (seek video) secara bawaan.
        return FileResponse(path, media_type=media, filename=name)

    @app.exception_handler(InputRejected)
    async def _rejected(_request: Request, exc: InputRejected):
        return JSONResponse({"detail": exc.reasons}, status_code=422)

    return app

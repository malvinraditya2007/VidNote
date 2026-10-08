"""Antrean satu job di thread background untuk API.

Plan 5.4: satu job pada satu waktu. Job baru ditolak selama ada yang berjalan.
Progres dari callback runner disimpan sebagai daftar event; endpoint SSE
membacanya. Pembatalan memakai flag kooperatif yang dicek runner di antar-tahap.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend import runner
from backend.errors import InputRejected, VidNoteError


class Cancelled(VidNoteError):
    pass


@dataclass
class Event:
    seq: int
    ts: float
    type: str                 # stage | progress | info | done | error | cancelled
    data: dict


@dataclass
class Job:
    id: str
    source: str               # "url:<...>" atau "upload:<nama>"
    options: runner.RunOptions
    status: str = "running"   # running | done | error | cancelled
    events: list[Event] = field(default_factory=list)
    result: Optional[dict] = None    # ringkasan RunResult (path, files, ...)
    error: Optional[str] = None
    out_dir: Optional[Path] = None
    created: float = field(default_factory=time.time)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _cond: threading.Condition = field(default_factory=threading.Condition)

    def add(self, type_: str, data: dict) -> None:
        with self._cond:
            self.events.append(Event(len(self.events), time.time(), type_, data))
            self._cond.notify_all()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("job dibatalkan")

    def events_since(self, seq: int, timeout: float = 25.0) -> list[Event]:
        """Blok sampai ada event dengan index >= seq, atau timeout (untuk SSE)."""
        with self._cond:
            if seq >= len(self.events):
                self._cond.wait(timeout)
            return self.events[seq:]

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error", "cancelled")


class JobManager:
    """Satu job aktif pada satu waktu. Thread-safe."""

    def __init__(self, run_fn=runner.run):
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self._current: Optional[Job] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def current(self) -> Optional[Job]:
        return self._current

    def get(self, job_id: str) -> Optional[Job]:
        j = self._current
        return j if j and j.id == job_id else None

    def is_busy(self) -> bool:
        j = self._current
        return j is not None and not j.finished

    def submit(self, source: str, src_arg: str, options: runner.RunOptions) -> Job:
        with self._lock:
            if self.is_busy():
                raise VidNoteError("sedang ada job berjalan; coba lagi setelah selesai atau batalkan")
            job = Job(id=str(uuid.uuid4()), source=source, options=options)
            self._current = job
            self._thread = threading.Thread(target=self._run, args=(job, src_arg), daemon=True)
            self._thread.start()
            return job

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.finished:
            return False
        job._cancel.set()
        return True

    def _run(self, job: Job, src_arg: str) -> None:
        def on_stage(stage: str, msg: str) -> None:
            job.check_cancel()
            job.add("stage", {"stage": stage, "message": msg})

        def on_progress(stage: str, frac: float) -> None:
            job.check_cancel()
            job.add("progress", {"stage": stage, "fraction": round(frac, 3)})

        def on_info(info: dict) -> None:
            job.add("info", info)

        def on_oom(here: str, there: str) -> bool:
            job.add("stage", {"stage": "asr", "message": f"VRAM habis: {here} -> {there}"})
            return True  # API: otomatis fallback (tak ada prompt interaktif)

        try:
            res = self._run_fn(src_arg, job.options, on_stage=on_stage, on_progress=on_progress,
                               on_oom=on_oom, on_info=on_info)
            job.out_dir = res.out_dir
            job.result = {
                "title": res.title, "files": res.files, "timings": res.timings,
                "warnings": res.warnings, "num_speakers": res.info.get("num_speakers"),
                "language": res.info.get("language"), "models": res.info.get("models", {}),
            }
            job.status = "done"
            job.add("done", job.result)
        except Cancelled:
            job.status = "cancelled"
            job.add("cancelled", {})
        except (InputRejected, VidNoteError) as e:
            job.status = "error"
            job.error = str(e)
            job.add("error", {"message": str(e), "reasons": getattr(e, "reasons", None)})
        except Exception as e:  # noqa: BLE001 - jangan sampai thread mati diam-diam
            job.status = "error"
            job.error = f"kesalahan tak terduga: {e}"
            job.add("error", {"message": job.error})

"""Folder kerja per job: work/{uuid}/.

- Job ID = UUID4 acak; tidak pernah diambil dari input user.
- Folder dihapus saat selesai, gagal, atau Ctrl+C (lewat context manager).
- Saat start, folder job yang sudah basi (> STALE_JOB_HOURS) disapu.
"""
from __future__ import annotations

import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from backend import config


@dataclass
class Job:
    id: str
    dir: Path

    def path(self, name: str) -> Path:
        """Path file di dalam folder job. Nama harus nama file biasa (tanpa folder)."""
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError(f"nama file tidak valid: {name!r}")
        return self.dir / name


def _is_uuid(name: str) -> bool:
    try:
        return str(uuid.UUID(name)) == name
    except ValueError:
        return False


def create_job(work_dir: Optional[Path] = None) -> Job:
    base = Path(work_dir or config.WORK_DIR)
    base.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    d = base / job_id
    d.mkdir()
    return Job(job_id, d)


def cleanup(job: Job) -> None:
    shutil.rmtree(job.dir, ignore_errors=True)


def sweep_stale(work_dir: Optional[Path] = None, max_age_hours: float = config.STALE_JOB_HOURS) -> int:
    """Hapus folder job basi. Hanya folder bernama UUID yang disentuh."""
    base = Path(work_dir or config.WORK_DIR)
    if not base.is_dir():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for d in base.iterdir():
        if d.is_dir() and _is_uuid(d.name) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


@contextmanager
def job_context(keep: bool = False, work_dir: Optional[Path] = None) -> Iterator[Job]:
    """Buat job, lalu hapus foldernya di akhir (termasuk saat error/Ctrl+C) kecuali keep=True."""
    sweep_stale(work_dir)
    job = create_job(work_dir)
    try:
        yield job
    finally:
        if not keep:
            cleanup(job)

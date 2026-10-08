"""Terima file video lokal ke folder job.

Urutan: cek ekstensi + ukuran -> salin streaming ke work/{uuid}/input.ext
(nama asli diabaikan) -> probe salinan -> validasi. Probe dilakukan pada
salinan, bukan file asli, supaya yang divalidasi = yang diproses.
"""
from __future__ import annotations

from pathlib import Path

from backend import config
from backend.errors import InputRejected
from backend.jobs import Job
from backend.pipeline import probe as probe_mod

CHUNK = 1024 * 1024


def check_source(src: Path) -> str:
    """Validasi awal sebelum menyalin. Kembalikan ekstensi (lowercase)."""
    src = Path(src)
    if not src.is_file():
        raise InputRejected(f"file tidak ditemukan: {src}")
    ext = src.suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = " ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise InputRejected(f"ekstensi {ext or '(kosong)'} tidak didukung (boleh: {allowed})")
    size = src.stat().st_size
    if size == 0:
        raise InputRejected("file kosong")
    if size > config.MAX_UPLOAD_BYTES:
        raise InputRejected(
            f"ukuran {size / 1024**3:.2f} GB melebihi batas "
            f"{config.MAX_UPLOAD_BYTES / 1024**3:.0f} GB"
        )
    return ext


def copy_into_job(src: Path, job: Job, ext: str) -> Path:
    dst = job.path("input" + ext)
    copied = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            buf = fin.read(CHUNK)
            if not buf:
                break
            copied += len(buf)
            if copied > config.MAX_UPLOAD_BYTES:  # file membesar saat disalin
                raise InputRejected("ukuran file melebihi batas")
            fout.write(buf)
    return dst


def receive_file(src: Path, job: Job) -> tuple[Path, probe_mod.ProbeInfo]:
    """Salin + probe + validasi. Raise InputRejected jika tidak lolos."""
    ext = check_source(src)
    dst = copy_into_job(src, job, ext)
    return dst, probe_mod.check_file(dst)

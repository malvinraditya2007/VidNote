"""Unduh model non-Hugging-Face (sherpa-onnx) dengan aman.

- Hanya HTTPS, hanya host GitHub resmi.
- Ditulis ke file sementara, SHA256 diverifikasi, baru di-rename.
- Arsip tar: hanya satu member yang sudah ditentukan yang diekstrak
  (tanpa path traversal, tanpa symlink).
"""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from backend import config
from backend.errors import VidNoteError

ALLOWED_HOSTS = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def _check_url(url: str) -> None:
    p = urlsplit(url)
    if p.scheme != "https" or p.hostname not in ALLOWED_HOSTS:
        raise VidNoteError(f"URL model tidak diizinkan: {url}")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, dst: Path, sha256: str,
             on_progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Unduh url -> dst dengan verifikasi SHA256. Lewati jika dst sudah ada & cocok."""
    _check_url(url)
    dst = Path(dst)
    if dst.is_file() and sha256_of(dst) == sha256:
        return dst
    if config.OFFLINE:
        raise VidNoteError(f"model {dst.name} belum terunduh dan mode offline aktif. "
                           "Jalankan sekali: python vidnote.py download-models")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    opener = urllib.request.build_opener(_SafeRedirect)
    h = hashlib.sha256()
    done = 0
    try:
        with opener.open(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                done += len(buf)
                if done > config.DOWNLOAD_MAX_BYTES:
                    raise VidNoteError("file model melebihi batas ukuran")
                h.update(buf)
                f.write(buf)
                if on_progress:
                    on_progress(done, total)
        if h.hexdigest() != sha256:
            raise VidNoteError(f"hash SHA256 model tidak cocok untuk {dst.name}; unduhan ditolak")
        tmp.replace(dst)
    except OSError as e:
        raise VidNoteError(f"gagal mengunduh model {dst.name}: {e}")
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def extract_member(archive: Path, member: str, dst: Path) -> Path:
    """Ekstrak satu file reguler dari tar.* ke dst (bukan ke path dari arsip)."""
    with tarfile.open(archive, "r:*") as tar:
        try:
            info = tar.getmember(member)
        except KeyError:
            raise VidNoteError(f"{member} tidak ada di arsip {archive.name}")
        if not info.isfile():
            raise VidNoteError(f"{member} di arsip bukan file biasa")
        src = tar.extractfile(info)
        if src is None:
            raise VidNoteError(f"gagal membaca {member}")
        tmp = dst.with_name(dst.name + ".part")
        with src, open(tmp, "wb") as f:
            for buf in iter(lambda: src.read(CHUNK), b""):
                f.write(buf)
        tmp.replace(dst)
    return dst

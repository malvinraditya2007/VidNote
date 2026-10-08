"""Probe video dengan ffprobe dan validasi hasilnya.

Keamanan:
- Argumen subprocess berupa list, tanpa shell.
- `-protocol_whitelist file,pipe` supaya file berisi playlist/URL tidak bisa
  memicu ffprobe membuka jaringan atau file lain.
- Dijalankan dengan cwd = folder file dan nama file relatif, jadi path dengan
  ':' atau awalan '-' tidak bisa disalahartikan sebagai protokol/opsi.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from backend import config
from backend.errors import ToolError


@dataclass
class ProbeInfo:
    duration: float
    width: int            # lebar tampilan (setelah rotasi)
    height: int           # tinggi tampilan (setelah rotasi)
    rotation: int         # derajat, dinormalisasi ke 0/90/180/270
    orientation: str      # "landscape" | "portrait"
    has_video: bool
    has_audio: bool
    video_codec: Optional[str]
    audio_codec: Optional[str]
    container: str        # format_name dari ffprobe
    size_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- fungsi murni ----------

def _rotation_of(stream: dict) -> int:
    rot = None
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
            break
    if rot is None:
        rot = (stream.get("tags") or {}).get("rotate")
    try:
        rot = int(float(rot or 0))
    except (TypeError, ValueError):
        rot = 0
    return rot % 360


def _main_video_stream(streams: list[dict]) -> Optional[dict]:
    """Stream video pertama yang bukan cover art (attached_pic)."""
    for s in streams:
        if s.get("codec_type") != "video":
            continue
        if (s.get("disposition") or {}).get("attached_pic"):
            continue
        return s
    return None


def _float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # buang NaN


def parse_probe(data: dict) -> ProbeInfo:
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    v = _main_video_stream(streams)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _float(fmt.get("duration"))
    if duration is None:
        duration = max((_float(s.get("duration")) or 0.0 for s in streams), default=0.0)

    w = int(v.get("width") or 0) if v else 0
    h = int(v.get("height") or 0) if v else 0
    rotation = _rotation_of(v) if v else 0
    if rotation in (90, 270):
        w, h = h, w

    return ProbeInfo(
        duration=round(duration, 3),
        width=w,
        height=h,
        rotation=rotation,
        orientation="portrait" if h > w else "landscape",
        has_video=v is not None,
        has_audio=a is not None,
        video_codec=v.get("codec_name") if v else None,
        audio_codec=a.get("codec_name") if a else None,
        container=str(fmt.get("format_name") or ""),
        size_bytes=int(fmt.get("size") or 0),
    )


def validate(info: ProbeInfo) -> list[str]:
    """Kembalikan daftar alasan penolakan (kosong = valid)."""
    errs: list[str] = []
    containers = set(info.container.split(",")) if info.container else set()
    if not containers & config.ALLOWED_CONTAINERS:
        errs.append(f"format file tidak didukung ({info.container or 'tidak dikenali'})")
    if not info.has_video:
        errs.append("tidak ada stream video")
    if not info.has_audio:
        errs.append("tidak ada stream audio")
    if info.duration < config.MIN_DURATION_SEC:
        errs.append("durasi terlalu pendek atau tidak terbaca")
    if info.duration > config.MAX_DURATION_SEC:
        errs.append(
            f"durasi {fmt_duration(info.duration)} melebihi batas "
            f"{fmt_duration(config.MAX_DURATION_SEC)}"
        )
    if info.has_video:
        if min(info.width, info.height) < config.MIN_DIMENSION:
            errs.append(f"resolusi terlalu kecil ({info.width}x{info.height})")
        if max(info.width, info.height) > config.MAX_DIMENSION:
            errs.append(
                f"resolusi terlalu besar ({info.width}x{info.height}, "
                f"maks {config.MAX_DIMENSION} px)"
            )
    return errs


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------- pemanggilan ffprobe ----------

def run_ffprobe(path: Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise ToolError("ffprobe tidak ditemukan. Jalankan `python vidnote.py doctor`.")
    path = Path(path).resolve()
    cmd = [
        exe, "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-print_format", "json",
        "-show_format", "-show_streams",
        "file:" + path.name,
    ]
    try:
        out = subprocess.run(
            cmd, cwd=path.parent, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=config.FFPROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"ffprobe timeout ({config.FFPROBE_TIMEOUT} dtk)")
    if out.returncode != 0:
        msg = (out.stderr or "").strip().splitlines()
        raise ToolError("file tidak bisa dibaca sebagai video" + (f": {msg[-1]}" if msg else ""))
    try:
        return json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        raise ToolError("output ffprobe tidak valid")


def probe(path: Path) -> ProbeInfo:
    return parse_probe(run_ffprobe(path))


def check_file(path: Path) -> ProbeInfo:
    """Probe + validasi. Raise InputRejected (dengan info jika ada) bila tidak lolos."""
    from backend.errors import InputRejected

    try:
        info = probe(path)
    except ToolError as e:
        raise InputRejected(str(e))
    errs = validate(info)
    if errs:
        raise InputRejected(errs, info=info)
    return info

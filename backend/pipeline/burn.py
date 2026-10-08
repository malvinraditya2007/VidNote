"""Burn hardsub: video + subs.ass -> output.mp4 (H.264 + AAC, faststart).

- ffmpeg dijalankan dengan cwd = folder job dan filter `ass=subs.ass` (nama tetap),
  jadi tidak ada masalah escaping path Windows (C:\\...) di filtergraph.
- GPU: h264_nvenc. Jika NVENC gagal (driver/GPU tidak mendukung), otomatis ulang
  dengan libx264 veryfast.
- Rotasi metadata diterapkan otomatis oleh ffmpeg sebelum filter, jadi subtitle
  mengikuti orientasi tampilan. Dimensi dibulatkan ke genap (syarat yuv420p).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from backend import config
from backend.errors import ToolError
from backend.jobs import Job

SUBS_NAME = "subs.ass"
VIDEO_FILTER = "scale=trunc(iw/2)*2:trunc(ih/2)*2,ass=" + SUBS_NAME + ",format=yuv420p"


@dataclass
class BurnResult:
    path: Path
    encoder: str
    elapsed_sec: float
    fell_back: bool
    note: str = ""


def build_cmd(exe: str, input_name: str, encoder: str, audio_codec: Optional[str]) -> list[str]:
    venc = config.BURN_NVENC if encoder == "h264_nvenc" else config.BURN_X264
    # AAC dari sumber bisa di-copy; selain itu encode ulang (Opus/Vorbis tidak ramah MP4/browser).
    aenc = ["-c:a", "copy"] if audio_codec == "aac" else ["-c:a", "aac", "-b:a", config.BURN_AUDIO_BITRATE]
    return [
        exe, "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-protocol_whitelist", "file,pipe",
        "-i", "file:" + input_name,
        "-map", "0:v:0", "-map", "0:a:0", "-sn", "-dn",
        "-vf", VIDEO_FILTER,
        *venc,
        *aenc,
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        "file:" + config.BURN_OUTPUT,
    ]


def parse_progress_line(line: str) -> Optional[float]:
    """'out_time_us=12345678' -> 12.345678 detik."""
    key, _, val = line.strip().partition("=")
    if key in ("out_time_us", "out_time_ms") and val.isdigit():
        return int(val) / 1_000_000  # ffmpeg: out_time_ms juga dalam mikrodetik
    return None


def summarize_error(stderr: str) -> str:
    """Ambil baris error ffmpeg yang paling informatif (driver/nvenc), bukan baris penutup generik."""
    import re

    lines = [re.sub(r"^\[[^\]]*\]\s*", "", ln).strip() for ln in (stderr or "").splitlines()]
    lines = [ln for ln in lines if ln]
    for key in ("driver", "nvenc", "no capable devices", "cannot load", "not found"):
        for ln in lines:
            if key in ln.lower():
                return ln
    return lines[-1] if lines else ""


def _run(cmd: list[str], cwd: Path, duration: float, timeout: float,
         on_progress: Optional[Callable[[float], None]]) -> tuple[int, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            t = parse_progress_line(line)
            if t is not None and on_progress and duration > 0:
                on_progress(min(1.0, t / duration))
            if time.monotonic() > deadline:
                proc.kill()
                raise ToolError(f"burn hardsub timeout ({int(timeout)} dtk)")
        stderr = proc.stderr.read() if proc.stderr else ""
        return proc.wait(timeout=30), stderr
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise


def burn(input_path: Path, job: Job, device: str, duration: float, audio_codec: Optional[str],
         on_status: Optional[Callable[[str], None]] = None,
         on_progress: Optional[Callable[[float], None]] = None) -> BurnResult:
    """Burn job/subs.ass ke video input (keduanya harus ada di folder job)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise ToolError("ffmpeg tidak ditemukan. Jalankan `python vidnote.py doctor`.")
    input_path = Path(input_path).resolve()
    if input_path.parent != job.dir.resolve():
        raise ValueError("input harus berada di folder job")
    if not job.path(SUBS_NAME).is_file():
        raise ToolError(f"{SUBS_NAME} tidak ada di folder job")
    status = on_status or (lambda _m: None)
    out = job.path(config.BURN_OUTPUT)
    encoders = ["h264_nvenc", "libx264"] if device == "gpu" else ["libx264"]
    note = ""
    t0 = time.perf_counter()
    for i, enc in enumerate(encoders):
        dev = "gpu" if enc == "h264_nvenc" else "cpu"
        timeout = config.BURN_TIMEOUT_BASE + config.BURN_TIMEOUT_FACTOR[dev] * max(duration, 0)
        status(f"Burn hardsub ({enc})")
        code, err = _run(build_cmd(exe, input_path.name, enc, audio_codec), job.dir, duration,
                         timeout, on_progress)
        if code == 0 and out.is_file() and out.stat().st_size > 0:
            return BurnResult(out, enc, round(time.perf_counter() - t0, 1), i > 0, note)
        out.unlink(missing_ok=True)
        msg = summarize_error(err) or f"exit {code}"
        if enc == "h264_nvenc" and i + 1 < len(encoders):
            note = f"NVENC gagal ({msg}); diulang dengan libx264"
            status(note)
            continue
        raise ToolError(f"burn hardsub gagal: {msg}")
    raise ToolError("burn hardsub gagal")  # tidak tercapai

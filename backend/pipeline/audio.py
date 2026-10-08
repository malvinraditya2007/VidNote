"""Ekstraksi audio: WAV PCM 16-bit, 16 kHz, mono, dinormalisasi (loudnorm).

Dipakai oleh ASR (faster-whisper) dan diarization (sherpa-onnx), keduanya
mengharapkan 16 kHz mono. Filter: downmix mono -> loudnorm -> resample 16 kHz.

Keamanan: argumen list tanpa shell, -protocol_whitelist file,pipe,
cwd = folder job dengan nama file tetap, timeout sesuai durasi.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from backend import config
from backend.errors import ToolError
from backend.jobs import Job


@dataclass
class AudioInfo:
    path: Path
    sample_rate: int
    channels: int
    duration: float
    rms_dbfs: float
    peak_dbfs: float

    @property
    def is_near_silent(self) -> bool:
        return self.rms_dbfs < config.SILENCE_RMS_DBFS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["is_near_silent"] = self.is_near_silent
        return d


def build_ffmpeg_cmd(exe: str, input_name: str, output_name: str) -> list[str]:
    af = ",".join([
        "aformat=channel_layouts=mono",
        config.LOUDNORM_FILTER,
        f"aresample={config.AUDIO_SAMPLE_RATE}",
    ])
    return [
        exe, "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-protocol_whitelist", "file,pipe",
        "-i", "file:" + input_name,
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-af", af,
        "-ar", str(config.AUDIO_SAMPLE_RATE), "-ac", "1",
        "-c:a", "pcm_s16le", "-f", "wav",
        "file:" + output_name,
    ]


def _dbfs(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -120.0


def read_stats(path: Path) -> AudioInfo:
    """Baca header + level RMS/peak dari WAV PCM 16-bit."""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if sw != 2:
            raise ToolError(f"WAV tidak terduga (sample width {sw})")
        samples = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return AudioInfo(
        path=Path(path), sample_rate=sr, channels=ch,
        duration=round(n / sr, 3) if sr else 0.0,
        rms_dbfs=round(_dbfs(rms), 1), peak_dbfs=round(_dbfs(peak), 1),
    )


def extract_audio(input_path: Path, job: Job, duration: float) -> AudioInfo:
    """Ekstrak audio dari input (harus di dalam folder job) ke job/audio.wav."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise ToolError("ffmpeg tidak ditemukan. Jalankan `python vidnote.py doctor`.")
    input_path = Path(input_path).resolve()
    if input_path.parent != job.dir.resolve():
        raise ValueError("input harus berada di folder job")
    out = job.path(config.AUDIO_FILENAME)
    timeout = config.AUDIO_TIMEOUT_BASE + config.AUDIO_TIMEOUT_FACTOR * max(duration, 0)
    cmd = build_ffmpeg_cmd(exe, input_path.name, out.name)
    try:
        res = subprocess.run(cmd, cwd=job.dir, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        out.unlink(missing_ok=True)
        raise ToolError(f"ekstraksi audio timeout ({int(timeout)} dtk)")
    if res.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        lines = (res.stderr or "").strip().splitlines()
        raise ToolError("ekstraksi audio gagal" + (f": {lines[-1]}" if lines else ""))
    return read_stats(out)

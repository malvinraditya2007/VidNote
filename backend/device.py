"""Deteksi perangkat (CUDA/VRAM) dan pemilihan model per device.

Library CUDA (cuBLAS, cuDNN 9) dipasang lewat pip (paket nvidia-*-cu12).
Di Windows, folder DLL-nya harus didaftarkan dengan os.add_dll_directory
sebelum ctranslate2 memuat model di GPU.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend import config

_dll_handles: list = []  # simpan handle supaya direktori tetap terdaftar


def nvidia_pip_lib_dirs() -> list[Path]:
    """Cari folder bin/lib dari paket pip nvidia-* (cublas, cudnn, dll.)."""
    dirs: list[Path] = []
    for site in map(Path, sys.path):
        nv = site / "nvidia"
        if not nv.is_dir():
            continue
        for pkg in nv.iterdir():
            for sub in ("bin", "lib"):
                d = pkg / sub
                if d.is_dir():
                    dirs.append(d)
    return dirs


def setup_cuda_libs() -> list[Path]:
    """Daftarkan DLL CUDA dari pip. Aman dipanggil berkali-kali.

    Windows: os.add_dll_directory + PATH.
    Linux: library harus ada di LD_LIBRARY_PATH sebelum proses start;
    di sini hanya dikembalikan daftar foldernya agar doctor bisa memberi saran.
    """
    dirs = nvidia_pip_lib_dirs()
    if sys.platform == "win32":
        for d in dirs:
            if any(str(d) == getattr(h, "path", None) for h in _dll_handles):
                continue
            try:
                _dll_handles.append(os.add_dll_directory(str(d)))
            except OSError:
                continue
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    return dirs


@dataclass
class GpuInfo:
    name: str
    vram_mb: int


def query_nvidia_smi() -> Optional[GpuInfo]:
    """Baca nama GPU dan total VRAM lewat nvidia-smi (None jika tidak ada)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return parse_nvidia_smi(out)


def parse_nvidia_smi(output: str) -> Optional[GpuInfo]:
    """Parse baris pertama 'Nama GPU, 6141'."""
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            return GpuInfo(name=parts[0], vram_mb=int(parts[1]))
    return None


def cuda_device_count() -> int:
    """Jumlah GPU CUDA yang bisa dipakai ctranslate2 (0 jika tidak ada/belum terpasang)."""
    setup_cuda_libs()
    try:
        import ctranslate2  # type: ignore
    except ImportError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def gpu_available() -> bool:
    return sys.platform != "darwin" and cuda_device_count() > 0


def require_device(device: str) -> None:
    """Pastikan device yang diminta bisa dipakai. Tidak diam-diam pindah ke CPU."""
    from backend.errors import VidNoteError

    if device not in ("gpu", "cpu"):
        raise VidNoteError(f"device harus 'gpu' atau 'cpu', bukan {device!r}")
    if device == "gpu" and not gpu_available():
        raise VidNoteError(
            "GPU (CUDA) tidak tersedia. Pakai --device cpu, atau cek `python vidnote.py doctor`."
        )


def cpu_threads() -> int:
    """Perkiraan jumlah core fisik (hyperthreading jarang membantu ctranslate2)."""
    n = os.cpu_count() or 4
    return max(4, n // 2)


def _gpu_used_mb() -> Optional[int]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
        return int(out[0]) if out and out[0].strip().isdigit() else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


class VramMonitor:
    """Catat puncak pemakaian VRAM (seluruh GPU) selama blok `with` via nvidia-smi.

    Angka = puncak - baseline sebelum mulai, jadi aplikasi lain ikut terhitung
    jika pemakaiannya berubah. Cukup untuk perkiraan.
    """

    def __init__(self, interval: float = 0.5):
        import threading

        self.interval = interval
        self.baseline: Optional[int] = None
        self.peak: Optional[int] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = _gpu_used_mb()
            if used is not None:
                self.peak = used if self.peak is None else max(self.peak, used)
            self._stop.wait(self.interval)

    def __enter__(self) -> "VramMonitor":
        self.baseline = _gpu_used_mb()
        if self.baseline is not None:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    @property
    def delta_mb(self) -> Optional[int]:
        if self.baseline is None or self.peak is None:
            return None
        return max(0, self.peak - self.baseline)


def models_for(device: str) -> dict:
    if device not in config.MODELS:
        raise ValueError(f"device harus 'gpu' atau 'cpu', bukan {device!r}")
    return dict(config.MODELS[device])

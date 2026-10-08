"""Pengecekan environment: `python vidnote.py doctor`.

Tiap cek mengembalikan CheckResult (OK / WARN / FAIL) plus saran perbaikan per OS.
Fungsi parsing dibuat murni supaya bisa di-unit-test tanpa program terpasang.
"""
from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from backend import config, device

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    hint: str = ""


# ---------- helper murni (di-unit-test) ----------

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version(text: str) -> Optional[tuple[int, ...]]:
    """Ambil versi pertama 'x.y[.z]' dari teks. 'ffmpeg version n7.1-...' -> (7, 1)."""
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


def python_status(ver: tuple[int, int]) -> str:
    if ver < config.PYTHON_MIN:
        return FAIL
    if ver > config.PYTHON_MAX:
        return WARN
    return OK


def ram_status(total_gb: float) -> str:
    if total_gb < config.MIN_RAM_GB:
        return FAIL
    if total_gb < config.RECOMMENDED_RAM_GB:
        return WARN
    return OK


def disk_status(free_gb: float) -> str:
    return OK if free_gb >= config.MIN_FREE_DISK_GB else FAIL


def overall(results: list[CheckResult]) -> str:
    statuses = {r.status for r in results}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return OK


def install_hint(tool: str, platform: str = sys.platform) -> str:
    hints = {
        "python": {
            "win32": "winget install Python.Python.3.11",
            "linux": "sudo apt install python3.11 python3.11-venv",
            "darwin": "brew install python@3.11",
        },
        "ffmpeg": {
            "win32": "winget install Gyan.FFmpeg (lalu buka terminal baru)",
            "linux": "sudo apt install ffmpeg",
            "darwin": "brew install ffmpeg",
        },
        "deno": {
            "win32": "winget install DenoLand.Deno",
            "linux": "curl -fsSL https://deno.land/install.sh | sh",
            "darwin": "brew install deno",
        },
        "ollama": {
            "win32": "winget install Ollama.Ollama",
            "linux": "lihat https://ollama.com/download/linux",
            "darwin": "brew install ollama",
        },
    }
    key = "linux" if platform.startswith("linux") else platform
    return hints.get(tool, {}).get(key, "lihat README.md")


# ---------- pengecekan nyata ----------

def _run_version(exe: str, arg: str = "-version") -> str:
    out = subprocess.run([exe, arg], capture_output=True, text=True, timeout=15)
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else ""


def check_python() -> CheckResult:
    ver = sys.version_info[:2]
    st = python_status(ver)
    lo, hi = config.PYTHON_MIN, config.PYTHON_MAX
    hint = "" if st == OK else f"Butuh Python {lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}: {install_hint('python')}"
    return CheckResult("Python", st, f"{ver[0]}.{ver[1]} ({sys.executable})", hint)


def check_binary(name: str, tool_hint: str) -> CheckResult:
    exe = shutil.which(name)
    if not exe:
        return CheckResult(name, FAIL, "tidak ditemukan di PATH", install_hint(tool_hint))
    try:
        line = _run_version(exe)
    except (subprocess.SubprocessError, OSError) as e:
        return CheckResult(name, FAIL, f"gagal dijalankan: {e}", install_hint(tool_hint))
    return CheckResult(name, OK, line or exe)


def check_deno() -> CheckResult:
    """Deno dipakai yt-dlp untuk challenge JS YouTube. Opsional: upload file tetap jalan."""
    exe = shutil.which("deno")
    if not exe:
        return CheckResult("Deno", WARN, "tidak ditemukan (link YouTube bisa gagal; upload tetap jalan)",
                           install_hint("deno"))
    try:
        line = _run_version(exe, "--version")
    except (subprocess.SubprocessError, OSError):
        line = exe
    ver = parse_version(line)
    if ver and ver < (2, 3):
        return CheckResult("Deno", WARN, f"{line} (butuh >= 2.3)", install_hint("deno"))
    return CheckResult("Deno", OK, line)


def check_module(display: str, module: str, required: bool = True) -> CheckResult:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        st = FAIL if required else WARN
        return CheckResult(display, st, "belum terpasang", "pip install -r requirements.txt")
    ver = getattr(mod, "__version__", None)
    if ver is None and module == "yt_dlp":
        ver = getattr(importlib.import_module("yt_dlp.version"), "__version__", None)
    return CheckResult(display, OK, str(ver or "terpasang"))


def check_ollama() -> list[CheckResult]:
    exe = shutil.which("ollama")
    results = []
    if not exe:
        results.append(CheckResult("Ollama", FAIL, "tidak ditemukan di PATH", install_hint("ollama")))
        return results
    try:
        line = _run_version(exe, "--version")
    except (subprocess.SubprocessError, OSError):
        line = exe
    results.append(CheckResult("Ollama", OK, line))
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/version", timeout=3) as r:
            ver = json.loads(r.read().decode()).get("version", "?")
        results.append(CheckResult("Ollama server", OK, f"jalan di {config.OLLAMA_URL} (v{ver})"))
    except (urllib.error.URLError, OSError, ValueError):
        results.append(CheckResult(
            "Ollama server", WARN, f"tidak merespons di {config.OLLAMA_URL}",
            "Jalankan aplikasi Ollama atau `ollama serve`",
        ))
    return results


def check_gpu() -> list[CheckResult]:
    if sys.platform == "darwin":
        return [CheckResult("GPU", WARN, "macOS: hanya mode CPU", "Pakai --device cpu")]
    info = device.query_nvidia_smi()
    if info is None:
        return [CheckResult("GPU", WARN, "GPU NVIDIA tidak terdeteksi, mode GPU nonaktif",
                            "Pakai --device cpu")]
    res = [CheckResult("GPU", OK, f"{info.name}, {info.vram_mb} MiB VRAM")]
    if info.vram_mb < 4096:
        res[0].status = WARN
        res[0].hint = "VRAM < 4 GB: large-v3 kemungkinan OOM, akan turun ke turbo/CPU"
    n = device.cuda_device_count()
    if n > 0:
        res.append(CheckResult("CUDA (ctranslate2)", OK, f"{n} device siap"))
    else:
        hint = "pip install -r requirements-gpu.txt"
        if sys.platform.startswith("linux"):
            dirs = device.nvidia_pip_lib_dirs()
            if dirs:
                hint += "; lalu export LD_LIBRARY_PATH=" + ":".join(map(str, dirs))
        res.append(CheckResult("CUDA (ctranslate2)", WARN,
                               "ctranslate2 belum bisa memakai GPU", hint))
    return res


def parse_nvenc_driver(stderr: str) -> Optional[str]:
    """'minimum required Nvidia driver for nvenc is 610.00' -> '610.00'."""
    m = re.search(r"minimum required nvidia driver for nvenc is ([\d.]+)", stderr or "", re.I)
    return m.group(1) if m else None


def check_nvenc() -> Optional[CheckResult]:
    """Coba encode 1 frame dengan h264_nvenc. None jika tidak relevan (tanpa GPU/ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if not exe or sys.platform == "darwin" or device.query_nvidia_smi() is None:
        return None
    cmd = [exe, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.1",
           "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError) as e:
        return CheckResult("NVENC (hardsub)", WARN, f"tidak bisa dites: {e}")
    if r.returncode == 0:
        return CheckResult("NVENC (hardsub)", OK, "h264_nvenc siap")
    need = parse_nvenc_driver(r.stderr)
    hint = (f"Update driver NVIDIA ke >= {need} (dibutuhkan FFmpeg versi ini)" if need
            else "Update driver NVIDIA")
    return CheckResult("NVENC (hardsub)", WARN, "tidak bisa dipakai, hardsub GPU akan pakai libx264 (CPU)", hint)


def total_ram_gb() -> Optional[float]:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1024**3
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024**2
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5).stdout
            return int(out.strip()) / 1024**3
    except Exception:
        return None
    return None


def check_pyannote() -> Optional[CheckResult]:
    """Opsional: hanya tampil jika pyannote.audio terpasang (extra A/B diarization)."""
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return None
    from backend.pipeline.diarize_pyannote import _hf_token, _load_dotenv

    cloned = config.PYANNOTE_DIR.is_dir() and any(config.PYANNOTE_DIR.iterdir())
    _load_dotenv()
    if cloned:
        return CheckResult("pyannote (opsional)", OK, "terpasang, model lokal siap (offline)")
    if _hf_token():
        return CheckResult("pyannote (opsional)", OK, "terpasang, HF_TOKEN tersedia")
    return CheckResult("pyannote (opsional)", WARN, "terpasang, tapi model belum ada & HF_TOKEN kosong",
                       "set HF_TOKEN (mis. di .env) atau clone model; lihat README")


def check_ram() -> CheckResult:
    gb = total_ram_gb()
    if gb is None:
        return CheckResult("RAM", WARN, "tidak bisa dibaca")
    st = ram_status(gb)
    hint = "" if st == OK else f"Disarankan >= {config.RECOMMENDED_RAM_GB} GB"
    return CheckResult("RAM", st, f"{gb:.1f} GB", hint)


def check_disk() -> CheckResult:
    free_gb = shutil.disk_usage(config.ROOT_DIR).free / 1024**3
    st = disk_status(free_gb)
    hint = "" if st == OK else f"Butuh >= {config.MIN_FREE_DISK_GB} GB kosong"
    return CheckResult("Disk", st, f"{free_gb:.1f} GB kosong di {config.ROOT_DIR.anchor}", hint)


CHECKS: list[Callable[[], object]] = [
    check_python,
    lambda: check_binary("ffmpeg", "ffmpeg"),
    lambda: check_binary("ffprobe", "ffmpeg"),
    lambda: check_module("yt-dlp", "yt_dlp"),
    lambda: check_module("yt-dlp-ejs", "yt_dlp_ejs", required=False),
    check_deno,
    lambda: check_module("faster-whisper", "faster_whisper"),
    lambda: check_module("sherpa-onnx", "sherpa_onnx"),
    lambda: check_module("fastapi", "fastapi", required=False),
    lambda: check_module("uvicorn", "uvicorn", required=False),
    check_pyannote,
    check_ollama,
    check_gpu,
    check_nvenc,
    check_ram,
    check_disk,
]


def run_all() -> list[CheckResult]:
    results: list[CheckResult] = []
    for fn in CHECKS:
        r = fn()
        if r is None:
            continue
        results.extend(r if isinstance(r, list) else [r])
    return results


def print_report(results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{r.status:<4}] {r.name:<{width}}  {r.detail}")
        if r.hint:
            print(f"        {'':<{width}}  -> {r.hint}")
    total = overall(results)
    gpu_ok = any(r.name == "CUDA (ctranslate2)" and r.status == OK for r in results)
    print()
    print(f"Mode tersedia: CPU{' + GPU' if gpu_ok else ''}")
    print(f"Status: {total}")
    return total

"""Download video YouTube ke folder job (maks 720p).

Keamanan:
- URL user hanya dipakai untuk mengambil video ID (11 karakter). yt-dlp
  menerima URL kanonik buatan sendiri, jadi parameter lain (list=, dsb.)
  dan trik domain tidak pernah sampai ke yt-dlp.
- Metadata dicek dulu (playlist, live, durasi) sebelum download dimulai.
- yt-dlp dipakai lewat Python API: file config user tidak ikut dibaca,
  tanpa cookies, tanpa subtitle/thumbnail, batas ukuran file.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlsplit

from backend import config
from backend.errors import InputRejected, ToolError
from backend.jobs import Job
from backend.pipeline import probe as probe_mod

from dataclasses import dataclass

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
SUPPORTED_BROWSERS = {"chrome", "chromium", "edge", "firefox", "brave", "opera", "vivaldi", "safari"}


@dataclass
class Cookies:
    """Sumber cookie untuk yt-dlp (mengatasi blokir 'confirm you're not a bot')."""
    from_browser: Optional[str] = None   # mis. "chrome", "firefox", "edge"
    file: Optional[str] = None           # path cookies.txt (format Netscape)

    @classmethod
    def parse(cls, from_browser: Optional[str], file: Optional[str]) -> Optional["Cookies"]:
        if from_browser:
            b = from_browser.strip().lower()
            if b not in SUPPORTED_BROWSERS:
                raise InputRejected(f"browser '{from_browser}' tidak didukung "
                                    f"(pilihan: {', '.join(sorted(SUPPORTED_BROWSERS))})")
            return cls(from_browser=b)
        if file:
            if not Path(file).is_file():
                raise InputRejected(f"file cookies tidak ditemukan: {file}")
            return cls(file=file)
        return None
SHORT_HOSTS = {"youtu.be", "www.youtu.be"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")
MAX_URL_LEN = 2048

ToS_WARNING = (
    "Peringatan: mengunduh video YouTube dapat melanggar Ketentuan Layanan YouTube, "
    "dan isinya biasanya berhak cipta. Gunakan hanya untuk keperluan pribadi; "
    "jalur utama VidNote adalah upload file."
)


# ---------- validasi URL (murni) ----------

def normalize_url(url: str) -> tuple[str, str]:
    """Validasi URL YouTube. Kembalikan (url_kanonik, video_id) atau raise InputRejected."""
    url = (url or "").strip()
    if not url or len(url) > MAX_URL_LEN:
        raise InputRejected("URL kosong atau terlalu panjang")
    try:
        p = urlsplit(url)
        port = p.port
    except ValueError:
        raise InputRejected("URL tidak valid")
    if p.scheme.lower() not in ("http", "https"):
        raise InputRejected("URL harus diawali https://")
    if p.username or p.password:
        raise InputRejected("URL tidak boleh berisi user/password")
    if port not in (None, 80, 443):
        raise InputRejected("URL tidak boleh memakai port khusus")
    host = (p.hostname or "").lower().rstrip(".")

    vid = ""
    if host in SHORT_HOSTS:
        vid = p.path.lstrip("/").split("/")[0]
    elif host in YOUTUBE_HOSTS:
        path = p.path.rstrip("/")
        if path == "/watch":
            vid = (parse_qs(p.query).get("v") or [""])[0]
        elif path == "/playlist":
            raise InputRejected("link playlist tidak didukung, pakai link satu video")
        else:
            for prefix in _PATH_PREFIXES:
                if path.startswith(prefix):
                    vid = path[len(prefix):].split("/")[0]
                    break
            else:
                raise InputRejected("bukan link video YouTube (channel/halaman lain tidak didukung)")
    else:
        raise InputRejected("hanya link youtube.com atau youtu.be yang didukung")

    if not _VIDEO_ID.match(vid):
        raise InputRejected("video ID pada link tidak valid")
    return f"https://www.youtube.com/watch?v={vid}", vid


# ---------- cek metadata (murni) ----------

def check_info(info: dict) -> list[str]:
    errs: list[str] = []
    if info.get("_type") in ("playlist", "multi_video"):
        errs.append("link berisi playlist, bukan satu video")
        return errs
    live = info.get("live_status")
    if info.get("is_live") or live in ("is_live", "is_upcoming", "post_live"):
        errs.append("live stream / premiere belum selesai tidak didukung")
    dur = info.get("duration")
    if not dur:
        errs.append("durasi video tidak diketahui")
    elif dur > config.MAX_DURATION_SEC:
        errs.append(
            f"durasi {probe_mod.fmt_duration(dur)} melebihi batas "
            f"{probe_mod.fmt_duration(config.MAX_DURATION_SEC)}"
        )
    return errs


# ---------- yt-dlp ----------

def format_selector(max_h: int = config.YOUTUBE_MAX_HEIGHT) -> str:
    # Prioritas H.264 + AAC (ramah browser & cepat di-burn), lalu apa saja <= max_h.
    return (
        f"bv*[height<={max_h}][vcodec^=avc1]+ba[ext=m4a]/"
        f"bv*[height<={max_h}][ext=mp4]+ba[ext=m4a]/"
        f"b[height<={max_h}][ext=mp4]/"
        f"bv*[height<={max_h}]+ba/"
        f"b[height<={max_h}]"
    )


class _Logger:
    """Teruskan pesan yt-dlp tanpa membanjiri terminal."""

    def __init__(self):
        self.errors: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        self.errors.append(msg)


def _ydl_opts(job: Job, logger: _Logger, on_progress: Optional[Callable[[dict], None]],
              cookies: Optional["Cookies"] = None) -> dict:
    opts = {
        "format": format_selector(),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "outtmpl": {"default": "download.%(ext)s"},
        "paths": {"home": str(job.dir), "temp": str(job.dir)},
        "max_filesize": config.MAX_UPLOAD_BYTES,
        "socket_timeout": 30,
        "retries": 3,
        "cachedir": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writethumbnail": False,
        "writeinfojson": False,
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": logger,
        "progress_hooks": [on_progress] if on_progress else [],
    }
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    if cookies and cookies.from_browser:
        opts["cookiesfrombrowser"] = (cookies.from_browser,)
    elif cookies and cookies.file:
        opts["cookiefile"] = cookies.file
    return opts


def _clean(msg: str) -> str:
    return re.sub(r"^(ERROR:\s*)?(\[[^\]]+\]\s*)?([A-Za-z0-9_-]{11}:\s*)?", "", str(msg)).strip()


def _friendly_error(e, cookies: Optional["Cookies"]) -> str:
    raw = _clean(e.msg if hasattr(e, "msg") else e)
    low = raw.lower()
    if "could not copy" in low and "cookie" in low:
        return (f"gagal membaca cookie dari browser '{cookies.from_browser if cookies else '?'}'. "
                "Tutup browser itu dulu, lalu coba lagi (database cookie terkunci saat browser terbuka).")
    if "sign in to confirm" in low or "not a bot" in low:
        return ("YouTube meminta verifikasi bot. Coba lagi nanti, atau pakai "
                "--cookies-from-browser <browser> (tutup browser-nya dulu) atau --cookies-file cookies.txt.")
    return f"gagal membaca info video: {raw}"


def fetch_url(url: str, job: Job,
              on_progress: Optional[Callable[[dict], None]] = None,
              cookies: Optional["Cookies"] = None) -> tuple[Path, probe_mod.ProbeInfo, dict]:
    """Validasi + download + probe. Kembalikan (path_input, probe_info, meta)."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    canonical, vid = normalize_url(url)
    logger = _Logger()
    with yt_dlp.YoutubeDL(_ydl_opts(job, logger, on_progress, cookies)) as ydl:
        try:
            info = ydl.extract_info(canonical, download=False)
        except DownloadError as e:
            raise ToolError(_friendly_error(e, cookies))
        errs = check_info(info or {})
        if errs:
            raise InputRejected(errs)
        try:
            info = ydl.process_ie_result(info, download=True)
        except DownloadError as e:
            raise ToolError(f"download gagal: {_clean(e.msg if hasattr(e, 'msg') else e)}")

    files = [p for p in job.dir.glob("download.*") if p.suffix.lower() in config.ALLOWED_EXTENSIONS]
    if not files:
        detail = _clean(logger.errors[-1]) if logger.errors else "file hasil tidak ditemukan"
        raise ToolError(f"download gagal: {detail}")
    src = max(files, key=lambda p: p.stat().st_size)
    dst = job.path("input" + src.suffix.lower())
    src.replace(dst)

    meta = {
        "video_id": vid,
        "url": canonical,
        "title": str(info.get("title") or ""),
        "uploader": str(info.get("uploader") or ""),
        "duration": info.get("duration"),
        "height": info.get("height"),
    }
    return dst, probe_mod.check_file(dst), meta

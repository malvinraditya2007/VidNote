"""Exception VidNote. Pesan ditujukan langsung ke user (bahasa Indonesia)."""
from __future__ import annotations


class VidNoteError(Exception):
    """Error umum yang aman ditampilkan ke user."""


class InputRejected(VidNoteError):
    """Input ditolak validasi (ekstensi, ukuran, durasi, stream, dll.)."""

    def __init__(self, reasons: list[str] | str, info: object = None):
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        self.info = info  # ProbeInfo jika sudah sempat di-probe
        super().__init__("; ".join(self.reasons))


class ToolError(VidNoteError):
    """Program eksternal (ffmpeg/ffprobe/yt-dlp) gagal atau timeout."""

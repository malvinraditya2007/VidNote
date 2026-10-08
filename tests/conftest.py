"""Fixture video sintetis dibuat dengan ffmpeg (testsrc + sine)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe tidak ada di PATH"
)


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", *args], check=True, timeout=60)


def make_video(out: Path, w: int = 320, h: int = 180, dur: float = 2,
               audio: bool = True, codec: str = "libx264") -> Path:
    args = ["-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=15:duration={dur}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
    args += ["-c:v", codec, "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "libopus" if out.suffix == ".webm" else "aac", "-shortest"]
    _ffmpeg(*args, str(out))
    return out


@pytest.fixture(scope="session")
def videos(tmp_path_factory) -> dict[str, Path]:
    if FFMPEG is None:
        pytest.skip("ffmpeg tidak ada")
    d = tmp_path_factory.mktemp("videos")
    v = {
        "landscape": make_video(d / "landscape.mp4", 320, 180),
        "portrait": make_video(d / "portrait.mp4", 180, 320),
        "no_audio": make_video(d / "no_audio.mp4", audio=False),
        "mkv": make_video(d / "clip.mkv"),
        "webm": make_video(d / "clip.webm", codec="libvpx-vp9"),
        "tiny": make_video(d / "tiny.mp4", 32, 32),
    }
    # Landscape yang diberi metadata rotasi 90° -> tampil portrait
    rotated = d / "rotated.mp4"
    _ffmpeg("-display_rotation", "90", "-i", str(v["landscape"]), "-c", "copy", str(rotated))
    v["rotated"] = rotated
    # File teks yang diberi ekstensi .mp4
    fake = d / "fake.mp4"
    fake.write_text("ini bukan video\n" * 100)
    v["fake"] = fake
    # Playlist HLS yang menyamar jadi .mp4 (tidak boleh memicu akses jaringan)
    m3u8 = d / "playlist.mp4"
    m3u8.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://127.0.0.1:9/a.ts\n#EXT-X-ENDLIST\n")
    v["m3u8"] = m3u8
    # Audio saja dalam .mp4
    audio_only = d / "audio_only.mp4"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(audio_only))
    v["audio_only"] = audio_only
    return v

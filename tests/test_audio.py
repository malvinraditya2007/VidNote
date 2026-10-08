from __future__ import annotations

import subprocess

import pytest

from backend import config, jobs
from backend.errors import ToolError
from backend.pipeline.audio import build_ffmpeg_cmd, extract_audio
from backend.pipeline.receive_file import copy_into_job
from tests.conftest import FFMPEG, needs_ffmpeg


def _video(path, audio_src: str, dur: float = 3):
    subprocess.run([
        FFMPEG, "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=10:duration={dur}",
        "-f", "lavfi", "-i", audio_src,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-ar", "48000",
        "-t", str(dur), str(path),
    ], check=True, timeout=60)
    return path


def _in_job(src, tmp_path):
    job = jobs.create_job(tmp_path / "work")
    return job, copy_into_job(src, job, src.suffix)


def test_cmd_is_safe():
    cmd = build_ffmpeg_cmd("ffmpeg", "input.mp4", "audio.wav")
    assert "-protocol_whitelist" in cmd and "file:input.mp4" in cmd
    assert cmd[-1] == "file:audio.wav"
    assert "-ar" in cmd and str(config.AUDIO_SAMPLE_RATE) in cmd


@needs_ffmpeg
def test_extract_format_and_duration(tmp_path):
    src = _video(tmp_path / "v.mp4", "sine=frequency=440:sample_rate=48000:duration=3")
    job, inp = _in_job(src, tmp_path)
    a = extract_audio(inp, job, 3)
    assert a.path == job.path("audio.wav")
    assert (a.sample_rate, a.channels) == (16000, 1)
    assert abs(a.duration - 3.0) < 0.15
    assert not a.is_near_silent


@needs_ffmpeg
def test_quiet_audio_is_normalized_louder(tmp_path):
    src = _video(tmp_path / "q.mp4", "sine=frequency=300:duration=3,volume=0.02")
    job, inp = _in_job(src, tmp_path)
    a = extract_audio(inp, job, 3)
    # sinus volume 0.02 ~ -37 dBFS RMS; setelah loudnorm (-16 LUFS) jauh lebih keras
    assert a.rms_dbfs > -25


@needs_ffmpeg
def test_silence_ok_and_flagged(tmp_path):
    src = _video(tmp_path / "s.mp4", "anullsrc=r=48000:cl=stereo")
    job, inp = _in_job(src, tmp_path)
    a = extract_audio(inp, job, 3)
    assert a.is_near_silent
    assert abs(a.duration - 3.0) < 0.15


@needs_ffmpeg
def test_no_audio_stream_raises(videos, tmp_path):
    job, inp = _in_job(videos["no_audio"], tmp_path)
    with pytest.raises(ToolError, match="ekstraksi audio gagal"):
        extract_audio(inp, job, 2)
    assert not job.path("audio.wav").exists()


def test_input_outside_job_rejected(tmp_path):
    job = jobs.create_job(tmp_path / "work")
    outside = tmp_path / "x.mp4"
    outside.write_bytes(b"x")
    with pytest.raises((ValueError, ToolError)):
        extract_audio(outside, job, 1)

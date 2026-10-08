import json
import subprocess

import pytest

from backend import config, device, jobs
from backend.pipeline import burn as B
from backend.pipeline import subtitle as S
from backend.pipeline.receive_file import copy_into_job
from tests.conftest import FFMPEG, needs_ffmpeg

CUES = [{"id": 0, "start": 0.2, "end": 1.5, "speaker": 0, "text": "Halo {\\an8} dunia"},
        {"id": 1, "start": 1.6, "end": 2.0, "speaker": 1, "text": "Hai juga"}]


def test_progress_parse():
    assert B.parse_progress_line("out_time_us=2500000\n") == 2.5
    assert B.parse_progress_line("out_time_ms=1000000") == 1.0
    assert B.parse_progress_line("out_time_us=N/A") is None
    assert B.parse_progress_line("progress=end") is None


def test_cmd_audio_copy_vs_encode():
    c = B.build_cmd("ffmpeg", "input.mp4", "libx264", "aac")
    assert c[c.index("-c:a") + 1] == "copy"
    c = B.build_cmd("ffmpeg", "input.webm", "h264_nvenc", "opus")
    assert c[c.index("-c:a") + 1] == "aac" and "h264_nvenc" in c
    assert "-protocol_whitelist" in c and c[-1] == "file:" + config.BURN_OUTPUT
    assert "ass=subs.ass" in c[c.index("-vf") + 1]


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def _setup(videos, tmp_path, key, orientation):
    job = jobs.create_job(tmp_path / "work")
    inp = copy_into_job(videos[key], job, videos[key].suffix)
    job.path("subs.ass").write_text(S.to_ass(CUES, orientation, 2), encoding="utf-8")
    return job, inp


@needs_ffmpeg
@pytest.mark.parametrize("key,orientation,acodec", [
    ("landscape", "landscape", "aac"), ("portrait", "portrait", "aac"),
    ("webm", "landscape", "opus"), ("rotated", "portrait", "aac"),
])
def test_burn_cpu(videos, tmp_path, key, orientation, acodec):
    job, inp = _setup(videos, tmp_path, key, orientation)
    res = B.burn(inp, job, "cpu", 2.0, acodec)
    st = _probe(res.path)
    assert res.encoder == "libx264" and not res.fell_back
    assert st["video"]["codec_name"] == "h264" and st["video"]["pix_fmt"] == "yuv420p"
    assert st["audio"]["codec_name"] == "aac"
    w, h = st["video"]["width"], st["video"]["height"]
    assert (h > w) == (orientation == "portrait")


@needs_ffmpeg
@pytest.mark.skipif(not device.gpu_available(), reason="GPU NVIDIA tidak ada")
def test_burn_gpu_nvenc(videos, tmp_path):
    job, inp = _setup(videos, tmp_path, "landscape", "landscape")
    res = B.burn(inp, job, "gpu", 2.0, "aac")
    assert _probe(res.path)["video"]["codec_name"] == "h264"
    assert res.encoder in ("h264_nvenc", "libx264")


@needs_ffmpeg
def test_burn_missing_subs(videos, tmp_path):
    job = jobs.create_job(tmp_path / "work")
    inp = copy_into_job(videos["landscape"], job, ".mp4")
    with pytest.raises(Exception, match="subs.ass"):
        B.burn(inp, job, "cpu", 2.0, "aac")


NVENC_ERR = """[h264_nvenc @ 0x1] Driver does not support the required nvenc API version. Required: 13.1 Found: 12.2
[h264_nvenc @ 0x1] The minimum required Nvidia driver for nvenc is 610.00 or newer
[out#0/mp4 @ 0x2] Nothing was written into output file, because at least one of its streams received no packets."""


def test_summarize_error_prefers_driver_line():
    assert B.summarize_error(NVENC_ERR).startswith("Driver does not support")
    assert B.summarize_error("[x] a\n[y] final error") == "final error"
    assert B.summarize_error("") == ""


def test_parse_nvenc_driver():
    from backend.doctor import parse_nvenc_driver

    assert parse_nvenc_driver(NVENC_ERR) == "610.00"
    assert parse_nvenc_driver("other") is None

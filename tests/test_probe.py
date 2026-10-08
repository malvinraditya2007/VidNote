from __future__ import annotations

import os
import time

import pytest

from backend import config, jobs
from backend.errors import InputRejected
from backend.pipeline import probe
from backend.pipeline.receive_file import check_source, receive_file
from tests.conftest import needs_ffmpeg


def _data(w=1920, h=1080, dur="60.0", rotation=None, rotate_tag=None, audio=True,
          fmt="mov,mp4,m4a,3gp,3g2,mj2", attached_pic=False):
    v = {"codec_type": "video", "codec_name": "h264", "width": w, "height": h,
         "disposition": {"attached_pic": 1 if attached_pic else 0}}
    if rotation is not None:
        v["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": rotation}]
    if rotate_tag is not None:
        v["tags"] = {"rotate": rotate_tag}
    streams = [v]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    return {"streams": streams, "format": {"duration": dur, "format_name": fmt, "size": "1000"}}


# ---------- fungsi murni ----------

def test_parse_landscape():
    info = probe.parse_probe(_data())
    assert (info.width, info.height, info.orientation) == (1920, 1080, "landscape")
    assert info.has_audio and info.has_video
    assert probe.validate(info) == []


def test_parse_rotation_side_data_and_tag():
    for d in (_data(rotation=-90), _data(rotation=90), _data(rotate_tag="270")):
        info = probe.parse_probe(d)
        assert (info.width, info.height, info.orientation) == (1080, 1920, "portrait")
    info = probe.parse_probe(_data(rotation=180))
    assert info.rotation == 180 and info.orientation == "landscape"


def test_attached_pic_is_not_video():
    info = probe.parse_probe(_data(attached_pic=True))
    assert not info.has_video
    assert "tidak ada stream video" in probe.validate(info)


def test_validate_duration_limits():
    too_long = probe.parse_probe(_data(dur=str(config.MAX_DURATION_SEC + 1)))
    assert any("melebihi batas" in e for e in probe.validate(too_long))
    exact = probe.parse_probe(_data(dur=str(config.MAX_DURATION_SEC)))
    assert probe.validate(exact) == []
    missing = probe.parse_probe(_data(dur="N/A"))
    assert any("durasi" in e for e in probe.validate(missing))


def test_validate_resolution_and_container():
    assert any("terlalu besar" in e for e in probe.validate(probe.parse_probe(_data(8000, 4000))))
    assert any("terlalu kecil" in e for e in probe.validate(probe.parse_probe(_data(32, 32))))
    assert any("format" in e for e in probe.validate(probe.parse_probe(_data(fmt="hls"))))
    assert probe.validate(probe.parse_probe(_data(fmt="matroska,webm"))) == []


def test_validate_no_audio():
    assert "tidak ada stream audio" in probe.validate(probe.parse_probe(_data(audio=False)))


def test_fmt_duration():
    assert probe.fmt_duration(65) == "01:05"
    assert probe.fmt_duration(3725) == "1:02:05"


def test_check_source_extension_and_size(tmp_path):
    bad = tmp_path / "a.avi"
    bad.write_bytes(b"x")
    with pytest.raises(InputRejected, match="ekstensi"):
        check_source(bad)
    empty = tmp_path / "a.mp4"
    empty.write_bytes(b"")
    with pytest.raises(InputRejected, match="kosong"):
        check_source(empty)
    with pytest.raises(InputRejected, match="tidak ditemukan"):
        check_source(tmp_path / "nope.mp4")
    upper = tmp_path / "A.MP4"
    upper.write_bytes(b"x")
    assert check_source(upper) == ".mp4"


# ---------- folder kerja ----------

def test_job_context_cleans_up_on_error(tmp_path):
    with pytest.raises(RuntimeError):
        with jobs.job_context(work_dir=tmp_path) as job:
            job.path("x.txt").write_text("x")
            raise RuntimeError("boom")
    assert not job.dir.exists()


def test_job_context_keep(tmp_path):
    with jobs.job_context(keep=True, work_dir=tmp_path) as job:
        pass
    assert job.dir.exists()


def test_job_path_rejects_traversal(tmp_path):
    job = jobs.create_job(tmp_path)
    for bad in ("../x", "..", "a/b", "", "..\\x"):
        with pytest.raises(ValueError):
            job.path(bad)


def test_sweep_only_stale_uuid_dirs(tmp_path):
    old = jobs.create_job(tmp_path)
    fresh = jobs.create_job(tmp_path)
    other = tmp_path / "jangan-dihapus"
    other.mkdir()
    past = time.time() - (config.STALE_JOB_HOURS + 1) * 3600
    os.utime(old.dir, (past, past))
    os.utime(other, (past, past))
    assert jobs.sweep_stale(tmp_path) == 1
    assert not old.dir.exists() and fresh.dir.exists() and other.exists()


# ---------- integrasi dengan ffprobe ----------

@needs_ffmpeg
@pytest.mark.parametrize("key,orientation", [
    ("landscape", "landscape"), ("portrait", "portrait"), ("rotated", "portrait"),
    ("mkv", "landscape"), ("webm", "landscape"),
])
def test_receive_valid(videos, tmp_path, key, orientation):
    job = jobs.create_job(tmp_path)
    dst, info = receive_file(videos[key], job)
    assert dst.parent == job.dir and dst.stem == "input"
    assert info.orientation == orientation
    assert info.has_audio and info.has_video
    assert 1.5 <= info.duration <= 2.5


@needs_ffmpeg
@pytest.mark.parametrize("key,reason", [
    ("no_audio", "audio"),
    ("fake", "tidak bisa dibaca"),
    ("m3u8", ""),
    ("audio_only", "video"),
    ("tiny", "terlalu kecil"),
])
def test_receive_rejected(videos, tmp_path, key, reason):
    job = jobs.create_job(tmp_path)
    with pytest.raises(InputRejected) as ei:
        receive_file(videos[key], job)
    assert reason in str(ei.value)


@needs_ffmpeg
def test_original_name_ignored(videos, tmp_path):
    src = tmp_path / "..evil name; rm -rf.mp4"
    src.write_bytes(videos["landscape"].read_bytes())
    job = jobs.create_job(tmp_path / "work")
    dst, _ = receive_file(src, job)
    assert dst.name == "input.mp4"

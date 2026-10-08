import pytest

from backend import config, models_dl, runner
from backend.errors import InputRejected, VidNoteError


def test_safe_name():
    assert runner.safe_name("Tips Jawab: \"Ceritakan\" | Ruang HRD!") == "tips-jawab-ceritakan-ruang-hrd"
    assert runner.safe_name("..\\..\\etc/passwd") == "etc-passwd"
    assert runner.safe_name("日本語") == "video"
    assert len(runner.safe_name("a" * 200)) <= 60


def test_estimate_and_disk():
    lo, hi = runner.estimate_minutes(30 * 60, "gpu")
    assert 1 <= lo < hi
    assert runner.estimate_minutes(10, "cpu") == (1, 1)
    assert runner.required_disk_bytes(1024**3) == 4 * 1024**3


def test_check_disk_rejects(tmp_path):
    with pytest.raises(InputRejected, match="disk"):
        runner.check_disk(tmp_path, 10**18)


def test_is_url():
    assert runner.is_url(" https://youtu.be/x") and not runner.is_url("D:\\video.mp4")


def test_offline_blocks_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)
    with pytest.raises(VidNoteError, match="offline"):
        models_dl.download("https://github.com/x", tmp_path / "m.onnx", "0" * 64)
    with pytest.raises(VidNoteError, match="offline"):
        runner.run("https://youtu.be/dQw4w9WgXcQ", runner.RunOptions(device="cpu"))
    with pytest.raises(VidNoteError, match="offline"):
        runner.download_models(["cpu"])

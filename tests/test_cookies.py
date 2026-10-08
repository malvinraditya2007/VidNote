import pytest

from backend.errors import InputRejected
from backend.pipeline.fetch_url import Cookies, _Logger, _ydl_opts


def test_parse_browser():
    c = Cookies.parse("Chrome", None)
    assert c.from_browser == "chrome" and c.file is None


def test_parse_browser_invalid():
    with pytest.raises(InputRejected, match="tidak didukung"):
        Cookies.parse("safari-beta", None)


def test_parse_file_missing():
    with pytest.raises(InputRejected, match="tidak ditemukan"):
        Cookies.parse(None, "C:/nope/cookies.txt")


def test_parse_file_ok(tmp_path):
    f = tmp_path / "cookies.txt"
    f.write_text("# Netscape HTTP Cookie File\n")
    c = Cookies.parse(None, str(f))
    assert c.file == str(f) and c.from_browser is None


def test_parse_none():
    assert Cookies.parse(None, None) is None


def test_browser_wins_over_file(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("x")
    c = Cookies.parse("firefox", str(f))
    assert c.from_browser == "firefox" and c.file is None


class _Job:
    def __init__(self, d):
        self.dir = d


def test_ydl_opts_inject_cookies(tmp_path):
    job = _Job(tmp_path)
    o = _ydl_opts(job, _Logger(), None, Cookies(from_browser="edge"))
    assert o["cookiesfrombrowser"] == ("edge",)
    o2 = _ydl_opts(job, _Logger(), None, Cookies(file="c.txt"))
    assert o2["cookiefile"] == "c.txt" and "cookiesfrombrowser" not in o2
    o3 = _ydl_opts(job, _Logger(), None, None)
    assert "cookiesfrombrowser" not in o3 and "cookiefile" not in o3


def test_friendly_error_messages():
    from backend.pipeline.fetch_url import _friendly_error

    class E:
        def __init__(self, m): self.msg = m

    locked = _friendly_error(E("ERROR: Could not copy Chrome cookie database"), Cookies(from_browser="edge"))
    assert "Tutup browser" in locked and "edge" in locked
    bot = _friendly_error(E("ERROR: Sign in to confirm you're not a bot"), None)
    assert "verifikasi bot" in bot and "--cookies" in bot
    other = _friendly_error(E("ERROR: Video unavailable"), None)
    assert "Video unavailable" in other

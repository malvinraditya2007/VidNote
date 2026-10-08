import pytest

from backend import config
from backend.errors import InputRejected
from backend.pipeline.fetch_url import check_info, format_selector, normalize_url

VID = "dQw4w9WgXcQ"
CANON = f"https://www.youtube.com/watch?v={VID}"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtube.com/watch?v={VID}&t=42s",
    f"http://m.youtube.com/watch?v={VID}",
    f"https://www.youtube.com/watch?v={VID}&list=PL1234567890",  # list= diabaikan
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?si=abc&t=10",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}",
    f"https://music.youtube.com/watch?v={VID}",
    f"  https://WWW.YouTube.com/watch?v={VID}  ",
    f"https://www.youtube.com:443/watch?v={VID}",
])
def test_valid_urls_become_canonical(url):
    assert normalize_url(url) == (CANON, VID)


@pytest.mark.parametrize("url,reason", [
    ("", "kosong"),
    ("x" * 3000, "terlalu panjang"),
    (f"ftp://youtube.com/watch?v={VID}", "https"),
    (f"file:///C:/watch?v={VID}", "https"),
    (f"javascript:alert(1)//youtube.com/watch?v={VID}", "https"),
    (f"https://youtube.com.evil.com/watch?v={VID}", "youtube.com"),
    (f"https://evil.com/youtube.com/watch?v={VID}", "youtube.com"),
    (f"https://notyoutube.com/watch?v={VID}", "youtube.com"),
    (f"https://user:pw@youtube.com/watch?v={VID}", "user"),
    (f"https://youtube.com\\@evil.com/watch?v={VID}", ""),
    (f"https://youtube.com:8080/watch?v={VID}", "port"),
    ("https://www.youtube.com/playlist?list=PL1234567890", "playlist"),
    ("https://www.youtube.com/@channel", "bukan link video"),
    ("https://www.youtube.com/channel/UC123", "bukan link video"),
    ("https://www.youtube.com/watch?v=short", "ID"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXc$", "ID"),
    ("https://youtu.be/", "ID"),
    ("https://example.com/video.m3u8", "youtube.com"),
])
def test_invalid_urls(url, reason):
    with pytest.raises(InputRejected) as ei:
        normalize_url(url)
    assert reason in str(ei.value)


def test_check_info_ok():
    assert check_info({"duration": 600, "live_status": "not_live"}) == []
    assert check_info({"duration": config.MAX_DURATION_SEC}) == []


@pytest.mark.parametrize("info,reason", [
    ({"_type": "playlist"}, "playlist"),
    ({"duration": 600, "is_live": True}, "live"),
    ({"duration": 600, "live_status": "is_upcoming"}, "live"),
    ({"duration": 600, "live_status": "post_live"}, "live"),
    ({"duration": None}, "tidak diketahui"),
    ({"duration": config.MAX_DURATION_SEC + 1}, "melebihi"),
])
def test_check_info_rejects(info, reason):
    errs = check_info(info)
    assert errs and any(reason in e for e in errs)


def test_format_selector_caps_height():
    sel = format_selector(720)
    assert "height<=720" in sel and "1080" not in sel

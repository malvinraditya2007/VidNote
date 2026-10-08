from backend import security as S


def test_host_ok():
    assert S.host_ok("127.0.0.1:8765", 8765)
    assert S.host_ok("localhost:8765", 8765)
    assert not S.host_ok("127.0.0.1:9999", 8765)      # port beda (DNS rebinding)
    assert not S.host_ok("evil.com:8765", 8765)
    assert not S.host_ok(None, 8765)


def test_origin_ok():
    assert S.origin_ok("http://127.0.0.1:8765", 8765)
    assert S.origin_ok("http://localhost:8765", 8765)
    assert not S.origin_ok("http://evil.com", 8765)
    assert not S.origin_ok("http://127.0.0.1:1234", 8765)
    assert not S.origin_ok(None, 8765)                 # Origin wajib untuk POST
    assert not S.origin_ok("null", 8765)


def test_token_ok():
    assert S.token_ok("abc", "abc")
    assert not S.token_ok("abc", "xyz")
    assert not S.token_ok(None, "abc")
    assert not S.token_ok("", "abc")


def test_is_public():
    assert S.is_public("/") and S.is_public("/health")
    assert not S.is_public("/system") and not S.is_public("/jobs/url")

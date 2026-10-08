import time

import pytest
from fastapi.testclient import TestClient

from backend import runner, security
from backend.app import create_app
from backend.jobqueue import JobManager

PORT = 8765
H = {"host": f"127.0.0.1:{PORT}", "x-vidnote-token": security.SESSION_TOKEN}
ORIGIN = {"origin": f"http://127.0.0.1:{PORT}"}


class FakeRunner:
    """Runner palsu: panggil callback lalu tulis file hasil minimal."""

    def __init__(self, tmp, delay=0.0, fail=None):
        self.tmp, self.delay, self.fail = tmp, delay, fail

    def __call__(self, src, opts, on_stage=None, on_progress=None, on_oom=None, on_info=None):
        if on_info:
            on_info({"title": "Demo", "duration": 60, "orientation": "landscape",
                     "width": 1280, "height": 720, "estimate_min": (1, 2)})
        for st in ("audio", "asr", "burn"):
            if on_stage:
                on_stage(st, runner.STAGE_LABEL[st])
            if on_progress:
                on_progress(st, 1.0)
            time.sleep(self.delay)
        if self.fail:
            from backend.errors import VidNoteError
            raise VidNoteError(self.fail)
        out = self.tmp / "demo-00000000-000000"
        out.mkdir(parents=True, exist_ok=True)
        (out / "transcript.json").write_text('{"num_speakers":2}', encoding="utf-8")
        (out / "summary.json").write_text('{"points":[]}', encoding="utf-8")
        (out / "subs.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhai\n", encoding="utf-8")
        (out / "video_hardsub.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42demo")
        return runner.RunResult(out, "Demo", sorted(p.name for p in out.iterdir()),
                                {"asr": 1.0}, [], {"num_speakers": 2, "language": "id", "models": {}})


@pytest.fixture
def client(tmp_path):
    mgr = JobManager(run_fn=FakeRunner(tmp_path))
    return TestClient(create_app(mgr, PORT))


def wait_done(c, jid, timeout=10):
    for _ in range(int(timeout * 20)):
        s = c.get(f"/jobs/{jid}", headers=H).json()
        if s["status"] != "running":
            return s
        time.sleep(0.05)
    raise AssertionError("job tidak selesai")


# ---------- keamanan ----------

def test_health_public(client):
    assert client.get("/health", headers={"host": f"127.0.0.1:{PORT}"}).status_code == 200


def test_bad_host_rejected(client):
    r = client.get("/system", headers={"host": "evil.com", "x-vidnote-token": security.SESSION_TOKEN})
    assert r.status_code == 400


def test_token_required(client):
    assert client.get("/system", headers={"host": f"127.0.0.1:{PORT}"}).status_code == 401
    assert client.get("/system", headers=H).status_code == 200


def test_post_requires_origin(client):
    r = client.post("/jobs/url", headers=H, json={"url": "x", "device": "cpu"})
    assert r.status_code == 403  # tanpa Origin
    r = client.post("/jobs/url", headers={**H, "origin": "http://evil.com"}, json={})
    assert r.status_code == 403


def test_csp_header(client):
    r = client.get("/system", headers=H)
    assert "default-src 'self'" in r.headers["content-security-policy"]


# ---------- lifecycle ----------

def test_system(client):
    j = client.get("/system", headers=H).json()
    assert j["token"] == security.SESSION_TOKEN and "cpu" in j["devices"]


def test_url_job_full_lifecycle(client):
    r = client.post("/jobs/url", headers={**H, **ORIGIN},
                    json={"url": "https://youtu.be/dQw4w9WgXcQ", "device": "cpu"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    res = client.get(f"/jobs/{jid}/result", headers=H).json()
    assert res["num_speakers"] == 2 and "subs.srt" in res["available"]
    assert client.get(f"/jobs/{jid}/srt", headers=H).status_code == 200
    assert client.get(f"/jobs/{jid}/video", headers=H).status_code == 200


def test_invalid_url_rejected(client):
    r = client.post("/jobs/url", headers={**H, **ORIGIN}, json={"url": "https://evil.com", "device": "cpu"})
    assert r.status_code == 422


def test_bad_device_rejected(client):
    r = client.post("/jobs/url", headers={**H, **ORIGIN},
                    json={"url": "https://youtu.be/dQw4w9WgXcQ", "device": "tpu"})
    assert r.status_code == 422


def test_one_job_at_a_time(tmp_path):
    mgr = JobManager(run_fn=FakeRunner(tmp_path, delay=0.3))
    c = TestClient(create_app(mgr, PORT))
    body = {"url": "https://youtu.be/dQw4w9WgXcQ", "device": "cpu"}
    assert c.post("/jobs/url", headers={**H, **ORIGIN}, json=body).status_code == 200
    assert c.post("/jobs/url", headers={**H, **ORIGIN}, json=body).status_code == 409


def test_result_before_done_409(client):
    r = client.post("/jobs/url", headers={**H, **ORIGIN},
                    json={"url": "https://youtu.be/dQw4w9WgXcQ", "device": "cpu"})
    jid = r.json()["job_id"]
    # langsung minta hasil (kemungkinan masih running) -> 409 atau sudah done
    rr = client.get(f"/jobs/{jid}/result", headers=H)
    assert rr.status_code in (200, 409)
    wait_done(client, jid)


def test_upload_rejects_bad_ext(client):
    r = client.post("/jobs/upload", headers={**H, **ORIGIN},
                    files={"file": ("x.txt", b"hello", "text/plain")}, data={"device": "cpu"})
    assert r.status_code == 422


def test_upload_job(client):
    r = client.post("/jobs/upload", headers={**H, **ORIGIN},
                    files={"file": ("clip.mp4", b"\x00" * 1024, "video/mp4")}, data={"device": "cpu"})
    assert r.status_code == 200
    assert wait_done(client, r.json()["job_id"])["status"] == "done"


def test_cancel(tmp_path):
    mgr = JobManager(run_fn=FakeRunner(tmp_path, delay=0.5))
    c = TestClient(create_app(mgr, PORT))
    jid = c.post("/jobs/url", headers={**H, **ORIGIN},
                 json={"url": "https://youtu.be/dQw4w9WgXcQ", "device": "cpu"}).json()["job_id"]
    assert c.post(f"/jobs/{jid}/cancel", headers={**H, **ORIGIN}).json()["cancelled"] is True
    assert wait_done(c, jid)["status"] == "cancelled"


def test_job_error_surfaced(tmp_path):
    mgr = JobManager(run_fn=FakeRunner(tmp_path, fail="video rusak"))
    c = TestClient(create_app(mgr, PORT))
    jid = c.post("/jobs/url", headers={**H, **ORIGIN},
                 json={"url": "https://youtu.be/dQw4w9WgXcQ", "device": "cpu"}).json()["job_id"]
    s = wait_done(c, jid)
    assert s["status"] == "error" and "rusak" in s["error"]


def test_404_unknown_job(client):
    assert client.get("/jobs/nope", headers=H).status_code == 404


# ---------- penyajian frontend ----------

HOST_ONLY = {"host": f"127.0.0.1:{PORT}"}


def test_index_injects_token_and_is_public(client):
    r = client.get("/", headers=HOST_ONLY)   # tanpa token: halaman harus tetap terbuka
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert security.SESSION_TOKEN in r.text and "__VIDNOTE_TOKEN__" not in r.text


def test_static_served(client):
    assert client.get("/style.css", headers=HOST_ONLY).status_code == 200
    js = client.get("/app.js", headers=HOST_ONLY)
    assert js.status_code == 200 and "application/javascript" in js.headers["content-type"]


def test_favicon_no_content(client):
    assert client.get("/favicon.ico", headers=HOST_ONLY).status_code == 204


def test_sse_and_file_accept_query_token(tmp_path):
    mgr = JobManager(run_fn=FakeRunner(tmp_path))
    c = TestClient(create_app(mgr, PORT))
    jid = c.post("/jobs/url", headers={**H, **ORIGIN},
                 json={"url": "https://youtu.be/dQw4w9WgXcQ", "device": "cpu"}).json()["job_id"]
    wait_done(c, jid)
    # tanpa header token, tapi dengan ?token= pada GET -> boleh
    r = c.get(f"/jobs/{jid}/srt?token={security.SESSION_TOKEN}", headers=HOST_ONLY)
    assert r.status_code == 200
    bad = c.get(f"/jobs/{jid}/srt?token=salah", headers=HOST_ONLY)
    assert bad.status_code == 401


def test_query_token_rejected_on_post(client):
    # token via query tidak berlaku untuk POST (hanya GET/HEAD)
    r = client.post(f"/jobs/x/cancel?token={security.SESSION_TOKEN}", headers={**HOST_ONLY, **ORIGIN})
    assert r.status_code == 401

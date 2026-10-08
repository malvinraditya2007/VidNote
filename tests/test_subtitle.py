import subprocess

import pytest

from backend.pipeline import subtitle as S
from tests.conftest import FFMPEG, needs_ffmpeg


def cue(i, start, end, spk, text):
    return {"id": i, "start": start, "end": end, "speaker": spk, "text": text}


CUES = [
    cue(0, 0.0, 2.0, 0, "Halo semuanya, selamat datang di VidNote."),
    cue(1, 2.1, 4.0, 1, "Terima kasih, senang bisa ikut."),
    cue(2, 4.0, 4.2, 1, "Oke."),
]
T = {"orientation": "landscape", "num_speakers": 2, "cues": CUES}


def test_time_formats():
    assert S.srt_time(3725.5) == "01:02:05,500"
    assert S.vtt_time(0.001) == "00:00:00.001"
    assert S.ass_time(65.256) == "0:01:05.26"
    assert S.srt_time(-1) == "00:00:00,000"


def test_wrap_balanced_two_lines():
    text = "Selamat datang di ruang HRD tempat aku bantuin kamu cari kerja"
    lines = S.wrap_lines(text, 42)
    assert len(lines) == 2 and all(len(x) <= 42 for x in lines)
    assert abs(len(lines[0]) - len(lines[1])) < 15
    assert S.wrap_lines("pendek", 42) == ["pendek"]


def test_escape_ass_blocks_override_tags():
    esc = S.escape_ass(r"{\an8\fs200}hack\N")
    assert "{" not in esc and "}" not in esc and "\\" not in esc


def test_escape_vtt():
    assert S.escape_vtt("<b>a & b</b> -->") == "&lt;b&gt;a &amp; b&lt;/b&gt; →"


def test_normalize_extends_short_and_no_overlap():
    cs = S.normalize_cues(CUES + [cue(3, 9, 9, 0, "  ")])
    assert len(cs) == 3
    assert cs[2]["end"] - cs[2]["start"] >= S.MIN_CUE_SEC - 1e-9
    assert all(a["end"] <= b["start"] for a, b in zip(cs, cs[1:]))


def test_srt_labels_only_on_speaker_change():
    srt = S.to_srt(CUES, "landscape", 2)
    assert srt.count("(Pembicara 2)") == 1 and srt.count("(Pembicara 1)") == 1
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "(Pembicara" not in S.to_srt(CUES, "landscape", 1)


def test_vtt_voice_and_style():
    vtt = S.to_vtt(CUES, "landscape", 2)
    assert vtt.startswith("WEBVTT")
    assert "<v Pembicara 2>Terima kasih" in vtt
    assert '::cue(v[voice="Pembicara 1"]) { color: #FFFFFF; }' in vtt


def test_ass_styles_colors_and_layout():
    ass = S.to_ass(CUES, "portrait", 2)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Style: S0,Arial,66,&H00FFFFFF" in ass
    assert "Style: S1,Arial,66,&H004DE1FF" in ass  # FFE14D -> BGR
    assert ass.count("Dialogue:") == 3
    assert S.color_of(8) == S.color_of(0)


@needs_ffmpeg
@pytest.mark.parametrize("name", ["subs.srt", "subs.vtt", "subs.ass"])
def test_ffmpeg_parses_output(tmp_path, name):
    for n, content in S.build_all(T).items():
        (tmp_path / n).write_text(content, encoding="utf-8")
    # Konversi ke SRT: gagal jika ffmpeg tidak bisa mem-parse file.
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", name, "file:check.srt"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "check.srt").read_text(encoding="utf-8").count("-->") == 3

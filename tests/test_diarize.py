import io
import tarfile

import pytest

from backend import models_dl
from backend.errors import VidNoteError
from backend.pipeline.diarize import count_speakers, normalize_turns, speaker_label, speaker_stats


def test_relabel_by_first_appearance():
    turns = normalize_turns([(5.0, 7.0, 3), (0.0, 2.0, 1), (2.5, 4.0, 3), (8.0, 9.0, 0)])
    assert [(t.start, t.speaker) for t in turns] == [(0.0, 0), (2.5, 1), (5.0, 1), (8.0, 2)]
    assert count_speakers(turns) == 3
    assert speaker_label(0) == "Pembicara 1"


def test_merge_adjacent_same_speaker():
    turns = normalize_turns([(0, 1, 0), (1.0, 2, 0), (2.5, 3, 0)], merge_gap=0.0)
    assert [(t.start, t.end) for t in turns] == [(0, 2), (2.5, 3)]
    turns = normalize_turns([(0, 1, 0), (1.2, 2, 0)], merge_gap=0.5)
    assert len(turns) == 1


def test_drops_empty_turns_and_stats():
    turns = normalize_turns([(1, 1, 0), (0, 2, 1), (3, 4, 0)])
    assert count_speakers(turns) == 2
    assert speaker_stats(turns) == {0: 2.0, 1: 1.0}


def test_empty():
    assert normalize_turns([]) == [] and count_speakers([]) == 0


# ---------- download aman ----------

@pytest.mark.parametrize("url", [
    "http://github.com/x.onnx",
    "https://evil.com/x.onnx",
    "https://github.com.evil.com/x.onnx",
    "file:///C:/x.onnx",
])
def test_download_rejects_bad_urls(url, tmp_path):
    with pytest.raises(VidNoteError, match="tidak diizinkan"):
        models_dl.download(url, tmp_path / "x", "0" * 64)


def test_download_skips_when_hash_matches(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"model")
    # URL tidak pernah dihubungi karena file lokal sudah cocok
    assert models_dl.download("https://github.com/x", f, models_dl.sha256_of(f)) == f


def _tar(tmp_path, members):
    p = tmp_path / "a.tar.bz2"
    with tarfile.open(p, "w:bz2") as t:
        for name, data, kind in members:
            ti = tarfile.TarInfo(name)
            if kind == "sym":
                ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
                t.addfile(ti)
            else:
                ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))
    return p


def test_extract_member_only_writes_dst(tmp_path):
    arc = _tar(tmp_path, [("pkg/model.onnx", b"abc", "file"), ("../../evil.txt", b"x", "file")])
    dst = tmp_path / "out" / "model.onnx"
    dst.parent.mkdir()
    models_dl.extract_member(arc, "pkg/model.onnx", dst)
    assert dst.read_bytes() == b"abc"
    assert not (tmp_path.parent / "evil.txt").exists()


def test_extract_member_rejects_symlink_and_missing(tmp_path):
    arc = _tar(tmp_path, [("pkg/model.onnx", b"", "sym")])
    with pytest.raises(VidNoteError, match="bukan file biasa"):
        models_dl.extract_member(arc, "pkg/model.onnx", tmp_path / "m")
    with pytest.raises(VidNoteError, match="tidak ada"):
        models_dl.extract_member(arc, "pkg/other.onnx", tmp_path / "m")

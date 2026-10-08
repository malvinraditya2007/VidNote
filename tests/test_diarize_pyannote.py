"""Test adapter pyannote tanpa benar-benar memuat PyTorch (API dites via mock)."""
import sys
import types

import pytest

from backend import config
from backend.errors import VidNoteError
from backend.pipeline import diarize_pyannote as P
from backend.pipeline.diarize import run_diarization


def test_dispatch_unknown_backend(tmp_path):
    with pytest.raises(VidNoteError, match="tidak dikenal"):
        run_diarization(tmp_path / "a.wav", backend="xyz")


def test_resolve_source_requires_token(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PYANNOTE_DIR", tmp_path / "none")
    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.delenv(config.HF_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)  # tak ada .env
    with pytest.raises(VidNoteError, match="token Hugging Face"):
        P._resolve_source()


def test_resolve_source_offline_without_clone(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PYANNOTE_DIR", tmp_path / "none")
    monkeypatch.setattr(config, "OFFLINE", True)
    with pytest.raises(VidNoteError, match="offline"):
        P._resolve_source()


def test_resolve_source_local_clone_no_token(monkeypatch, tmp_path):
    d = tmp_path / "clone"
    d.mkdir()
    (d / "config.yaml").write_text("x")
    monkeypatch.setattr(config, "PYANNOTE_DIR", d)
    monkeypatch.setattr(config, "OFFLINE", False)
    source, token = P._resolve_source()
    assert source == str(d) and token is None


def test_load_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.delenv(config.HF_TOKEN_ENV, raising=False)
    (tmp_path / ".env").write_text('HF_TOKEN="hf_abc123"\n# komentar\nLAIN=1\n')
    P._load_dotenv()
    assert P._hf_token() == "hf_abc123"


class _Turn:
    def __init__(self, s, e): self.start, self.end = s, e


class _Annot:
    def __init__(self, items): self._items = items
    def itertracks(self, yield_label=True):
        for s, e, spk in self._items:
            yield _Turn(s, e), None, spk


class _Output:
    def __init__(self, items): self.speaker_diarization = _Annot(items)


def test_diarize_pyannote_maps_to_turns(monkeypatch, tmp_path):
    # pipeline palsu -> dua pembicara, urutan kemunculan
    # SPK_B bicara duluan -> jadi Pembicara 0; keduanya > 3 dtk agar tak digabung.
    fake = lambda src, **kw: _Output([(0.0, 10.0, "SPK_B"), (10.1, 20.0, "SPK_A"), (20.1, 30.0, "SPK_B")])
    monkeypatch.setattr(P, "_get_pipeline", lambda device: fake)
    monkeypatch.setattr(P, "_load_waveform", lambda path: {"waveform": None, "sample_rate": 16000})
    res = P.diarize_pyannote(tmp_path / "a.wav")
    assert res.num_speakers == 2 and res.embedding == "pyannote-community-1"
    assert [t.speaker for t in res.turns] == [0, 1, 0]


def test_diarize_pyannote_rejects_bad_num_speakers(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_get_pipeline", lambda device: (lambda p, **k: _Output([])))
    monkeypatch.setattr(P, "_load_waveform", lambda path: {"waveform": None, "sample_rate": 16000})
    with pytest.raises(VidNoteError, match="num-speakers"):
        P.diarize_pyannote(tmp_path / "a.wav", num_speakers=99)

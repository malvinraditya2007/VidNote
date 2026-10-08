import pytest

from backend import config
from backend.pipeline.asr import (
    Segment, decide_language, fallback_chain, filter_hallucinations, is_oom_error, pick_chunks,
)


def seg(text, start=0.0, end=1.0, logprob=-0.2, nsp=0.01, cr=1.3):
    return Segment(0, start, end, text, [], logprob, nsp, cr)


# ---------- bahasa ----------

def test_lang_clear_indonesian():
    d = decide_language([{"id": 0.95, "en": 0.02}, {"id": 0.9, "en": 0.05}, {"id": 0.97}])
    assert d.language == "id" and not d.mixed


def test_lang_clear_english():
    d = decide_language([{"en": 0.98}, {"en": 0.97, "id": 0.01}, {"en": 0.99}])
    assert d.language == "en" and not d.mixed


def test_lang_malay_counts_as_indonesian():
    d = decide_language([{"ms": 0.6, "id": 0.3}, {"id": 0.8}, {"ms": 0.5, "id": 0.4}])
    assert d.language == "id"


def test_lang_mixed_warning():
    d = decide_language([{"id": 0.9}, {"en": 0.9}, {"id": 0.9}])
    assert d.language == "id" and d.mixed


def test_lang_unsupported_rejected():
    d = decide_language([{"ja": 0.95, "en": 0.02}, {"ja": 0.9}, {"ja": 0.97}])
    assert d.language is None and "bukan Indonesia/Inggris" in d.reason


def test_lang_forced_and_empty():
    assert decide_language([{"ja": 1.0}], forced="en").language == "en"
    d = decide_language([])
    assert d.language is None and "tidak ada ucapan" in d.reason


def test_pick_chunks():
    sr = 16000
    assert pick_chunks(0, sr, 3, 30) == []
    assert pick_chunks(10 * sr, sr, 3, 30) == [(0, 10 * sr)]
    ch = pick_chunks(300 * sr, sr, 3, 30)
    assert len(ch) == 3 and ch[0][0] == 0 and ch[-1][1] == 300 * sr
    assert all(b - a == 30 * sr for a, b in ch)


# ---------- filter halusinasi ----------

def test_filter_keeps_normal_speech():
    kept, dropped = filter_hallucinations([seg("Halo semuanya, selamat datang."), seg("Hari ini kita belajar.")])
    assert len(kept) == 2 and dropped == []
    assert [s.id for s in kept] == [0, 1]


@pytest.mark.parametrize("s,reason", [
    (seg("   "), "kosong"),
    (seg("musik", nsp=0.9, logprob=-1.5), "bukan ucapan"),
    (seg("ya ya ya ya ya ya ya ya ya ya ya ya", cr=3.1), "berulang"),
    (seg("Subtitle by Amara.org community"), "subtitle"),
    (seg("Jangan lupa subscribe ya!"), "subscribe"),
    (seg("Terima kasih telah menonton!", logprob=-0.9), "kepercayaan rendah"),
])
def test_filter_drops(s, reason):
    kept, dropped = filter_hallucinations([s])
    assert kept == [] and reason in dropped[0]["reason"]


def test_filter_keeps_confident_thanks():
    kept, _ = filter_hallucinations([seg("Terima kasih.", logprob=-0.1, nsp=0.01)])
    assert len(kept) == 1


def test_filter_drops_exact_consecutive_duplicates():
    kept, dropped = filter_hallucinations([
        seg("Kita lanjut ke bagian berikutnya."),
        seg("Kita lanjut ke bagian berikutnya."),
        seg("Bagian kedua."),
    ])
    assert [s.text for s in kept] == ["Kita lanjut ke bagian berikutnya.", "Bagian kedua."]
    assert dropped[0]["reason"].startswith("duplikat")


# ---------- OOM ----------

@pytest.mark.parametrize("msg", [
    "CUDA failed with error out of memory",
    "RuntimeError: CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA_ERROR_OUT_OF_MEMORY",
])
def test_is_oom(msg):
    assert is_oom_error(RuntimeError(msg))


def test_not_oom():
    assert not is_oom_error(RuntimeError("Library cudnn_ops64_9.dll is not found"))


def test_fallback_chain():
    assert fallback_chain("large-v3", "gpu") == config.ASR_OOM_CHAIN
    assert fallback_chain("large-v3-turbo", "gpu") == config.ASR_OOM_CHAIN[1:]
    assert fallback_chain("medium", "cpu") == [("medium", "cpu")]
    assert fallback_chain("small", "gpu") == [("small", "gpu"), ("medium", "cpu")]

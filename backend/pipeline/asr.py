"""ASR dengan faster-whisper.

Alur:
1. Decode audio.wav (16 kHz mono) -> numpy.
2. Silero VAD -> ambil bagian ucapan saja.
3. Deteksi bahasa dari 3 potongan ucapan (awal, tengah, akhir), dibatasi id/en,
   kecuali user memaksa --lang.
4. Transcribe: VAD aktif, word timestamps, condition_on_previous_text=False.
5. Filter halusinasi.
6. Unload model (bebaskan VRAM) sebelum tahap berikutnya.

OOM di GPU: bebaskan memori, tanya user (callback), lalu turun bertahap
large-v3 -> large-v3-turbo -> medium (CPU).
"""
from __future__ import annotations

import gc
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from backend import config, device as device_mod
from backend.errors import VidNoteError

# ---------- tipe data ----------


@dataclass
class Word:
    start: float
    end: float
    word: str
    prob: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word]
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


@dataclass
class LangDecision:
    language: Optional[str]          # "id" | "en" | None (ditolak)
    probs: dict[str, float]          # rata-rata probabilitas id/en dari semua potongan
    supported_prob: float            # id + en
    mixed: bool
    forced: bool = False
    reason: str = ""


@dataclass
class AsrResult:
    model: str
    device: str
    compute_type: str
    language: str
    lang: LangDecision
    duration: float
    speech_sec: float
    segments: list[Segment]
    dropped: list[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0
    vram_peak_mb: Optional[int] = None
    fallbacks: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- fungsi murni (di-unit-test) ----------

def decide_language(chunk_probs: list[dict[str, float]], forced: Optional[str] = None) -> LangDecision:
    """Pilih id/en dari probabilitas per potongan.

    chunk_probs: tiap elemen = {kode_bahasa: prob} dari detect_language.
    """
    if forced in config.SUPPORTED_LANGS:
        return LangDecision(forced, {}, 1.0, False, forced=True)
    if not chunk_probs:
        return LangDecision(None, {}, 0.0, False, reason="tidak ada ucapan yang terdeteksi")

    def eff(c: dict[str, float]) -> dict[str, float]:
        # Whisper sering melabeli bahasa Indonesia sebagai Melayu (ms); gabungkan ke id.
        return {"id": c.get("id", 0.0) + c.get("ms", 0.0), "en": c.get("en", 0.0)}

    chunks = [eff(c) for c in chunk_probs]
    n = len(chunks)
    avg = {k: round(sum(c[k] for c in chunks) / n, 4) for k in ("id", "en")}
    supported = round(avg["id"] + avg["en"], 4)
    if supported < config.LANG_MIN_SUPPORTED_PROB:
        first = chunk_probs[0] or {"?": 1.0}
        top = max(first, key=first.get)
        return LangDecision(
            None, avg, supported, False,
            reason=f"bahasa video sepertinya bukan Indonesia/Inggris (terdeteksi: {top}). "
                   f"Paksa dengan --lang id atau --lang en jika keliru.",
        )
    lang = "id" if avg["id"] >= avg["en"] else "en"
    other = "en" if lang == "id" else "id"
    per_chunk = {("id" if c["id"] >= c["en"] else "en") for c in chunks}
    mixed = avg[other] / supported >= config.LANG_MIXED_RATIO or len(per_chunk) > 1
    return LangDecision(lang, avg, supported, mixed)


_NONSPEECH_PHRASES = [
    # Pola khas halusinasi Whisper dari data subtitle YouTube; hampir tidak pernah ucapan asli.
    "subtitle by", "subtitles by", "amara.org", "terjemahan oleh", "subtitle oleh",
    "please subscribe", "jangan lupa subscribe", "like and subscribe",
]
_LOWCONF_PHRASES = [
    # Bisa ucapan asli, jadi hanya dibuang jika kepercayaannya rendah.
    "terima kasih telah menonton", "terima kasih sudah menonton", "terima kasih",
    "thanks for watching", "thank you for watching", "thank you", "sampai jumpa",
    "see you next time", "bye",
]


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s.]", " ", t)
    return re.sub(r"\s+", " ", t).strip(" .")


def filter_hallucinations(segments: list[Segment]) -> tuple[list[Segment], list[dict]]:
    """Kembalikan (segmen_dipertahankan, daftar_dibuang_dengan_alasan)."""
    kept: list[Segment] = []
    dropped: list[dict] = []

    def drop(s: Segment, reason: str) -> None:
        dropped.append({"start": s.start, "end": s.end, "text": s.text, "reason": reason})

    for s in segments:
        norm = _norm(s.text)
        lowconf = s.avg_logprob < -0.5 or s.no_speech_prob > 0.2
        if not norm:
            drop(s, "teks kosong")
        elif s.no_speech_prob > config.HALLU_NO_SPEECH_PROB and s.avg_logprob < config.HALLU_LOGPROB:
            drop(s, "kemungkinan bukan ucapan")
        elif s.compression_ratio > config.HALLU_COMPRESSION_RATIO:
            drop(s, "teks berulang-ulang")
        elif any(p in norm for p in _NONSPEECH_PHRASES):
            drop(s, "pola halusinasi (subtitle/subscribe)")
        elif norm in _LOWCONF_PHRASES and lowconf:
            drop(s, "pola halusinasi (kepercayaan rendah)")
        elif kept and _norm(kept[-1].text) == norm and len(norm) > 3:
            drop(s, "duplikat segmen sebelumnya")
        else:
            kept.append(s)
    for i, s in enumerate(kept):
        s.id = i
    return kept, dropped


_OOM_PATTERNS = ("out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed",
                 "failed to allocate", "cudnn_status_alloc_failed")


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _OOM_PATTERNS)


def fallback_chain(model: str, dev: str) -> list[tuple[str, str]]:
    """Urutan (model, device) yang dicoba, mulai dari pilihan user."""
    chain = list(config.ASR_OOM_CHAIN)
    start = (model, dev)
    if start in chain:
        return chain[chain.index(start):]
    if dev == "gpu":
        return [start] + [c for c in chain if c[1] == "cpu"]
    return [start]


def pick_chunks(speech_len: int, sr: int, n: int, chunk_sec: int) -> list[tuple[int, int]]:
    """Ambil n jendela (awal, tengah, akhir) dari audio ucapan sepanjang speech_len sampel."""
    size = chunk_sec * sr
    if speech_len <= size:
        return [(0, speech_len)] if speech_len > 0 else []
    if n == 1:
        return [(0, size)]
    starts = [round(i * (speech_len - size) / (n - 1)) for i in range(n)]
    return [(s, s + size) for s in starts]


# ---------- model ----------

def compute_type_for(dev: str) -> str:
    return config.MODELS[dev]["asr_compute_type"]


def load_model(name: str, dev: str):
    from faster_whisper import WhisperModel

    device_mod.setup_cuda_libs()
    kwargs = dict(
        device="cuda" if dev == "gpu" else "cpu",
        compute_type=compute_type_for(dev),
        download_root=str(config.WHISPER_DIR),
        cpu_threads=device_mod.cpu_threads() if dev == "cpu" else 0,
    )
    config.WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return WhisperModel(name, local_files_only=True, **kwargs)
    except Exception as e:
        if is_oom_error(e):
            raise
        if config.OFFLINE:
            raise VidNoteError(f"model Whisper {name} belum terunduh dan mode offline aktif. "
                               f"Jalankan sekali: python vidnote.py download-models") from e
    # Belum ada di lokal -> unduh dari repo resmi (Systran / mobiuslabsgmbh di Hugging Face).
    return WhisperModel(name, local_files_only=False, **kwargs)


# ---------- langkah utama ----------

def _speech_audio(audio, sr: int):
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=config.ASR_VAD_MIN_SILENCE_MS))
    if not ts:
        return np.zeros(0, dtype=audio.dtype)
    return np.concatenate([audio[t["start"]:t["end"]] for t in ts])


def detect_language(model, audio, forced: Optional[str]) -> tuple[LangDecision, float]:
    """Kembalikan (keputusan, detik_ucapan)."""
    sr = config.AUDIO_SAMPLE_RATE
    speech = _speech_audio(audio, sr)
    speech_sec = len(speech) / sr
    if forced in config.SUPPORTED_LANGS:
        return decide_language([], forced), speech_sec
    probs: list[dict[str, float]] = []
    for a, b in pick_chunks(len(speech), sr, config.LANG_DETECT_CHUNKS, config.LANG_DETECT_CHUNK_SEC):
        if b - a < sr:  # < 1 dtk ucapan, abaikan
            continue
        _, _, all_probs = model.detect_language(audio=speech[a:b])
        probs.append(dict(all_probs))
    return decide_language(probs), speech_sec


def _to_segment(s) -> Segment:
    words = [Word(round(w.start, 3), round(w.end, 3), w.word, round(w.probability, 3))
             for w in (s.words or [])]
    return Segment(
        id=s.id, start=round(s.start, 3), end=round(s.end, 3), text=s.text.strip(), words=words,
        avg_logprob=round(s.avg_logprob, 4), no_speech_prob=round(s.no_speech_prob, 4),
        compression_ratio=round(s.compression_ratio, 3),
    )


def _transcribe_once(model, audio, language: str, duration: float,
                     on_progress: Optional[Callable[[float], None]]) -> list[Segment]:
    segments_iter, _info = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=config.ASR_BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": config.ASR_VAD_MIN_SILENCE_MS},
        word_timestamps=True,
        condition_on_previous_text=False,
        compression_ratio_threshold=config.HALLU_COMPRESSION_RATIO,
        log_prob_threshold=config.HALLU_LOGPROB,
        no_speech_threshold=config.HALLU_NO_SPEECH_PROB,
        hallucination_silence_threshold=config.ASR_HALLUCINATION_SILENCE_SEC,
    )
    out = []
    for s in segments_iter:  # generator: decoding terjadi di sini
        out.append(_to_segment(s))
        if on_progress and duration > 0:
            on_progress(min(1.0, s.end / duration))
    return out


def transcribe(
    audio_path: Path,
    dev: str,
    lang: str = "auto",
    model_name: Optional[str] = None,
    on_oom: Optional[Callable[[str, str], bool]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> AsrResult:
    """Jalankan ASR penuh. on_oom(dari, ke) -> True untuk lanjut ke fallback."""
    from faster_whisper import decode_audio

    device_mod.require_device(dev)
    status = on_status or (lambda _m: None)
    forced = lang if lang in config.SUPPORTED_LANGS else None
    name = model_name or config.MODELS[dev]["asr"]
    audio = decode_audio(str(audio_path), sampling_rate=config.AUDIO_SAMPLE_RATE)
    duration = len(audio) / config.AUDIO_SAMPLE_RATE

    fallbacks: list[str] = []
    chain = fallback_chain(name, dev)
    for i, (m_name, m_dev) in enumerate(chain):
        model = None
        t0 = time.perf_counter()
        try:
            with device_mod.VramMonitor() if m_dev == "gpu" else _NullMonitor() as mon:
                status(f"Memuat model {m_name} ({m_dev.upper()}, {compute_type_for(m_dev)})")
                model = load_model(m_name, m_dev)
                status("Mendeteksi bahasa")
                decision, speech_sec = detect_language(model, audio, forced)
                if decision.language is None:
                    if speech_sec < 1.0:
                        return AsrResult(m_name, m_dev, compute_type_for(m_dev), "", decision,
                                         round(duration, 3), round(speech_sec, 1), [],
                                         elapsed_sec=round(time.perf_counter() - t0, 1),
                                         fallbacks=fallbacks)
                    raise VidNoteError(decision.reason)
                status(f"Transkripsi ({decision.language})")
                raw = _transcribe_once(model, audio, decision.language, duration, on_progress)
            kept, dropped = filter_hallucinations(raw)
            return AsrResult(
                model=m_name, device=m_dev, compute_type=compute_type_for(m_dev),
                language=decision.language, lang=decision, duration=round(duration, 3),
                speech_sec=round(speech_sec, 1), segments=kept, dropped=dropped,
                elapsed_sec=round(time.perf_counter() - t0, 1),
                vram_peak_mb=getattr(mon, "delta_mb", None), fallbacks=fallbacks,
            )
        except Exception as e:
            if not (m_dev == "gpu" and is_oom_error(e)):
                raise
            model = None  # lepas referensi -> VRAM dibebaskan ctranslate2
            gc.collect()
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            if nxt is None:
                raise VidNoteError("VRAM tidak cukup dan tidak ada model cadangan. Coba --device cpu.")
            here, there = f"{m_name} (GPU)", f"{nxt[0]} ({nxt[1].upper()})"
            if on_oom is None or not on_oom(here, there):
                raise VidNoteError(f"VRAM tidak cukup untuk {here}. Dibatalkan.")
            fallbacks.append(f"{here} -> {there}: kehabisan VRAM")
        finally:
            model = None  # unload sebelum tahap berikutnya
            gc.collect()
    raise VidNoteError("ASR gagal")  # tidak tercapai


class _NullMonitor:
    delta_mb = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

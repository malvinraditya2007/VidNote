"""Speaker diarization dengan sherpa-onnx (CPU, tanpa token).

Model: pyannote-segmentation-3.0 (ONNX) + embedding 3D-Speaker CAM++.
Jumlah pembicara dideteksi otomatis lewat clustering (num_clusters=-1 +
threshold), atau dipaksa dengan num_speakers.

Label dinormalisasi berdasarkan urutan kemunculan: pembicara yang bicara
pertama = 0 (ditampilkan "Pembicara 1").
"""
from __future__ import annotations

import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from backend import config, device as device_mod
from backend.errors import VidNoteError
from backend.models_dl import download, extract_member, sha256_of


@dataclass
class Turn:
    start: float
    end: float
    speaker: int  # 0-based, urut kemunculan


@dataclass
class DiarResult:
    turns: list[Turn]
    num_speakers: int
    embedding: str
    threshold: float
    forced_speakers: Optional[int]
    duration: float
    elapsed_sec: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def speaker_label(idx: int) -> str:
    return f"Pembicara {idx + 1}"


# ---------- fungsi murni ----------

def normalize_turns(raw: list[tuple[float, float, int]], merge_gap: float = 0.0) -> list[Turn]:
    """Urutkan, relabel per urutan kemunculan, gabungkan turn berurutan dari pembicara sama."""
    raw = sorted((round(s, 3), round(e, 3), spk) for s, e, spk in raw if e > s)
    mapping: dict[int, int] = {}
    turns: list[Turn] = []
    for s, e, spk in raw:
        if spk not in mapping:
            mapping[spk] = len(mapping)
        lab = mapping[spk]
        if turns and turns[-1].speaker == lab and s - turns[-1].end <= merge_gap:
            turns[-1].end = max(turns[-1].end, e)
        else:
            turns.append(Turn(s, e, lab))
    return turns


def count_speakers(turns: list[Turn]) -> int:
    return len({t.speaker for t in turns})


def speaker_stats(turns: list[Turn]) -> dict[int, float]:
    out: dict[int, float] = {}
    for t in turns:
        out[t.speaker] = out.get(t.speaker, 0.0) + (t.end - t.start)
    return {k: round(v, 1) for k, v in sorted(out.items())}


def absorb_minor_speakers(turns: list[Turn], min_sec: float = config.DIAR_MINOR_MIN_SEC,
                          min_ratio: float = config.DIAR_MINOR_MIN_RATIO) -> tuple[list[Turn], int, float]:
    """Gabungkan pembicara dengan total bicara sangat sedikit ke turn pembicara utama terdekat.

    Kembalikan (turns_baru, jumlah_pembicara_digabung, batas_detik).
    """
    stats = speaker_stats(turns)
    if len(stats) <= 1:
        return turns, 0, 0.0
    limit = max(min_sec, min_ratio * sum(stats.values()))
    major = {s for s, v in stats.items() if v >= limit} or {max(stats, key=stats.get)}
    minor = set(stats) - major
    if not minor:
        return turns, 0, limit
    major_turns = [t for t in turns if t.speaker in major]
    out = []
    for t in turns:
        spk = t.speaker
        if spk in minor:
            spk = min(major_turns, key=lambda u: max(0.0, u.start - t.end, t.start - u.end)).speaker
        out.append((t.start, t.end, spk))
    return normalize_turns(out), len(minor), round(limit, 1)


# ---------- model ----------

def ensure_models(embedding: str = config.DIAR_DEFAULT_EMBEDDING,
                  on_status: Optional[Callable[[str], None]] = None) -> tuple[Path, Path]:
    """Pastikan model ada di models/diarization/ dan hash-nya cocok. Kembalikan (seg, emb)."""
    status = on_status or (lambda _m: None)
    if embedding not in config.DIAR_EMBEDDINGS:
        raise VidNoteError(f"embedding tidak dikenal: {embedding} "
                           f"(pilihan: {', '.join(config.DIAR_EMBEDDINGS)})")
    d = config.DIAR_DIR
    d.mkdir(parents=True, exist_ok=True)

    seg_cfg = config.DIAR_SEGMENTATION
    seg = d / seg_cfg["file"]
    if not (seg.is_file() and sha256_of(seg) == seg_cfg["member_sha256"]):
        status("Mengunduh model segmentation (sekali saja)")
        archive = download(seg_cfg["url"], d / "segmentation.tar.bz2", seg_cfg["sha256"])
        try:
            extract_member(archive, seg_cfg["member"], seg)
        finally:
            archive.unlink(missing_ok=True)
        if sha256_of(seg) != seg_cfg["member_sha256"]:
            seg.unlink(missing_ok=True)
            raise VidNoteError("hash model segmentation tidak cocok")

    emb_cfg = config.DIAR_EMBEDDINGS[embedding]
    emb = d / emb_cfg["file"]
    if not (emb.is_file() and sha256_of(emb) == emb_cfg["sha256"]):
        status(f"Mengunduh model embedding {embedding} (sekali saja)")
        download(emb_cfg["url"], emb, emb_cfg["sha256"])
    return seg, emb


def _build(seg: Path, emb: Path, num_clusters: int, threshold: float):
    import sherpa_onnx

    threads = device_mod.cpu_threads()
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=config.DIAR_MIN_DURATION_ON,
        min_duration_off=config.DIAR_MIN_DURATION_OFF,
    )
    if not cfg.validate():
        raise VidNoteError("konfigurasi diarization tidak valid (cek file model)")
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def read_wav_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise VidNoteError("audio untuk diarization harus WAV mono 16-bit")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr


def _run(sd, samples: np.ndarray, on_progress) -> list[tuple[float, float, int]]:
    def cb(done: int, total: int) -> int:
        if on_progress and total:
            on_progress(done / total)
        return 0

    result = sd.process(samples, callback=cb).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in result]


def run_diarization(
    audio_path: Path,
    backend: str = config.DIAR_DEFAULT_BACKEND,
    device: str = "cpu",
    num_speakers: Optional[int] = None,
    threshold: float = config.DIAR_THRESHOLD,
    embedding: str = config.DIAR_DEFAULT_EMBEDDING,
    on_status: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> DiarResult:
    """Pilih backend diarization: 'sherpa' (default) atau 'pyannote' (opsional)."""
    if backend == "pyannote":
        from backend.pipeline.diarize_pyannote import diarize_pyannote

        return diarize_pyannote(audio_path, device=device, num_speakers=num_speakers,
                                on_status=on_status, on_progress=on_progress)
    if backend != "sherpa":
        raise VidNoteError(f"diarizer tidak dikenal: {backend} (pilihan: {', '.join(config.DIAR_BACKENDS)})")
    return diarize(audio_path, num_speakers=num_speakers, threshold=threshold,
                   embedding=embedding, on_status=on_status, on_progress=on_progress)


def diarize(
    audio_path: Path,
    num_speakers: Optional[int] = None,
    threshold: float = config.DIAR_THRESHOLD,
    embedding: str = config.DIAR_DEFAULT_EMBEDDING,
    on_status: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> DiarResult:
    status = on_status or (lambda _m: None)
    if num_speakers is not None and not (1 <= num_speakers <= config.DIAR_MAX_SPEAKERS * 2):
        raise VidNoteError(f"--num-speakers harus 1-{config.DIAR_MAX_SPEAKERS * 2}")
    seg, emb = ensure_models(embedding, status)
    samples, sr = read_wav_float(audio_path)
    duration = len(samples) / sr if sr else 0.0
    t0 = time.perf_counter()
    notes: list[str] = []

    status("Memuat model diarization (CPU)")
    sd = _build(seg, emb, num_speakers or -1, threshold)
    if sr != sd.sample_rate:
        raise VidNoteError(f"sample rate audio {sr} Hz, model butuh {sd.sample_rate} Hz")

    status("Memisahkan pembicara")
    turns = normalize_turns(_run(sd, samples, on_progress))
    if num_speakers is None and count_speakers(turns) > config.DIAR_MAX_SPEAKERS:
        n = count_speakers(turns)
        notes.append(f"terdeteksi {n} pembicara (> {config.DIAR_MAX_SPEAKERS}); "
                     f"diulang dengan batas {config.DIAR_MAX_SPEAKERS}")
        status(f"Terlalu banyak pembicara ({n}), mengulang dengan batas {config.DIAR_MAX_SPEAKERS}")
        sd = _build(seg, emb, config.DIAR_MAX_SPEAKERS, threshold)
        turns = normalize_turns(_run(sd, samples, on_progress))
    del sd
    if num_speakers is None:
        turns, n_minor, limit = absorb_minor_speakers(turns)
        if n_minor:
            notes.append(f"{n_minor} pembicara dengan total bicara < {limit:.0f} dtk "
                         f"(kemungkinan musik/tawa) digabung ke pembicara terdekat")

    return DiarResult(
        turns=turns, num_speakers=count_speakers(turns), embedding=embedding,
        threshold=threshold, forced_speakers=num_speakers, duration=round(duration, 3),
        elapsed_sec=round(time.perf_counter() - t0, 1), notes=notes,
    )

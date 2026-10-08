"""Diarization dengan pyannote community-1 (OPSIONAL, A/B vs sherpa-onnx).

Butuh extra `requirements-pyannote.txt` (PyTorch + pyannote.audio) dan token HF
sekali untuk unduh (lalu di-clone ke models/ untuk offline). Lihat README.

Mengembalikan DiarResult yang sama bentuknya dengan backend sherpa, jadi tahap
berikutnya (segment, subtitle) tidak berubah.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

from backend import config
from backend.errors import VidNoteError
from backend.pipeline.diarize import (
    DiarResult, absorb_minor_speakers, count_speakers, normalize_turns,
)


def _hf_token() -> Optional[str]:
    return os.environ.get(config.HF_TOKEN_ENV) or None


def _load_dotenv() -> None:
    """Baca HF_TOKEN dari .env sederhana (KEY=VALUE) jika belum ada di environment."""
    if _hf_token():
        return
    env = config.ROOT_DIR / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == config.HF_TOKEN_ENV:
            os.environ.setdefault(config.HF_TOKEN_ENV, v.strip().strip('"').strip("'"))


def available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_source() -> tuple[str, Optional[str]]:
    """Kembalikan (sumber_model, token). Folder lokal -> offline tanpa token."""
    if config.PYANNOTE_DIR.is_dir() and any(config.PYANNOTE_DIR.iterdir()):
        return str(config.PYANNOTE_DIR), None
    if config.OFFLINE:
        raise VidNoteError(
            f"model pyannote belum di-clone ke {config.PYANNOTE_DIR} dan mode offline aktif. "
            "Clone dulu (lihat README) atau pakai --diarizer sherpa."
        )
    _load_dotenv()
    token = _hf_token()
    if not token:
        raise VidNoteError(
            "diarizer pyannote butuh token Hugging Face. Setujui syarat di "
            f"https://hf.co/{config.PYANNOTE_MODEL_ID}, buat token read-only di "
            "hf.co/settings/tokens, lalu set HF_TOKEN (mis. di file .env). "
            "Atau pakai --diarizer sherpa (tanpa token)."
        )
    return config.PYANNOTE_MODEL_ID, token


def _load_waveform(audio_path: Path) -> dict:
    """Baca WAV 16 kHz mono -> {'waveform': tensor[1, N], 'sample_rate': 16000}.

    Memuat audio di memori (cara resmi 'Processing from memory') menghindari
    torchcodec, yang sering gagal memuat DLL FFmpeg di Windows.
    """
    import wave

    import numpy as np
    import torch

    with wave.open(str(audio_path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    wav = torch.from_numpy(data.astype(np.float32) / 32768.0).unsqueeze(0)
    return {"waveform": wav, "sample_rate": sr}


_pipeline_cache = None


def _get_pipeline(device: str):
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache
    if not available():
        raise VidNoteError("pyannote.audio belum terpasang. Jalankan: "
                            "pip install -r requirements-pyannote.txt")
    from pyannote.audio import Pipeline

    source, token = _resolve_source()
    try:
        pipe = Pipeline.from_pretrained(source, token=token) if token \
            else Pipeline.from_pretrained(source)
    except Exception as e:  # noqa: BLE001
        raise VidNoteError(f"gagal memuat pipeline pyannote: {e}")
    if pipe is None:
        raise VidNoteError("pipeline pyannote None (token salah atau syarat model belum disetujui?)")
    if device == "gpu":
        try:
            import torch

            if torch.cuda.is_available():
                pipe.to(torch.device("cuda"))
        except Exception:
            pass
    _pipeline_cache = pipe
    return pipe


def diarize_pyannote(
    audio_path: Path,
    device: str = "cpu",
    num_speakers: Optional[int] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> DiarResult:
    status = on_status or (lambda _m: None)
    if num_speakers is not None and not (1 <= num_speakers <= config.DIAR_MAX_SPEAKERS * 2):
        raise VidNoteError(f"--num-speakers harus 1-{config.DIAR_MAX_SPEAKERS * 2}")
    status("Memuat model diarization (pyannote community-1)")
    pipe = _get_pipeline(device)

    status("Memisahkan pembicara (pyannote)")
    t0 = time.perf_counter()
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipe(_load_waveform(audio_path), **kwargs)
    annotation = getattr(output, "speaker_diarization", output)

    labels: dict[str, int] = {}
    raw: list[tuple[float, float, int]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        labels.setdefault(speaker, len(labels))
        raw.append((float(turn.start), float(turn.end), labels[speaker]))
    turns = normalize_turns(raw)

    notes: list[str] = []
    if num_speakers is None:
        turns, n_minor, limit = absorb_minor_speakers(turns)
        if n_minor:
            notes.append(f"{n_minor} pembicara dengan total bicara < {limit:.0f} dtk digabung")

    duration = max((t.end for t in turns), default=0.0)
    return DiarResult(
        turns=turns, num_speakers=count_speakers(turns), embedding="pyannote-community-1",
        threshold=0.0, forced_speakers=num_speakers, duration=round(duration, 3),
        elapsed_sec=round(time.perf_counter() - t0, 1), notes=notes,
    )

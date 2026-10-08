"""Pipeline lengkap VidNote: input -> audio -> ASR -> diarization -> transkrip ->
subtitle -> ringkasan -> hardsub -> output/{nama}-{waktu}/.

Dipisah dari CLI supaya bisa dipakai ulang oleh FastAPI (Fase 1): semua laporan
lewat callback on_stage(stage, pesan) dan on_progress(stage, 0..1).

- Satu tahap berat dalam satu waktu; tiap model di-unload oleh modulnya.
- Folder kerja dihapus otomatis (termasuk Ctrl+C). Output hanya ditulis di akhir,
  jadi job yang gagal/dibatalkan tidak meninggalkan folder output setengah jadi.
- Ringkasan gagal (mis. Ollama mati) tidak menggagalkan job; jadi peringatan.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from backend import config, jobs
from backend.errors import InputRejected, VidNoteError

STAGES = ["input", "audio", "asr", "diarize", "segment", "summary", "burn"]
STAGE_LABEL = {
    "input": "Cek video", "audio": "Ekstrak audio", "asr": "Transkripsi",
    "diarize": "Pemisahan pembicara", "segment": "Transkrip & subtitle",
    "summary": "Ringkasan", "burn": "Hardsub",
}
# Perkiraan kasar total waktu (x durasi video), dari tes di RTX 4050 Laptop. Belum dites luas.
# Video 4:32: GPU 2,1 mnt (0,46x), CPU 3,6 mnt (0,8x). Rentang diberi ruang untuk CPU lebih lemah.
ESTIMATE_FACTOR = {"gpu": (0.3, 0.8), "cpu": (0.6, 2.0)}


@dataclass
class RunOptions:
    device: str
    lang: str = "auto"
    asr_model: Optional[str] = None
    llm: Optional[str] = None
    num_speakers: Optional[int] = None
    threshold: float = config.DIAR_THRESHOLD
    embedding: str = config.DIAR_DEFAULT_EMBEDDING
    diarizer: str = config.DIAR_DEFAULT_BACKEND
    skip_summary: bool = False
    skip_burn: bool = False
    out_root: Path = config.OUTPUT_DIR
    keep: bool = False
    delete_source: bool = False   # hapus file sumber setelah disalin (upload staging)
    cookies_from_browser: Optional[str] = None
    cookies_file: Optional[str] = None


@dataclass
class RunResult:
    out_dir: Path
    title: str
    files: list[str]
    timings: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def is_url(s: str) -> bool:
    return str(s).strip().lower().startswith(("http://", "https://"))


def safe_name(text: str, max_len: int = 60) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()[:max_len].strip("-")
    return t or "video"


def estimate_minutes(duration_sec: float, device: str) -> tuple[int, int]:
    lo, hi = ESTIMATE_FACTOR[device]
    return max(1, round(duration_sec * lo / 60)), max(1, round(duration_sec * hi / 60))


def required_disk_bytes(input_size: int) -> int:
    # input + audio + hardsub (bisa ~3x input) + cadangan 1 GB
    return 3 * max(input_size, 0) + 1024**3


def check_disk(path: Path, need: int) -> None:
    free = shutil.disk_usage(path).free
    if free < need:
        raise InputRejected(f"ruang disk kurang: butuh ~{need / 1024**3:.1f} GB, "
                            f"tersedia {free / 1024**3:.1f} GB")


def _dl_hook(prog: Callable[[str, float], None]):
    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                prog("input", min(1.0, (d.get("downloaded_bytes") or 0) / total))
    return hook


def run(src: str, opts: RunOptions,
        on_stage: Optional[Callable[[str, str], None]] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_oom: Optional[Callable[[str, str], bool]] = None,
        on_info: Optional[Callable[[dict], None]] = None) -> RunResult:
    from backend import device as device_mod
    from backend.pipeline.asr import transcribe
    from backend.pipeline.audio import extract_audio
    from backend.pipeline.burn import burn
    from backend.pipeline.diarize import run_diarization
    from backend.pipeline.segment import build_transcript, transcript_text
    from backend.pipeline.subtitle import build_all
    from backend.pipeline.summarize import summarize, summary_markdown

    stage = on_stage or (lambda _s, _m: None)
    prog = on_progress or (lambda _s, _f: None)
    device_mod.require_device(opts.device)
    if is_url(src) and config.OFFLINE:
        raise VidNoteError("mode offline aktif: link YouTube tidak bisa diunduh, pakai file lokal")

    timings: dict[str, float] = {}
    warnings: list[str] = []
    files: dict[str, str | Path] = {}

    def timed(name: str, fn):
        stage(name, STAGE_LABEL[name])
        t0 = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - t0, 1)
        return out

    def dump(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    with jobs.job_context(keep=opts.keep) as job:
        # 1. input
        def load():
            if is_url(src):
                from backend.pipeline.fetch_url import Cookies, fetch_url

                ck = Cookies.parse(opts.cookies_from_browser, opts.cookies_file)
                p, i, m = fetch_url(src, job, on_progress=_dl_hook(prog), cookies=ck)
                return p, i, {"source": "youtube", **m}
            from backend.pipeline.receive_file import receive_file

            p, i = receive_file(Path(src), job)
            return p, i, {"source": "file", "title": Path(src).stem}

        try:
            path, info, meta = timed("input", load)
        finally:
            if opts.delete_source and not is_url(src):
                Path(src).unlink(missing_ok=True)   # bersihkan staging upload apa pun hasilnya
        check_disk(job.dir, required_disk_bytes(info.size_bytes))
        title = meta.get("title") or "video"
        lo, hi = estimate_minutes(info.duration, opts.device)
        if on_info:
            on_info({"title": title, "duration": info.duration, "orientation": info.orientation,
                     "width": info.width, "height": info.height, "estimate_min": (lo, hi)})

        # 2. audio
        a = timed("audio", lambda: extract_audio(path, job, info.duration))
        if a.is_near_silent:
            warnings.append("audio hampir hening; transkrip kemungkinan kosong")

        # 3. ASR
        res = timed("asr", lambda: transcribe(
            a.path, opts.device, lang=opts.lang, model_name=opts.asr_model, on_oom=on_oom,
            on_status=lambda m: stage("asr", m), on_progress=lambda f: prog("asr", f)))
        warnings += [f"ASR fallback: {f}" for f in res.fallbacks]
        if res.lang.mixed:
            warnings.append("video sepertinya campur Indonesia-Inggris; akurasi bisa turun")
        if not res.segments:
            raise VidNoteError("tidak ada ucapan yang terdeteksi di video")

        # 4. diarization
        dia = timed("diarize", lambda: run_diarization(
            a.path, backend=opts.diarizer, device=opts.device, num_speakers=opts.num_speakers,
            threshold=opts.threshold, embedding=opts.embedding,
            on_status=lambda m: stage("diarize", m), on_progress=lambda f: prog("diarize", f)))
        warnings += dia.notes

        # 5. transkrip + subtitle
        def seg():
            t = build_transcript(res.segments, dia.turns, info.orientation)
            t.update(title=title, language=res.language, duration=res.duration,
                     asr_model=res.model, device=res.device, embedding=dia.embedding)
            return t

        transcript = timed("segment", seg)
        files["transcript.json"] = dump(transcript)
        files["transcript.txt"] = transcript_text(transcript, title)
        files.update(build_all(transcript, title))
        files["asr.json"] = dump(res.to_dict())
        files["diarization.json"] = dump(dia.to_dict())
        for name, content in files.items():
            job.path(name).write_text(content, encoding="utf-8")

        # 6. ringkasan (gagal -> peringatan saja)
        summary = None
        if opts.skip_summary:
            warnings.append("ringkasan dilewati (--no-summary)")
        else:
            try:
                summary = timed("summary", lambda: summarize(
                    transcript, opts.device, model=opts.llm, on_status=lambda m: stage("summary", m)))
                files["summary.json"] = dump(summary.to_dict())
                files["summary.md"] = summary_markdown(summary, title)
                warnings += summary.warnings
            except VidNoteError as e:
                warnings.append(f"ringkasan gagal: {e}")

        # 7. hardsub
        burn_res = None
        if opts.skip_burn:
            warnings.append("hardsub dilewati (--no-burn)")
        else:
            burn_res = timed("burn", lambda: burn(
                path, job, opts.device, info.duration, info.audio_codec,
                on_status=lambda m: stage("burn", m), on_progress=lambda f: prog("burn", f)))
            if burn_res.note:
                warnings.append(burn_res.note)

        # 8. tulis output (baru di akhir)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(opts.out_root) / f"{safe_name(title)}-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            (out_dir / name).write_text(str(content), encoding="utf-8")
        if burn_res:
            shutil.copyfile(burn_res.path, out_dir / "video_hardsub.mp4")
        run_info = {
            "vidnote": "fase-0",
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": {k: v for k, v in meta.items() if k != "title"},
            "title": title,
            "video": info.to_dict(),
            "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(opts).items()},
            "models": {"asr": f"{res.model} ({res.device}, {res.compute_type})",
                       "diarization": f"sherpa-onnx pyannote-3.0 + {dia.embedding}",
                       "llm": summary.model if summary else None,
                       "encoder": burn_res.encoder if burn_res else None},
            "language": res.language,
            "num_speakers": transcript["num_speakers"],
            "timings_sec": timings,
            "warnings": warnings,
        }
        (out_dir / "run.json").write_text(dump(run_info), encoding="utf-8")

    names = sorted(p.name for p in out_dir.iterdir())
    return RunResult(out_dir, title, names, timings, warnings, run_info)


# ---------- unduh model untuk mode offline ----------

def download_models(devices: list[str], on_status: Optional[Callable[[str], None]] = None) -> list[str]:
    """Unduh semua model default untuk device yang dipilih. Kembalikan daftar yang siap."""
    import subprocess

    from backend import device as device_mod
    from backend.pipeline.diarize import ensure_models

    status = on_status or (lambda _m: None)
    if config.OFFLINE:
        raise VidNoteError("download-models tidak bisa jalan di mode offline")
    done: list[str] = []
    device_mod.setup_cuda_libs()
    from faster_whisper.utils import download_model

    config.WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    for dev in devices:
        m = config.MODELS[dev]
        for name in (m["asr"], m["asr_fallback"]):
            status(f"Whisper {name}")
            download_model(name, cache_dir=str(config.WHISPER_DIR))
            done.append(f"whisper:{name}")
    status("Diarization (sherpa-onnx)")
    ensure_models(config.DIAR_DEFAULT_EMBEDDING, status)
    done.append(f"diarization:{config.DIAR_DEFAULT_EMBEDDING}")
    exe = shutil.which("ollama")
    for dev in devices:
        llm = config.MODELS[dev]["llm"]
        if not exe:
            status(f"Ollama tidak ditemukan, lewati {llm}")
            continue
        status(f"Ollama {llm} (bisa beberapa GB)")
        r = subprocess.run([exe, "pull", llm], timeout=6 * 3600)
        if r.returncode != 0:
            raise VidNoteError(f"ollama pull {llm} gagal")
        done.append(f"ollama:{llm}")
    return done

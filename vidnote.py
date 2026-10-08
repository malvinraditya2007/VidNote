"""VidNote CLI (Fase 0). Video in, notes out.

Contoh:
    python vidnote.py run video.mp4|URL --device gpu|cpu [--lang auto|id|en] [--num-speakers N]
    python vidnote.py download-models [--device gpu|cpu|all]
    python vidnote.py serve [--port 8765]
    python vidnote.py --offline run video.mp4 --device gpu
    python vidnote.py doctor
    python vidnote.py probe video.mp4 [--keep] [--json]
    python vidnote.py fetch "https://youtu.be/VIDEO_ID" [--keep] [--json]
    python vidnote.py audio video.mp4|URL [--out audio.wav] [--keep]
    python vidnote.py asr video.mp4|URL --device gpu|cpu [--lang auto|id|en] [--asr-model NAME] [--out DIR]
    python vidnote.py diarize video.mp4|URL [--num-speakers N] [--threshold T] [--out DIR]
    python vidnote.py segment video.mp4|URL --device gpu|cpu [--lang ..] [--num-speakers N] [--out DIR]
    python vidnote.py subs output\folder [--orientation portrait] [--out DIR]
    python vidnote.py burn video.mp4|URL output\folder --device gpu|cpu [--out hasil.mp4]
    python vidnote.py summarize output\folder --device gpu|cpu [--llm qwen3:4b]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXIT_OK, EXIT_ERROR, EXIT_REJECTED = 0, 1, 2


def cmd_doctor(_args: argparse.Namespace) -> int:
    from backend import doctor

    status = doctor.print_report(doctor.run_all())
    return EXIT_ERROR if status == doctor.FAIL else EXIT_OK


def _print_probe(info, as_json: bool) -> None:
    from backend.pipeline.probe import fmt_duration

    if as_json:
        print(json.dumps(info.to_dict(), indent=2))
        return
    rot = f" (rotasi {info.rotation}°)" if info.rotation else ""
    print(f"  Durasi     : {fmt_duration(info.duration)} ({info.duration:.1f} dtk)")
    print(f"  Resolusi   : {info.width}x{info.height}{rot}")
    print(f"  Orientasi  : {info.orientation}")
    print(f"  Video      : {info.video_codec or '-'}")
    print(f"  Audio      : {info.audio_codec or '-'}")
    print(f"  Container  : {info.container}")
    print(f"  Ukuran     : {info.size_bytes / 1024**2:.1f} MB")


def cmd_probe(args: argparse.Namespace) -> int:
    from backend import jobs
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.receive_file import receive_file

    with jobs.job_context(keep=args.keep) as job:
        try:
            dst, info = receive_file(Path(args.input), job)
        except InputRejected as e:
            if e.info is not None:
                _print_probe(e.info, args.json)
            print("Status: DITOLAK")
            for r in e.reasons:
                print(f"  - {r}")
            return EXIT_REJECTED
        except VidNoteError as e:
            print(f"Error: {e}")
            return EXIT_ERROR
        _print_probe(info, args.json)
        print("Status: VALID")
        if args.keep:
            print(f"Folder job: {job.dir}")
    return EXIT_OK


def _progress_printer():
    state = {"last": -1}

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            pct = int(done * 100 / total) if total else -1
            if pct != state["last"] and pct >= 0:
                state["last"] = pct
                print(f"\r  Mengunduh: {pct:3d}% ({done / 1024**2:.1f} MB)", end="", flush=True)
        elif d.get("status") == "finished":
            state["last"] = -1
            print("\r  Mengunduh: selesai" + " " * 20)

    return hook


def cmd_fetch(args: argparse.Namespace) -> int:
    from backend import jobs
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.fetch_url import ToS_WARNING, fetch_url

    print(ToS_WARNING)
    print()
    with jobs.job_context(keep=args.keep) as job:
        try:
            dst, info, meta = fetch_url(args.url, job, on_progress=_progress_printer(),
                                        cookies=_cookies_from(args))
        except InputRejected as e:
            if e.info is not None:
                _print_probe(e.info, args.json)
            print("Status: DITOLAK")
            for r in e.reasons:
                print(f"  - {r}")
            return EXIT_REJECTED
        except VidNoteError as e:
            print(f"\nError: {e}")
            return EXIT_ERROR
        if args.json:
            print(json.dumps({"meta": meta, "probe": info.to_dict()}, indent=2, ensure_ascii=False))
        else:
            print(f"  Judul      : {meta['title']}")
            print(f"  Channel    : {meta['uploader']}")
            _print_probe(info, False)
        print("Status: VALID")
        if args.keep:
            print(f"File: {dst}")
    return EXIT_OK


def _is_url(s: str) -> bool:
    return s.strip().lower().startswith(("http://", "https://"))


def _cookies_from(args):
    from backend.pipeline.fetch_url import Cookies

    return Cookies.parse(getattr(args, "cookies_from_browser", None), getattr(args, "cookies_file", None))


def load_input(src: str, job, args=None):
    """File lokal atau link YouTube -> (path_input_di_job, ProbeInfo). Raise VidNoteError."""
    if _is_url(src):
        from backend.pipeline.fetch_url import ToS_WARNING, fetch_url

        print(ToS_WARNING)
        cookies = _cookies_from(args) if args else None
        dst, info, meta = fetch_url(src, job, on_progress=_progress_printer(), cookies=cookies)
        print(f"  Judul      : {meta['title']}")
        return dst, info
    from backend.pipeline.receive_file import receive_file

    return receive_file(Path(src), job)


def add_cookie_args(p):
    g = p.add_argument_group("cookie YouTube (atasi blokir 'confirm you're not a bot')")
    g.add_argument("--cookies-from-browser", metavar="BROWSER",
                   help="ambil cookie dari browser: chrome, firefox, edge, brave, ...")
    g.add_argument("--cookies-file", metavar="PATH", help="file cookies.txt (format Netscape)")


def _report_rejected(e) -> int:
    if e.info is not None:
        _print_probe(e.info, False)
    print("Status: DITOLAK")
    for r in e.reasons:
        print(f"  - {r}")
    return EXIT_REJECTED


def cmd_audio(args: argparse.Namespace) -> int:
    import shutil
    import time

    from backend import jobs
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.audio import extract_audio

    with jobs.job_context(keep=args.keep) as job:
        try:
            src, info = load_input(args.input, job, args)
            t0 = time.perf_counter()
            a = extract_audio(src, job, info.duration)
            elapsed = time.perf_counter() - t0
        except InputRejected as e:
            return _report_rejected(e)
        except VidNoteError as e:
            print(f"Error: {e}")
            return EXIT_ERROR
        print(f"  Audio      : {a.sample_rate} Hz, {a.channels} ch, {a.duration:.1f} dtk")
        print(f"  Level      : RMS {a.rms_dbfs} dBFS, peak {a.peak_dbfs} dBFS")
        print(f"  Waktu      : {elapsed:.1f} dtk")
        if a.is_near_silent:
            print("  Peringatan : audio hampir hening, transkrip kemungkinan kosong")
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(a.path, out)
            print(f"  Disimpan   : {out.resolve()}")
        elif args.keep:
            print(f"  Disimpan   : {a.path}")
        print("Status: OK")
    return EXIT_OK


def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _ask_oom(here: str, there: str, auto_yes: bool) -> bool:
    print(f"\n  VRAM tidak cukup untuk {here}.")
    if auto_yes:
        print(f"  --yes: lanjut dengan {there}")
        return True
    try:
        ans = input(f"  Coba lagi dengan {there}? [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "ya", "yes")


def _asr_progress():
    state = {"last": -1}

    def hook(frac: float) -> None:
        pct = int(frac * 100)
        if pct != state["last"]:
            state["last"] = pct
            print(f"\r  Progres    : {pct:3d}%", end="", flush=True)

    return hook


def cmd_asr(args: argparse.Namespace) -> int:
    from backend import jobs
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.asr import transcribe
    from backend.pipeline.audio import extract_audio

    with jobs.job_context(keep=args.keep) as job:
        try:
            src, info = load_input(args.input, job, args)
            a = extract_audio(src, job, info.duration)
            if a.is_near_silent:
                print("  Peringatan : audio hampir hening")
            res = transcribe(
                a.path, args.device, lang=args.lang, model_name=args.asr_model,
                on_oom=lambda h, t: _ask_oom(h, t, args.yes),
                on_status=lambda m: print(f"  {m}..."),
                on_progress=_asr_progress(),
            )
        except InputRejected as e:
            return _report_rejected(e)
        except VidNoteError as e:
            print(f"\nError: {e}")
            return EXIT_ERROR
        print()
        data = res.to_dict()
        job.path("asr.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [f"[{_fmt_ts(s.start)}] {s.text}" for s in res.segments]
        print()
        print("\n".join(lines) if lines else "  (tidak ada ucapan terdeteksi)")
        print()
        d = res.lang
        if d.forced:
            print(f"  Bahasa     : {res.language} (dipaksa --lang)")
        elif res.language:
            print(f"  Bahasa     : {res.language} (id {d.probs.get('id', 0):.2f} / en {d.probs.get('en', 0):.2f})")
        if d.mixed:
            print("  Peringatan : video sepertinya campur Indonesia-Inggris; akurasi bisa turun")
        print(f"  Model      : {res.model} ({res.device.upper()}, {res.compute_type})")
        for fb in res.fallbacks:
            print(f"  Fallback   : {fb}")
        rtf = res.elapsed_sec / res.duration if res.duration else 0
        print(f"  Durasi     : {_fmt_ts(res.duration)} (ucapan {res.speech_sec:.0f} dtk)")
        print(f"  Waktu ASR  : {res.elapsed_sec:.1f} dtk (x{rtf:.2f} durasi, termasuk muat model)")
        if res.vram_peak_mb is not None:
            print(f"  VRAM puncak: +{res.vram_peak_mb} MiB")
        print(f"  Segmen     : {len(res.segments)} dipakai, {len(res.dropped)} dibuang filter")
        for dr in res.dropped[:10]:
            print(f"    - [{_fmt_ts(dr['start'])}] {dr['reason']}: {dr['text'][:60]!r}")
        print("  Catatan    : transkrip buatan AI bisa mengandung kesalahan")
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "asr.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "transcript_raw.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  Disimpan   : {out.resolve()}")
        print("Status: OK")
    return EXIT_OK


def cmd_diarize(args: argparse.Namespace) -> int:
    from backend import jobs
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.audio import extract_audio
    from backend.pipeline.diarize import run_diarization, speaker_label, speaker_stats

    with jobs.job_context(keep=args.keep) as job:
        try:
            src, info = load_input(args.input, job, args)
            a = extract_audio(src, job, info.duration)
            res = run_diarization(
                a.path, backend=args.diarizer, device=getattr(args, "device", "cpu"),
                num_speakers=args.num_speakers, threshold=args.threshold, embedding=args.embedding,
                on_status=lambda m: print(f"  {m}..."), on_progress=_asr_progress(),
            )
        except InputRejected as e:
            return _report_rejected(e)
        except VidNoteError as e:
            print(f"\nError: {e}")
            return EXIT_ERROR
        print("\n")
        data = res.to_dict()
        job.path("diarization.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        lines = [f"[{_fmt_ts(t.start)} - {_fmt_ts(t.end)}] {speaker_label(t.speaker)}"
                 f"  ({t.end - t.start:.1f} dtk)" for t in res.turns]
        print("\n".join(lines) if lines else "  (tidak ada ucapan terdeteksi)")
        print()
        mode = f"dipaksa {res.forced_speakers}" if res.forced_speakers else f"otomatis, threshold {res.threshold}"
        print(f"  Pembicara  : {res.num_speakers} ({mode})")
        for spk, sec in speaker_stats(res.turns).items():
            print(f"    {speaker_label(spk)}: {sec:.0f} dtk bicara")
        for n in res.notes:
            print(f"  Catatan    : {n}")
        rtf = res.elapsed_sec / res.duration if res.duration else 0
        print(f"  Embedding  : {res.embedding}")
        print(f"  Waktu      : {res.elapsed_sec:.1f} dtk (x{rtf:.2f} durasi, CPU)")
        print("  Catatan    : pemisahan pembicara buatan AI bisa keliru (terutama saat bicara bersamaan)")
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "diarization.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  Disimpan   : {out.resolve()}")
        print("Status: OK")
    return EXIT_OK


def cmd_segment(args: argparse.Namespace) -> int:
    import time

    from backend import jobs
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.asr import transcribe
    from backend.pipeline.audio import extract_audio
    from backend.pipeline.diarize import run_diarization, speaker_label
    from backend.pipeline.segment import build_transcript, transcript_text

    def dump(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    with jobs.job_context(keep=args.keep) as job:
        try:
            t0 = time.perf_counter()
            src, info = load_input(args.input, job, args)
            a = extract_audio(src, job, info.duration)
            print("[1/2] Transkripsi")
            res = transcribe(
                a.path, args.device, lang=args.lang, model_name=args.asr_model,
                on_oom=lambda h, t: _ask_oom(h, t, args.yes),
                on_status=lambda m: print(f"  {m}..."), on_progress=_asr_progress(),
            )
            print(f"\n  Selesai {res.elapsed_sec:.0f} dtk ({res.model}, {res.language or '-'})")
            print("[2/2] Pemisahan pembicara")
            dia = run_diarization(
                a.path, backend=args.diarizer, device=args.device,
                num_speakers=args.num_speakers, threshold=args.threshold, embedding=args.embedding,
                on_status=lambda m: print(f"  {m}..."), on_progress=_asr_progress(),
            )
            print(f"\n  Selesai {dia.elapsed_sec:.0f} dtk")
        except InputRejected as e:
            return _report_rejected(e)
        except VidNoteError as e:
            print(f"\nError: {e}")
            return EXIT_ERROR

        data = build_transcript(res.segments, dia.turns, info.orientation)
        data.update(language=res.language, duration=res.duration,
                    asr_model=res.model, device=res.device, embedding=dia.embedding)
        text = transcript_text(data)
        files = {
            "asr.json": dump(res.to_dict()),
            "diarization.json": dump(dia.to_dict()),
            "transcript.json": dump(data),
            "transcript.txt": text,
        }
        for name, content in files.items():
            job.path(name).write_text(content, encoding="utf-8")

        print()
        print(text)
        print(f"  Bahasa     : {res.language or '-'}" + ("  (campur id/en)" if res.lang.mixed else ""))
        print(f"  Orientasi  : {info.orientation} (maks {2 * (42 if info.orientation == 'landscape' else 26)} karakter/cue)")
        print(f"  Pembicara  : {data['num_speakers']}")
        for s in data["speakers"]:
            print(f"    {s['label']}: {s['seconds']:.0f} dtk")
        for n in dia.notes:
            print(f"  Catatan    : {n}")
        print(f"  Cue        : {len(data['cues'])}, paragraf {len(data['paragraphs'])}")
        print(f"  Total      : {time.perf_counter() - t0:.0f} dtk")
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            for name, content in files.items():
                (out / name).write_text(content, encoding="utf-8")
            print(f"  Disimpan   : {out.resolve()}")
        print("Status: OK")
    return EXIT_OK


def cmd_subs(args: argparse.Namespace) -> int:
    from backend.pipeline.subtitle import build_all

    src = Path(args.transcript)
    if src.is_dir():
        src = src / "transcript.json"
    if not src.is_file():
        print(f"Error: transcript.json tidak ditemukan: {src}")
        print("  Buat dulu dengan: python vidnote.py segment <video> --device gpu --out <folder>")
        return EXIT_ERROR
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"Error: transcript.json tidak valid: {e}")
        return EXIT_ERROR
    if args.orientation:
        data["orientation"] = args.orientation
    out = Path(args.out) if args.out else src.parent
    out.mkdir(parents=True, exist_ok=True)
    files = build_all(data, title=args.title or "VidNote")
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
        print(f"  {name:<9}: {(out / name).resolve()}")
    print(f"  Orientasi : {data.get('orientation', 'landscape')}, "
          f"{len(data.get('cues') or [])} cue, {data.get('num_speakers', 1)} pembicara")
    print("Status: OK")
    return EXIT_OK


def cmd_burn(args: argparse.Namespace) -> int:
    import shutil

    from backend import config, device as device_mod, jobs
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.burn import SUBS_NAME, burn

    subs = Path(args.subs)
    if subs.is_dir():
        subs = subs / SUBS_NAME
    if not subs.is_file() or subs.suffix.lower() != ".ass":
        print(f"Error: file .ass tidak ditemukan: {subs}")
        return EXIT_ERROR
    if subs.stat().st_size > config.MAX_SUBS_BYTES:
        print("Error: file subtitle terlalu besar")
        return EXIT_ERROR
    out = Path(args.out) if args.out else subs.parent / "video_hardsub.mp4"

    with jobs.job_context(keep=args.keep) as job:
        try:
            if args.device == "gpu" and not device_mod.gpu_available():
                raise VidNoteError("GPU tidak tersedia. Pakai --device cpu.")
            src, info = load_input(args.input, job, args)
            shutil.copyfile(subs, job.path(SUBS_NAME))
            res = burn(src, job, args.device, info.duration, info.audio_codec,
                       on_status=lambda m: print(f"  {m}..."), on_progress=_asr_progress())
        except InputRejected as e:
            return _report_rejected(e)
        except VidNoteError as e:
            print(f"\nError: {e}")
            return EXIT_ERROR
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res.path, out)
        rtf = res.elapsed_sec / info.duration if info.duration else 0
        print("\n")
        print(f"  Encoder    : {res.encoder}" + ("  (fallback)" if res.fell_back else ""))
        if res.note:
            print(f"  Catatan    : {res.note}")
        print(f"  Waktu      : {res.elapsed_sec:.1f} dtk (x{rtf:.2f} durasi)")
        print(f"  Ukuran     : {out.stat().st_size / 1024**2:.1f} MB")
        print(f"  Disimpan   : {out.resolve()}")
        print("Status: OK")
    return EXIT_OK


def cmd_summarize(args: argparse.Namespace) -> int:
    from backend.errors import VidNoteError
    from backend.pipeline.summarize import summarize, summary_markdown

    src = Path(args.transcript)
    if src.is_dir():
        src = src / "transcript.json"
    if not src.is_file():
        print(f"Error: transcript.json tidak ditemukan: {src}")
        return EXIT_ERROR
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        s = summarize(data, args.device, model=args.llm, on_status=lambda m: print(f"  {m}..."))
    except (OSError, ValueError) as e:
        print(f"Error: transcript.json tidak valid: {e}")
        return EXIT_ERROR
    except VidNoteError as e:
        print(f"Error: {e}")
        return EXIT_ERROR
    md = summary_markdown(s, args.title or "")
    out = Path(args.out) if args.out else src.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    for w in s.warnings:
        print(f"  Peringatan : {w}")
    print(f"  Model      : {s.model} ({s.device.upper()}), {s.chunks} bagian")
    print(f"  Waktu      : {s.elapsed_sec:.1f} dtk")
    print(f"  Disimpan   : {(out / 'summary.md').resolve()}")
    print("Status: OK")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    from backend import runner
    from backend.errors import InputRejected, VidNoteError
    from backend.pipeline.fetch_url import ToS_WARNING
    from backend.pipeline.probe import fmt_duration

    opts = runner.RunOptions(
        device=args.device, lang=args.lang, asr_model=args.asr_model, llm=args.llm,
        num_speakers=args.num_speakers, threshold=args.threshold, embedding=args.embedding,
        diarizer=args.diarizer, skip_summary=args.no_summary, skip_burn=args.no_burn,
        out_root=Path(args.out) if args.out else runner.config.OUTPUT_DIR, keep=args.keep,
        cookies_from_browser=args.cookies_from_browser, cookies_file=args.cookies_file,
    )
    if runner.is_url(args.input):
        print(ToS_WARNING)
        print()
    print("Proses berat: colok charger, pastikan ventilasi baik, tutup aplikasi berat. Ctrl+C untuk batal.")
    order = {s: i + 1 for i, s in enumerate(runner.STAGES)}
    state = {"stage": None, "pct": -1}

    def on_stage(st: str, msg: str) -> None:
        if state["stage"] != st:
            if state["pct"] >= 0:
                print()
            state.update(stage=st, pct=-1)
            print(f"[{order[st]}/{len(order)}] {runner.STAGE_LABEL[st]}")
        if msg != runner.STAGE_LABEL[st]:
            if state["pct"] >= 0:
                print()
                state["pct"] = -1
            print(f"      {msg}...")

    def on_progress(st: str, frac: float) -> None:
        pct = int(frac * 100)
        if pct != state["pct"]:
            state["pct"] = pct
            print(f"\r      {pct:3d}%", end="", flush=True)

    def on_info(i: dict) -> None:
        if state["pct"] >= 0:
            print()
            state["pct"] = -1
        lo, hi = i["estimate_min"]
        print(f"      {i['title']}")
        print(f"      {fmt_duration(i['duration'])}, {i['width']}x{i['height']} {i['orientation']}")
        print(f"      Perkiraan waktu total ({args.device.upper()}): ~{lo}-{hi} menit (kasar)")

    t0 = __import__("time").perf_counter()
    try:
        res = runner.run(args.input, opts, on_stage, on_progress,
                         on_oom=lambda h, t: _ask_oom(h, t, args.yes), on_info=on_info)
    except InputRejected as e:
        print()
        return _report_rejected(e)
    except VidNoteError as e:
        print(f"\nError: {e}")
        return EXIT_ERROR
    total = __import__("time").perf_counter() - t0
    if state["pct"] >= 0:
        print()
    print()
    print("Selesai.")
    print(f"  Folder     : {res.out_dir.resolve()}")
    print(f"  File       : {', '.join(res.files)}")
    print(f"  Bahasa     : {res.info.get('language')}, {res.info.get('num_speakers')} pembicara")
    for k, v in res.info.get("models", {}).items():
        if v:
            print(f"  {k:<11}: {v}")
    print("  Waktu      : " + ", ".join(f"{runner.STAGE_LABEL[k]} {v:.0f}s" for k, v in res.timings.items()))
    print(f"  Total      : {total / 60:.1f} menit")
    for w in res.warnings:
        print(f"  Peringatan : {w}")
    print("  Catatan    : transkrip & ringkasan buatan AI bisa salah; semua diproses lokal.")
    print("Status: OK")
    return EXIT_OK


def cmd_download_models(args: argparse.Namespace) -> int:
    from backend.errors import VidNoteError
    from backend.runner import download_models

    devices = ["gpu", "cpu"] if args.device == "all" else [args.device]
    try:
        done = download_models(devices, on_status=lambda m: print(f"  {m}..."))
    except VidNoteError as e:
        print(f"Error: {e}")
        return EXIT_ERROR
    print("Siap: " + ", ".join(done))
    print("Setelah ini VidNote bisa dijalankan offline: python vidnote.py --offline run video.mp4 ...")
    print("Status: OK")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from backend import security
    from backend.app import create_app

    app = create_app(port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print("VidNote (lokal)")
    print(f"  URL        : {url}")
    print(f"  Token sesi : {security.SESSION_TOKEN}")
    print("  Buka URL di browser. Token disuntik otomatis ke halaman. Ctrl+C untuk berhenti.")
    if args.open:
        _open_browser_when_ready(url, args.port)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return EXIT_OK


def _open_browser_when_ready(url: str, port: int) -> None:
    """Buka browser begitu server siap menerima koneksi (di thread terpisah)."""
    import socket
    import threading
    import time
    import webbrowser

    def wait_and_open():
        for _ in range(100):  # ~10 dtk
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.1)
        webbrowser.open(url)

    threading.Thread(target=wait_and_open, daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vidnote", description="VidNote: video in, notes out.")
    p.add_argument("--offline", action="store_true",
                   help="larang semua unduhan (model harus sudah ada; link YouTube ditolak)")
    sub = p.add_subparsers(dest="command", required=True)

    from backend import config as _cfg

    rn = sub.add_parser("run", help="pipeline lengkap: video -> hardsub, SRT/VTT/ASS, transkrip, ringkasan")
    rn.add_argument("input", help="file video (.mp4 .mov .mkv .webm) atau link YouTube")
    rn.add_argument("--device", choices=["gpu", "cpu"], required=True)
    rn.add_argument("--lang", choices=["auto", "id", "en"], default="auto")
    rn.add_argument("--num-speakers", type=int, help="paksa jumlah pembicara (default: otomatis)")
    rn.add_argument("--threshold", type=float, default=_cfg.DIAR_THRESHOLD)
    rn.add_argument("--embedding", choices=list(_cfg.DIAR_EMBEDDINGS), default=_cfg.DIAR_DEFAULT_EMBEDDING)
    rn.add_argument("--diarizer", choices=list(_cfg.DIAR_BACKENDS), default=_cfg.DIAR_DEFAULT_BACKEND,
                    help="backend pemisah pembicara (pyannote butuh token HF, lihat README)")
    rn.add_argument("--asr-model", help="override model ASR (A/B test)")
    rn.add_argument("--llm", help="override model Ollama (A/B test)")
    rn.add_argument("--no-summary", action="store_true", help="lewati ringkasan")
    rn.add_argument("--no-burn", action="store_true", help="lewati hardsub")
    rn.add_argument("--yes", action="store_true", help="otomatis setuju fallback saat VRAM habis")
    rn.add_argument("--out", help="folder induk output (default: output/)")
    rn.add_argument("--keep", action="store_true", help="simpan folder kerja (untuk debug)")
    add_cookie_args(rn)
    rn.set_defaults(func=cmd_run)

    dm = sub.add_parser("download-models", help="unduh semua model default (untuk pemakaian offline)")
    dm.add_argument("--device", choices=["gpu", "cpu", "all"], default="all")
    dm.set_defaults(func=cmd_download_models)

    sv = sub.add_parser("serve", help="jalankan API lokal + frontend (FastAPI, bind 127.0.0.1)")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--open", action="store_true", help="buka browser otomatis saat server siap")
    sv.set_defaults(func=cmd_serve)

    d = sub.add_parser("doctor", help="cek environment (Python, ffmpeg, Ollama, GPU, RAM, disk)")
    d.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("probe", help="validasi file video (durasi, stream, resolusi, orientasi)")
    pr.add_argument("input", help="path file video (.mp4 .mov .mkv .webm)")
    pr.add_argument("--keep", action="store_true", help="jangan hapus folder job (untuk debug)")
    pr.add_argument("--json", action="store_true", help="output JSON")
    pr.set_defaults(func=cmd_probe)

    fe = sub.add_parser("fetch", help="unduh video YouTube (maks 720p, maks 30 menit) lalu probe")
    fe.add_argument("url", help="link youtube.com/watch?v=..., youtu.be/..., atau /shorts/...")
    fe.add_argument("--keep", action="store_true", help="simpan hasil di folder job (untuk debug)")
    fe.add_argument("--json", action="store_true", help="output JSON")
    add_cookie_args(fe)
    fe.set_defaults(func=cmd_fetch)

    au = sub.add_parser("audio", help="ekstrak audio WAV 16 kHz mono (dinormalisasi)")
    au.add_argument("input", help="file video atau link YouTube")
    au.add_argument("--out", help="salin audio.wav ke path ini")
    au.add_argument("--keep", action="store_true", help="simpan folder job (untuk debug)")
    add_cookie_args(au)
    au.set_defaults(func=cmd_audio)

    asr = sub.add_parser("asr", help="transkripsi (faster-whisper) dengan deteksi bahasa id/en")
    asr.add_argument("input", help="file video atau link YouTube")
    asr.add_argument("--device", choices=["gpu", "cpu"], required=True)
    asr.add_argument("--lang", choices=["auto", "id", "en"], default="auto")
    asr.add_argument("--asr-model", help="override model (mis. large-v3-turbo, small) untuk A/B test")
    asr.add_argument("--yes", action="store_true", help="otomatis setuju fallback saat VRAM habis")
    asr.add_argument("--out", help="folder untuk menyimpan asr.json + transcript_raw.txt")
    asr.add_argument("--keep", action="store_true", help="simpan folder job (untuk debug)")
    add_cookie_args(asr)
    asr.set_defaults(func=cmd_asr)

    from backend import config as _cfg

    di = sub.add_parser("diarize", help="pisahkan pembicara (sherpa-onnx, CPU)")
    di.add_argument("input", help="file video atau link YouTube")
    di.add_argument("--num-speakers", type=int, help="paksa jumlah pembicara (default: otomatis)")
    di.add_argument("--threshold", type=float, default=_cfg.DIAR_THRESHOLD,
                    help="threshold clustering; menurut dokumentasi lebih besar = lebih sedikit "
                         "pembicara, tapi di tes nyata tidak selalu "
                         f"(default {_cfg.DIAR_THRESHOLD})")
    di.add_argument("--embedding", choices=list(_cfg.DIAR_EMBEDDINGS), default=_cfg.DIAR_DEFAULT_EMBEDDING,
                    help="model embedding pembicara (sherpa; untuk A/B test)")
    di.add_argument("--diarizer", choices=list(_cfg.DIAR_BACKENDS), default=_cfg.DIAR_DEFAULT_BACKEND,
                    help="backend: sherpa (default) atau pyannote (butuh token HF)")
    di.add_argument("--device", choices=["gpu", "cpu"], default="cpu", help="pyannote bisa pakai GPU")
    di.add_argument("--out", help="folder untuk menyimpan diarization.json")
    di.add_argument("--keep", action="store_true", help="simpan folder job (untuk debug)")
    add_cookie_args(di)
    di.set_defaults(func=cmd_diarize)

    se = sub.add_parser("segment", help="ASR + diarization -> transkrip per pembicara (transcript.json/txt)")
    se.add_argument("input", help="file video atau link YouTube")
    se.add_argument("--device", choices=["gpu", "cpu"], required=True)
    se.add_argument("--lang", choices=["auto", "id", "en"], default="auto")
    se.add_argument("--asr-model", help="override model ASR (A/B test)")
    se.add_argument("--yes", action="store_true", help="otomatis setuju fallback saat VRAM habis")
    se.add_argument("--num-speakers", type=int, help="paksa jumlah pembicara")
    se.add_argument("--threshold", type=float, default=_cfg.DIAR_THRESHOLD)
    se.add_argument("--embedding", choices=list(_cfg.DIAR_EMBEDDINGS), default=_cfg.DIAR_DEFAULT_EMBEDDING)
    se.add_argument("--diarizer", choices=list(_cfg.DIAR_BACKENDS), default=_cfg.DIAR_DEFAULT_BACKEND)
    se.add_argument("--out", help="folder untuk menyimpan asr/diarization/transcript")
    se.add_argument("--keep", action="store_true", help="simpan folder job (untuk debug)")
    add_cookie_args(se)
    se.set_defaults(func=cmd_segment)

    su = sub.add_parser("subs", help="buat subs.srt / subs.vtt / subs.ass dari transcript.json")
    su.add_argument("transcript", help="path transcript.json atau folder hasil `segment --out`")
    su.add_argument("--orientation", choices=["landscape", "portrait"], help="override orientasi")
    su.add_argument("--title", help="judul di header ASS")
    su.add_argument("--out", help="folder output (default: folder transcript.json)")
    su.set_defaults(func=cmd_subs)

    bu = sub.add_parser("burn", help="burn subtitle ASS ke video (hardsub MP4 H.264 + AAC)")
    bu.add_argument("input", help="file video atau link YouTube")
    bu.add_argument("subs", help="path subs.ass atau folder yang berisi subs.ass")
    bu.add_argument("--device", choices=["gpu", "cpu"], required=True, help="gpu = NVENC, cpu = libx264")
    bu.add_argument("--out", help="path MP4 hasil (default: <folder subs>/video_hardsub.mp4)")
    bu.add_argument("--keep", action="store_true", help="simpan folder job (untuk debug)")
    add_cookie_args(bu)
    bu.set_defaults(func=cmd_burn)

    sm = sub.add_parser("summarize", help="ringkasan poin + kesimpulan dari transcript.json (Ollama)")
    sm.add_argument("transcript", help="path transcript.json atau folder hasil `segment --out`")
    sm.add_argument("--device", choices=["gpu", "cpu"], required=True)
    sm.add_argument("--llm", help="override model Ollama (mis. qwen3:4b) untuk A/B test")
    sm.add_argument("--title", help="judul di summary.md")
    sm.add_argument("--out", help="folder output (default: folder transcript.json)")
    sm.set_defaults(func=cmd_summarize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offline:
        from backend import config

        config.set_offline(True)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nDibatalkan. Folder kerja sudah dibersihkan.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

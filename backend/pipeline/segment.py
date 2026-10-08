"""Gabung kata (ASR) dengan pembicara (diarization), lalu potong jadi cue.

- Pembicara dipilih per kata berdasarkan overlap waktu terbesar dengan turn
  diarization (pola WhisperX). Kata tanpa overlap memakai turn terdekat
  (<= SEG_WORD_MAX_DIST_SEC), lalu pembicara kata sebelumnya.
- Cue dipotong saat: pembicara berganti, jeda > SEG_MAX_GAP_SEC, teks melewati
  batas karakter (2 baris x 42 landscape / 26 portrait), atau durasi > SEG_MAX_CUE_SEC.
  Akhir kalimat (. ? !) juga jadi titik potong jika cue sudah cukup panjang.
- Paragraf transkrip = cue berurutan dari pembicara sama dengan jeda kecil.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from backend import config
from backend.pipeline.asr import Segment, Word
from backend.pipeline.diarize import Turn, speaker_label


@dataclass
class Cue:
    id: int
    start: float
    end: float
    speaker: int
    text: str


@dataclass
class Paragraph:
    start: float
    end: float
    speaker: int
    text: str


# ---------- kata -> pembicara ----------

def words_of(segments: list[Segment]) -> list[Word]:
    """Semua kata dari segmen ASR. Segmen tanpa word timestamp jadi satu 'kata'."""
    out: list[Word] = []
    for s in segments:
        if s.words:
            out.extend(w for w in s.words if w.word.strip())
        elif s.text.strip():
            out.append(Word(s.start, s.end, s.text, 1.0))
    return out


def assign_speakers(words: list[Word], turns: list[Turn],
                    max_dist: float = config.SEG_WORD_MAX_DIST_SEC) -> list[int]:
    if not turns:
        return [0] * len(words)
    turns = sorted(turns, key=lambda t: t.start)
    out: list[Optional[int]] = []
    for w in words:
        best, best_ov = None, 0.0
        near, near_d = None, float("inf")
        for t in turns:
            if t.start > w.end + max_dist:
                break
            ov = min(w.end, t.end) - max(w.start, t.start)
            if ov > best_ov:
                best, best_ov = t.speaker, ov
            d = max(0.0, t.start - w.end, w.start - t.end)
            if d < near_d:
                near, near_d = t.speaker, d
        if best is None and near_d <= max_dist:
            best = near
        out.append(best)
    # Isi yang kosong: pembicara kata sebelumnya, lalu sesudahnya.
    last = None
    for i, s in enumerate(out):
        if s is None:
            out[i] = last
        else:
            last = s
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return [s if s is not None else 0 for s in out]


# ---------- kata -> cue ----------

def max_cue_chars(orientation: str) -> int:
    line = config.SEG_LINE_CHARS.get(orientation, config.SEG_LINE_CHARS["landscape"])
    return line * config.SEG_MAX_LINES


def build_cues(words: list[Word], speakers: list[int], orientation: str = "landscape") -> list[Cue]:
    limit = max_cue_chars(orientation)
    soft = int(limit * 0.5)
    cues: list[Cue] = []
    cur: list[Word] = []
    cur_spk = -1

    def text_of(ws: list[Word]) -> str:
        return " ".join(w.word.strip() for w in ws)

    def flush() -> None:
        if cur:
            cues.append(Cue(len(cues), round(cur[0].start, 3), round(cur[-1].end, 3),
                            cur_spk, text_of(cur)))
            cur.clear()

    for w, spk in zip(words, speakers):
        tok = w.word.strip()
        if cur:
            cur_text = text_of(cur)
            brk = (
                spk != cur_spk
                or w.start - cur[-1].end > config.SEG_MAX_GAP_SEC
                or len(cur_text) + 1 + len(tok) > limit
                or w.end - cur[0].start > config.SEG_MAX_CUE_SEC
                or (cur_text[-1:] in ".?!" and len(cur_text) >= soft)
            )
            if brk:
                flush()
        if not cur:
            cur_spk = spk
        cur.append(w)
    flush()
    return cues


def build_paragraphs(cues: list[Cue], max_gap: float = config.PARA_MAX_GAP_SEC) -> list[Paragraph]:
    paras: list[Paragraph] = []
    for c in cues:
        if paras and paras[-1].speaker == c.speaker and c.start - paras[-1].end <= max_gap:
            p = paras[-1]
            p.end = c.end
            p.text = f"{p.text} {c.text}"
        else:
            paras.append(Paragraph(c.start, c.end, c.speaker, c.text))
    return paras


# ---------- keluaran ----------

def _ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_transcript(asr_segments: list[Segment], turns: list[Turn], orientation: str) -> dict:
    words = words_of(asr_segments)
    speakers = assign_speakers(words, turns)
    cues = build_cues(words, speakers, orientation)
    # Relabel per urutan kemunculan di transkrip (pembicara tanpa kata hilang).
    order: dict[int, int] = {}
    for c in cues:
        order.setdefault(c.speaker, len(order))
    for c in cues:
        c.speaker = order[c.speaker]
    paras = build_paragraphs(cues)
    secs: dict[int, float] = {}
    for c in cues:
        secs[c.speaker] = secs.get(c.speaker, 0.0) + (c.end - c.start)
    return {
        "orientation": orientation,
        "num_speakers": len(order),
        "speakers": [{"id": k, "label": speaker_label(k), "seconds": round(v, 1)}
                     for k, v in sorted(secs.items())],
        "cues": [asdict(c) for c in cues],
        "paragraphs": [asdict(p) for p in paras],
    }


def transcript_text(data: dict, title: str = "") -> str:
    lines = []
    if title:
        lines += [title, ""]
    lines += ["Transkrip dibuat AI secara lokal; bisa mengandung kesalahan.", ""]
    for p in data["paragraphs"]:
        lines.append(f"[{_ts(p['start'])}] {speaker_label(p['speaker'])}: {p['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

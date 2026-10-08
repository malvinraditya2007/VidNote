"""Generate subtitle SRT, WebVTT, dan ASS dari cue transkrip.

- SRT : teks polos; label "(Pembicara N)" hanya saat pembicara berganti (jika > 1 pembicara).
- VTT : tag voice <v Pembicara N> + blok STYLE warna per pembicara (untuk preview browser).
- ASS : satu style per pembicara (palet berputar), style berbeda untuk landscape/portrait.
        Dipakai untuk burn hardsub (Task 9).

Teks dari ASR = data tidak tepercaya: karakter override ASS ({ } \\) dan tag
HTML (< > &) di-escape supaya tidak bisa mengubah format/merender markup.
"""
from __future__ import annotations

from backend import config
from backend.pipeline.diarize import speaker_label

# Palet RGB (hex) yang kontras di atas outline hitam. Berputar jika pembicara > 8.
PALETTE = ["FFFFFF", "FFE14D", "4DD8FF", "7CFC8A", "FF8AD8", "FFA94D", "B39DFF", "FF7B7B"]

MIN_CUE_SEC = 0.7

# Style ASS per orientasi. PlayRes = kanvas virtual; libass menskalakan ke resolusi video.
ASS_LAYOUT = {
    "landscape": {"play_x": 1920, "play_y": 1080, "font_size": 56, "margin_v": 60, "margin_h": 90},
    # Portrait: margin bawah tinggi supaya tidak tertutup UI aplikasi short-video.
    "portrait": {"play_x": 1080, "play_y": 1920, "font_size": 66, "margin_v": 380, "margin_h": 60},
}
ASS_FONT = "Arial"


def color_of(speaker: int) -> str:
    return PALETTE[speaker % len(PALETTE)]


# ---------- waktu ----------

def _split(sec: float) -> tuple[int, int, int, int]:
    ms = max(0, int(round(sec * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return h, m, s, ms


def srt_time(sec: float) -> str:
    h, m, s, ms = _split(sec)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(sec: float) -> str:
    h, m, s, ms = _split(sec)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ass_time(sec: float) -> str:
    cs = max(0, int(round(sec * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# ---------- teks ----------

def wrap_lines(text: str, max_chars: int, max_lines: int = config.SEG_MAX_LINES) -> list[str]:
    """Bagi teks jadi <= max_lines baris, sebisa mungkin seimbang panjangnya."""
    text = " ".join(text.split())
    if len(text) <= max_chars or max_lines < 2:
        return [text]
    words = text.split(" ")
    best, best_score = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        over = max(0, len(a) - max_chars) + max(0, len(b) - max_chars)
        score = (over, abs(len(a) - len(b)))
        if best_score is None or score < best_score:
            best, best_score = [a, b], score
    return best or [text]


def escape_ass(text: str) -> str:
    # ASS tidak punya escape resmi; ganti dengan karakter lebar penuh yang mirip.
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ")


def escape_vtt(text: str) -> str:
    text = text.replace("-->", "→")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize_cues(cues: list[dict]) -> list[dict]:
    """Urutkan, buang cue kosong, perpanjang cue sangat pendek tanpa menabrak cue berikutnya."""
    cs = sorted((dict(c) for c in cues if str(c.get("text", "")).strip()), key=lambda c: c["start"])
    for i, c in enumerate(cs):
        if c["end"] - c["start"] < MIN_CUE_SEC:
            limit = cs[i + 1]["start"] if i + 1 < len(cs) else c["start"] + MIN_CUE_SEC
            c["end"] = max(c["end"], min(c["start"] + MIN_CUE_SEC, limit))
        if i + 1 < len(cs) and c["end"] > cs[i + 1]["start"]:
            c["end"] = cs[i + 1]["start"]
    return [c for c in cs if c["end"] > c["start"]]


def _line_chars(orientation: str) -> int:
    return config.SEG_LINE_CHARS.get(orientation, config.SEG_LINE_CHARS["landscape"])


# ---------- format ----------

def to_srt(cues: list[dict], orientation: str, num_speakers: int) -> str:
    out, prev = [], None
    for n, c in enumerate(normalize_cues(cues), 1):
        lines = wrap_lines(c["text"], _line_chars(orientation))
        if num_speakers > 1 and c["speaker"] != prev:
            lines[0] = f"({speaker_label(c['speaker'])}) {lines[0]}"
        prev = c["speaker"]
        out.append(f"{n}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def to_vtt(cues: list[dict], orientation: str, num_speakers: int) -> str:
    cues = normalize_cues(cues)
    speakers = sorted({c["speaker"] for c in cues})
    head = ["WEBVTT", "", "STYLE"]
    head += [f'::cue(v[voice="{speaker_label(s)}"]) {{ color: #{color_of(s)}; }}' for s in speakers]
    head += ["", "NOTE Dibuat oleh VidNote secara lokal; transkrip AI bisa mengandung kesalahan.", ""]
    body = []
    for c in cues:
        lines = [escape_vtt(x) for x in wrap_lines(c["text"], _line_chars(orientation))]
        body.append(f"{vtt_time(c['start'])} --> {vtt_time(c['end'])} line:85%\n"
                    f"<v {speaker_label(c['speaker'])}>" + "\n".join(lines) + "\n")
    return "\n".join(head) + "\n" + "\n".join(body)


def _ass_color(rgb: str, alpha: str = "00") -> str:
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"&H{alpha}{b}{g}{r}".upper()


def to_ass(cues: list[dict], orientation: str, num_speakers: int, title: str = "VidNote") -> str:
    lay = ASS_LAYOUT.get(orientation, ASS_LAYOUT["landscape"])
    cues = normalize_cues(cues)
    speakers = sorted({c["speaker"] for c in cues}) or [0]
    styles = []
    for s in speakers:
        styles.append(
            f"Style: S{s},{ASS_FONT},{lay['font_size']},{_ass_color(color_of(s))},&H000000FF,"
            f"&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,"
            f"{lay['margin_h']},{lay['margin_h']},{lay['margin_v']},1"
        )
    events = []
    for c in cues:
        lines = [escape_ass(x) for x in wrap_lines(c["text"], _line_chars(orientation))]
        events.append(
            f"Dialogue: 0,{ass_time(c['start'])},{ass_time(c['end'])},S{c['speaker']},"
            f"{speaker_label(c['speaker'])},0,0,0,," + r"\N".join(lines)
        )
    safe_title = escape_ass(title).replace(",", " ")[:120]
    return "\n".join([
        "[Script Info]",
        f"Title: {safe_title}",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {lay['play_x']}",
        f"PlayResY: {lay['play_y']}",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
        "",
    ])


def build_all(transcript: dict, title: str = "VidNote") -> dict[str, str]:
    """transcript = isi transcript.json. Kembalikan {nama_file: isi}."""
    cues = transcript.get("cues") or []
    orientation = transcript.get("orientation", "landscape")
    n = int(transcript.get("num_speakers") or 1)
    return {
        "subs.srt": to_srt(cues, orientation, n),
        "subs.vtt": to_vtt(cues, orientation, n),
        "subs.ass": to_ass(cues, orientation, n, title),
    }

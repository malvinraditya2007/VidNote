from backend.pipeline.asr import Segment, Word
from backend.pipeline.diarize import Turn, absorb_minor_speakers, count_speakers
from backend.pipeline.segment import (
    assign_speakers, build_cues, build_paragraphs, build_transcript, max_cue_chars,
    transcript_text, words_of,
)


def W(s, e, t):
    return Word(s, e, " " + t, 0.9)


# ---------- pembicara kecil ----------

def test_absorb_minor_speaker_into_nearest():
    turns = [Turn(0, 60, 0), Turn(60.2, 61.0, 2), Turn(61.5, 120, 1), Turn(130, 130.8, 3)]
    out, n, limit = absorb_minor_speakers(turns)
    assert n == 2 and count_speakers(out) == 2
    assert limit >= 3.0


def test_absorb_keeps_real_speakers():
    turns = [Turn(0, 30, 0), Turn(30, 40, 1)]
    out, n, _ = absorb_minor_speakers(turns)
    assert n == 0 and out == turns


def test_absorb_single_speaker_noop():
    out, n, _ = absorb_minor_speakers([Turn(0, 1, 0)])
    assert n == 0 and len(out) == 1


# ---------- kata -> pembicara ----------

def test_assign_by_max_overlap():
    turns = [Turn(0, 2.0, 0), Turn(2.0, 4.0, 1)]
    words = [W(0.5, 1.0, "a"), W(1.8, 2.6, "b"), W(1.5, 2.1, "c"), W(3, 3.5, "d")]
    assert assign_speakers(words, turns) == [0, 1, 0, 1]


def test_assign_no_overlap_uses_nearest_then_previous():
    turns = [Turn(0, 1, 0), Turn(5, 6, 1)]
    words = [W(1.3, 1.6, "near0"), W(3.0, 3.2, "far"), W(4.5, 4.8, "near1")]
    assert assign_speakers(words, turns) == [0, 0, 1]


def test_assign_without_turns():
    assert assign_speakers([W(0, 1, "a")], []) == [0]


# ---------- cue ----------

def test_cue_breaks_on_speaker_change_and_gap():
    words = [W(0, .3, "Halo"), W(.4, .7, "semua."), W(.8, 1, "Hai"), W(2.0, 2.3, "juga")]
    cues = build_cues(words, [0, 0, 1, 1])
    assert [(c.speaker, c.text) for c in cues] == [(0, "Halo semua."), (1, "Hai"), (1, "juga")]


def test_cue_respects_char_limit_portrait():
    words = [W(i * .3, i * .3 + .25, "kata") for i in range(30)]
    cues = build_cues(words, [0] * 30, "portrait")
    assert len(cues) > 1
    assert all(len(c.text) <= max_cue_chars("portrait") for c in cues)


def test_cue_max_duration():
    words = [W(i * .45, i * .45 + .4, "x") for i in range(40)]
    cues = build_cues(words, [0] * 40, "landscape")
    assert all(c.end - c.start <= 7.0 + 0.5 for c in cues)


def test_paragraphs_merge_same_speaker():
    words = [W(0, .3, "Satu."), W(.4, .7, "Dua")]
    cues = build_cues(words, [0, 0])
    cues += build_cues([W(1, 1.2, "Tiga")], [1])
    paras = build_paragraphs(cues)
    assert [p.speaker for p in paras] == [0, 1]


def test_build_transcript_relabels_and_text():
    segs = [
        Segment(0, 0, 2, "Halo apa kabar?", [W(0, .5, "Halo"), W(.6, 1, "apa"), W(1.1, 2, "kabar?")], -.2, .01, 1.2),
        Segment(1, 2.2, 3, "Baik.", [W(2.2, 3, "Baik.")], -.2, .01, 1.2),
    ]
    turns = [Turn(0, 2.1, 5), Turn(2.1, 3.1, 2)]
    data = build_transcript(segs, turns, "landscape")
    assert data["num_speakers"] == 2
    assert [c["speaker"] for c in data["cues"]] == [0, 1]
    txt = transcript_text(data)
    assert "[00:00] Pembicara 1: Halo apa kabar?" in txt and "Pembicara 2: Baik." in txt


def test_words_of_falls_back_to_segment():
    segs = [Segment(0, 0, 1, "tanpa kata", [], -.2, .01, 1.0)]
    assert [w.word for w in words_of(segs)] == ["tanpa kata"]

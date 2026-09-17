"""Tests for abc_score: the ABC subset parser (pure, no Qt/GPU)."""
import pytest

from abc_score import (
    Score,
    load_score,
    parse_score,
)


SAMPLE = """X:1
T:
M:4/4
L:1/32
Q:1/4=140
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:F#m
% intro
V: Vocal
z4"F#m"z24z4|"F#m"z32|
V: Ins
z4F4F4z16z4|F4F4F4F4F4F4F4|
% verse
V: Vocal
"D"z32|"Bm"z32|
w: la la
"""


def test_headers_tempo_key_title():
    score = parse_score(SAMPLE)
    assert score.bpm == pytest.approx(140.0)
    assert score.key == "F#m"
    assert isinstance(score, Score)


def test_sections_split_at_comment_lines():
    score = parse_score(SAMPLE)
    assert [s.name for s in score.sections] == ["intro", "verse"]
    assert score.sections[0].start == pytest.approx(0.0)
    assert score.sections[1].start > 0.0


def test_rests_advance_but_emit_nothing():
    score = parse_score("X:1\nL:1/8\nK:C\nz4 C2|\n")
    assert len(score.notes) == 1
    # z4 = 4 eighths = 2 quarters, so C starts at beat 2.
    assert score.notes[0].start == pytest.approx(2.0)
    assert score.notes[0].dur == pytest.approx(1.0)  # C2 = 1 quarter


def test_octaves_and_middle_c():
    score = parse_score("X:1\nL:1/4\nK:C\nC, C C c c'|\n")
    assert [n.midi for n in score.notes] == [48, 60, 60, 72, 84]


def test_key_signature_and_explicit_accidentals():
    # K:F: Bb. K:G with =B natural.
    assert parse_score("X:1\nL:1/4\nK:F\nB|\n").notes[0].midi == 70  # Bb4
    assert parse_score("X:1\nL:1/4\nK:C\n_B|\n").notes[0].midi == 70
    assert parse_score("X:1\nL:1/4\nK:G\n=B|\n").notes[0].midi == 71  # B4
    assert parse_score("X:1\nL:1/4\nK:C\n^F|\n").notes[0].midi == 66  # F#4
    # K:F#m: F# and C#.
    score = parse_score("X:1\nL:1/4\nK:F#m\nF C|\n")
    assert [n.midi for n in score.notes] == [66, 61]


def test_fractional_lengths():
    score = parse_score("X:1\nL:1/8\nK:C\nA/2 A/ A3/2 A2/|\n")
    durs = [n.dur for n in score.notes]
    assert durs == pytest.approx([0.25, 0.25, 0.75, 0.5])


def test_chord_symbols_recorded_not_played():
    score = parse_score('X:1\nL:1/8\nK:C\n"Dm"F2"G7"G2|\n')
    assert [n.midi for n in score.notes] == [65, 67]
    assert [(c.beat, c.name) for c in score.chords] == [
        (0.0, "Dm"), (1.0, "G7")]


def test_clusters_sound_simultaneously():
    score = parse_score("X:1\nL:1/4\nK:C\n[CEG]|\n")
    assert [(n.midi, n.start, n.dur) for n in score.notes] == [
        (60, 0.0, 1.0), (64, 0.0, 1.0), (67, 0.0, 1.0)]
    assert score.total_beats == pytest.approx(1.0)


def test_voices_merge_onto_one_timeline():
    score = parse_score("X:1\nL:1/4\nK:C\n% a\nV: Alto\nC2|\nV: Tenor\nE2|\n")
    # Sequential staves share the timeline: the second voice's notes land
    # after the first voice's (file order), not on top of them.
    assert [(n.midi, n.start) for n in score.notes] == [(60, 0.0), (64, 2.0)]


def test_lyrics_headers_and_bars_skipped():
    # ">" is lenient noise: A and B both sound (no broken-rhythm glue).
    score = parse_score("X:1\nT: tune\nM:6/8\nL:1/8\nQ:120\nK:D\n|: A>B |\nw: la\n")
    assert [n.midi for n in score.notes] == [69, 71]
    assert score.bpm == pytest.approx(120.0)


def test_midi_clamped():
    score = parse_score("X:1\nL:1/4\nK:C\nC,,,,,,|\n")
    assert score.notes[0].midi == 0


def test_empty_and_garbage_never_raise():
    assert parse_score("").notes == ()
    # No A-G/z letters anywhere: decorations and braces swallow the rest.
    assert parse_score("qrs tuv 123\n!broken\n{T broken\n").notes == ()
    assert parse_score("X:1\nK:C\n").total_beats == pytest.approx(0.0)


def test_total_seconds():
    score = parse_score("X:1\nL:1/4\nQ:120\nK:C\nC2|\n")
    assert score.total_seconds == pytest.approx(1.0)


def test_load_score_reads_file_and_rejects_empty(tmp_path):
    path = tmp_path / "s.abc"
    path.write_text("X:1\nL:1/4\nK:C\nC2|\n", encoding="utf-8")
    assert len(load_score(path).notes) == 1
    empty = tmp_path / "e.abc"
    empty.write_text("X:1\nK:C\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no playable notes"):
        load_score(empty)
    with pytest.raises(OSError):
        load_score(tmp_path / "gone.abc")

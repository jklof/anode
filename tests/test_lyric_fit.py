"""Tests for lyric_fit: parsing, syllables, phrase plans, alignment."""
import pytest

from abc_score import parse_score
from lyric_fit import (
    count_syllables,
    fit_lyrics,
    fit_lyrics_to_score,
    parse_lyrics,
    score_sections,
)

SCORE = """X:1
L:1/4
Q:120
K:C
% verse
C2 D2|z4 E2|
% chorus
A2 A2|
"""


def test_count_syllables():
    assert count_syllables("singer") == 2
    assert count_syllables("understand") == 3
    assert count_syllables("scorching this earth") == 4
    assert count_syllables("love to change") == 3
    assert count_syllables("") == 0
    assert count_syllables("...") == 0


def test_parse_lyrics_tags_and_untagged():
    sections = parse_lyrics("[Verse]\nla\nla\n\n[Chorus]\nna\n")
    assert [(s.tag, len(s.lines)) for s in sections] == [
        ("Verse", 2), ("Chorus", 1)]
    assert parse_lyrics("just words\nmore words\n")[0].tag == "Verse"
    assert parse_lyrics("") == []
    assert parse_lyrics("[Bridge]\n")[0].lines == ()


def test_score_sections_split_phrases_on_gaps():
    score = parse_score(SCORE)
    plans = score_sections(score)
    assert [(p.name, len(p.phrases), p.instrumental) for p in plans] == [
        ("", 0, True), ("verse", 2, False), ("chorus", 1, False)]
    assert [p.slots for p in score_sections(score)[1].phrases] == [2, 1]


def test_score_sections_empty_score():
    score = parse_score("X:1\nK:C\n")
    plans = score_sections(score)
    assert len(plans) == 1 and plans[0].instrumental


def test_exact_fit_no_warnings():
    lyrics = "[Verse]\nla la\nlo\n\n[Chorus]\nna na\n"
    result = fit_lyrics_to_score(lyrics, parse_score(SCORE))
    assert result.warnings == ()
    assert result.dropped == () and result.repeated == 0
    assert result.coverage == pytest.approx(1.0)
    assert "[chorus]\nna na\n" in result.text


def test_verse_overflow_drops_with_warning():
    lyrics = "[Verse]\na\nb\nc\n\n[Chorus]\nna\n"
    result = fit_lyrics_to_score(lyrics, parse_score(SCORE))
    # Verse has 2 phrases for 3 lines: the third line drops.
    assert "c" in result.dropped
    assert any("dropped 1" in w for w in result.warnings)
    assert result.repeated == 0


def test_chorus_shortfall_repeats():
    score = parse_score("X:1\nL:1/4\nK:C\n% chorus\nA1 z1 B1 z1 C1|\n")
    result = fit_lyrics_to_score("[Chorus]\nna\n", score)
    assert result.repeated == 2  # 1 line stretched over 3 phrases
    assert result.coverage == pytest.approx(1.0)
    assert result.text.count("na\n") == 3


def test_verse_shortfall_warns_not_invents():
    score = parse_score("X:1\nL:1/4\nK:C\n% verse\nA1 z1 B1 z1 C1|\n")
    result = fit_lyrics_to_score("[Verse]\nla\n", score)
    assert result.coverage == pytest.approx(1 / 3)
    assert any("unsung" in w for w in result.warnings)
    assert result.repeated == 0


def test_instrumental_sections_stay_empty():
    score = parse_score("X:1\nL:1/4\nK:C\n% intro\nz4|\n% verse\nC2|\n")
    result = fit_lyrics_to_score("[Verse]\nla la\n", score)
    assert "[intro]\n\n[verse]\nla la\n" in result.text
    assert result.coverage == pytest.approx(1.0)


def test_leftover_lyric_sections_warn():
    result = fit_lyrics_to_score("[Verse]\nla\n\n[Bridge]\norphan words here\n",
                                 parse_score("X:1\nL:1/4\nK:C\nC2|\n"))
    assert any("unused" in w for w in result.warnings)


def test_cram_warning_on_dense_line():
    score = parse_score("X:1\nL:1/4\nK:C\n% verse\nC1|\n")
    result = fit_lyrics_to_score(
        "[Verse]\nsupercalifragilisticexpialidocious amazing wonderful\n", score)
    assert any("cram" in w for w in result.warnings)


def test_empty_lyrics_rejected():
    with pytest.raises(ValueError, match="no singable lines"):
        fit_lyrics_to_score("[Verse]\n", parse_score(SCORE))

"""Tests for lyric_fit template planning/rendering and abc_score.select_voice."""
import pytest

from abc_score import parse_score, select_voice
from lyric_fit import plan_template, render_template, suggested_lines

DUAL = """X:1
L:1/32
Q:1/4=140
V: Vocal
V: Ins
K:C
% intro
V: Vocal
z32|z32|
V: Ins
C4C4C4C4C4C4C4C4|C4C4C4C4C4C4C4C4|
% verse
V: Vocal
C4D4E4F4|G4A4B4c4|
V: Ins
E4E4E4E4E4E4E4E4|
"""

GAP = """X:1
L:1/32
Q:1/4=120
K:C
% verse
C4D4E4F4z4G4A4B4c4z16D4E4|
"""

METER = """X:1
M:4/4
L:1/32
Q:1/4=120
V: Vocal
V: Ins
K:C
% verse
V: Vocal
C4D4E4F4|
V: Ins
E4E4E4E4|
V: Vocal
M:2/4
G4A4|
V: Ins
M:2/4
B4c4|
V: Vocal
M:4/4
d4e4f4g4|
V: Ins
M:4/4
a4b4c'4d'4|
"""

PRECHORUS = """X:1
L:1/32
Q:1/4=120
K:C
% pre-chorus
C4D4E4F4|G4A4B4c4|
"""


def test_select_voice_restricts_to_vocal_timeline():
    vocal = parse_score(select_voice(DUAL, "Vocal"))
    merged = parse_score(DUAL)
    # Vocal staff: 8 beats intro rest + 4 beats verse; the Ins staff's
    # 12 beats must not leak in, sections stay intact.
    assert vocal.total_beats == pytest.approx(12.0)
    assert merged.total_beats == pytest.approx(24.0)
    assert [s.name for s in vocal.sections] == ["intro", "verse"]
    assert len(vocal.notes) == 8


def test_select_voice_single_staff_unchanged():
    text = "X:1\nK:C\nC4|\n"
    assert select_voice(text) == text
    assert select_voice("") == ""


def _named(template, name):
    return next(s for s in template.sections if s.name == name)


def test_template_splits_on_gaps_not_breaths():
    verse = _named(plan_template(GAP), "verse")
    assert [p.slots for p in verse.phrases] == [8, 2]
    assert verse.phrases[0].breaths == ((2.0, 0.5),)
    assert verse.phrases[1].gap_before == pytest.approx(2.0)


def test_template_flags_holds():
    template = plan_template("X:1\nL:1/32\nK:C\n% verse\nC4D4C16|\n")
    (phrase,) = _named(template, "verse").phrases
    assert phrase.holds == ((1.0, 2.0),)


def test_template_attributes_meter_changes_once():
    verse = _named(plan_template(METER), "verse")
    # The duplicated Vocal/Ins M: pairs collapse; the header M:4/4 is
    # not a change.
    assert verse.meter_changes == ("M:2/4 @ line 14", "M:4/4 @ line 20")


def test_render_template_marks_instrumental_and_hints_tags():
    text = render_template(plan_template(DUAL))
    assert "[Intro]" in text
    assert "[Verse]" in text
    assert "*(instrumental)*" in text
    assert "8 notes" in text
    assert "1. ..." in text
    hint = render_template(plan_template(PRECHORUS))
    assert "[pre-chorus]" in hint
    assert "not a YuE2 tag" in hint


def test_render_template_empty_score_does_not_crash():
    text = render_template(plan_template(""))
    assert "[Verse]" not in text


def test_suggested_lines_scales_with_budget():
    assert suggested_lines(0) == 1
    assert suggested_lines(8) == 1
    assert suggested_lines(24) == 3
    assert suggested_lines(73) == 8


def test_render_flags_phrase_leading_into_long_bed():
    text = render_template(plan_template(
        "X:1\nL:1/32\nK:C\n% verse\nC4D4z64|\n"))
    assert "instrumental bed" in text
    text = render_template(plan_template(
        "X:1\nL:1/32\nK:C\n% verse\nC4D4z4E4F4|\n"))
    assert "instrumental bed" not in text

"""Fit arbitrary lyrics onto an ABC score's phrase structure.

Direction: words -> fixed melody. Takes any lyric file (tagged sections or
plain lines) plus a parsed :class:`abc_score.Score` and produces YuE2-ready
lyrics: same tags in score order, lines distributed across sung phrases,
repeats written out, instrumental stretches left empty, plus a fit report
saying what was repeated, dropped, or left short. Works for same-song
covers and for fitting one song's words onto another song's melody.

Capacity model: one sung note takes about one syllable, so a phrase's note
count is its syllable budget. Syllables are a vowel-group heuristic
(English-biased, documented below) used only for packing and warnings --
never for rejection.

Pure stdlib, no node state: the fitter node runs this on an NRT worker.
"""

from dataclasses import dataclass, field

from abc_score import Score

#: A rest gap of this many beats (or more) between notes starts a new phrase.
PHRASE_GAP_BEATS = 1.0

#: Sample rate that ``start_sample``/``end_sample`` word timestamps count
#: against (Qwen3 / forced-aligner convention).
WORDS_SAMPLE_RATE = 16000.0


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start_s: float
    end_s: float


def _words_items(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("words", "tokens", "segments"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if any(k in data for k in ("word", "text", "token")):
            return [data]
        return []
    return []


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def parse_words_json(text_or_data, sample_rate: float = WORDS_SAMPLE_RATE):
    """Parse word-timestamp JSON into ascending WordTimestamp items."""
    if isinstance(text_or_data, (bytes, bytearray)):
        text_or_data = bytes(text_or_data).decode("utf-8", errors="replace")
    if isinstance(text_or_data, str):
        try:
            import json
            data = json.loads(text_or_data)
        except Exception as e:
            raise ValueError(f"invalid words JSON: {e}")
    else:
        data = text_or_data
    items = _words_items(data)
    words = []
    rate = sample_rate if sample_rate and sample_rate > 0 else WORDS_SAMPLE_RATE
    for entry in items:
        if not isinstance(entry, dict):
            continue
        text = entry.get("word", entry.get("text", entry.get("token", "")))
        text = str(text).strip() if text is not None else ""
        if not text:
            continue
        start_s = end_s = None
        if "start_sample" in entry or "end_sample" in entry:
            s = _num(entry.get("start_sample"))
            e = _num(entry.get("end_sample"))
            if s is not None:
                start_s = s / rate
            if e is not None:
                end_s = e / rate
        elif "start_ms" in entry or "end_ms" in entry:
            s = _num(entry.get("start_ms"))
            e = _num(entry.get("end_ms"))
            if s is not None:
                start_s = s / 1000.0
            if e is not None:
                end_s = e / 1000.0
        elif "t0" in entry or "t1" in entry:
            s = _num(entry.get("t0"))
            e = _num(entry.get("t1"))
            if s is not None:
                start_s = s
            if e is not None:
                end_s = e
        elif "start" in entry or "end" in entry:
            s = _num(entry.get("start"))
            e = _num(entry.get("end"))
            if s is not None:
                start_s = s / 1000.0 if abs(s) > 10000 else s
            if e is not None:
                end_s = e / 1000.0 if abs(e) > 10000 else e
        if start_s is None:
            continue
        if end_s is None:
            end_s = start_s
        if end_s < start_s:
            end_s = start_s
        words.append(WordTimestamp(text, float(start_s), float(end_s)))
    if not words:
        raise ValueError("no words")
    words.sort(key=lambda w: w.start_s)
    return words


def is_words_json(text: str) -> bool:
    """True when text looks like word-timestamp JSON (never raises)."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        import json
        data = json.loads(stripped)
    except Exception:
        return False
    items = _words_items(data)
    if not items:
        return False
    timing_keys = ("start_sample", "end_sample", "start", "end",
                   "start_ms", "end_ms", "t0", "t1")
    word_keys = ("word", "text", "token")
    for entry in items:
        if not isinstance(entry, dict):
            continue
        if any(k in entry for k in timing_keys) and any(
                k in entry for k in word_keys):
            return True
    return False


def group_words_to_lines(words, gap_s: float = 0.8,
                         max_line_s: float = 6.0, max_words: int = 12):
    """Group word timestamps into plain lyric lines (no score involved)."""
    lines = []
    current: list = []
    line_start = 0.0
    prev_end = 0.0
    for w in words or []:
        if current and ((w.start_s - prev_end) >= gap_s
                        or (w.end_s - line_start) > max_line_s
                        or len(current) >= max_words):
            lines.append(" ".join(x.word for x in current))
            current = []
        if not current:
            line_start = w.start_s
        current.append(w)
        prev_end = w.end_s
    if current:
        lines.append(" ".join(x.word for x in current))
    return lines


@dataclass(frozen=True)
class LyricLine:
    text: str
    syllables: int


@dataclass(frozen=True)
class LyricSection:
    tag: str  # "Verse", "Chorus", ... ("" when untagged)
    lines: tuple


@dataclass(frozen=True)
class ScorePhrase:
    start: float  # beats
    slots: int  # sung notes ~ syllable budget


@dataclass(frozen=True)
class SectionPlan:
    name: str  # score section name ("" for the opening stretch)
    phrases: tuple  # ScorePhrase; empty when instrumental
    instrumental: bool


@dataclass
class FitResult:
    text: str  # fitted lyrics file content
    report: str  # human-readable fit report (also kept next to the file)
    coverage: float  # sung phrases carrying >= 1 line, 0..1
    repeated: int  # hook lines duplicated to fill choruses
    dropped: tuple  # lyric lines with no phrase to ride on
    warnings: tuple


def count_syllables(line):
    """Vowel-group count: runs of aeiouy (plus trailing-e correction).

    English-biased approximation for packing guidance, not phonetics:
    "singer" -> 2, "understand" -> 3, "scorching" -> 2, "yeah" -> 1.
    Empty/wordless lines count 0.
    """
    import re
    words = re.findall(r"[A-Za-z']+", line.lower())
    total = 0
    for word in words:
        word = word.strip("'")
        groups = re.findall(r"[aeiouy]+", word)
        count = len(groups)
        # Silent trailing e ("love", "change") is not a sung syllable;
        # keep it when the word is nothing but the e ("the", "be").
        if word.endswith("e") and count > 1 and not word.endswith("ee"):
            count -= 1
        # "-ed" after t/d ("wanted", "lifted") adds a syllable back.
        if re.search(r"[td]ed$", word):
            count += 1
        total += max(count, 0)
    return total


def parse_lyrics(text):
    """Split lyric text into tagged sections. Blank lines separate; a
    ``[Tag]`` line on its own starts a section. Leading untagged lines
    become a ``Verse`` (most stray lyric files are one)."""
    sections = []
    tag, lines = None, []
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            if lines or tag is not None:
                sections.append((tag, lines))
                tag, lines = None, []
            continue
        if stripped.startswith("[") and stripped.endswith("]") and len(stripped) > 2:
            if lines or tag is not None:
                sections.append((tag, lines))
            tag, lines = stripped[1:-1].strip() or "Verse", []
            continue
        lines.append(stripped)
    if lines or tag is not None:
        sections.append((tag, lines))
    out = []
    for section_tag, section_lines in sections:
        out.append(LyricSection(
            section_tag or "Verse",
            tuple(LyricLine(t, count_syllables(t)) for t in section_lines)))
    return out


def score_sections(score):
    """Reduce a Score to per-section phrase plans in order.

    A phrase is a run of notes separated from the next by a rest gap of at
    least PHRASE_GAP_BEATS. Sections with no notes are instrumental. The
    opening stretch before the first % line keeps name "".
    """
    marks = list(score.sections)
    # Consecutive (name, start, end) spans; the opening stretch before the
    # first % line keeps name "". A scoreless % map yields one whole span.
    spans = []
    prev_name, prev_start = "", 0.0
    for mark in marks:
        spans.append((prev_name, prev_start, mark.start))
        prev_name, prev_start = mark.name, mark.start
    spans.append((prev_name, prev_start, None))
    plans = []
    for name, start, end in spans:
        span = [n for n in score.notes
                if n.start >= start - 1e-9 and (end is None or n.start < end - 1e-9)]
        phrases = []
        run = []
        prev_end = None
        for note in span:
            if run and note.start - prev_end >= PHRASE_GAP_BEATS - 1e-9:
                phrases.append(ScorePhrase(run[0].start, len(run)))
                run = []
            run.append(note)
            prev_end = note.start + note.dur
        if run:
            phrases.append(ScorePhrase(run[0].start, len(run)))
        plans.append(SectionPlan(name, tuple(phrases), not phrases))
    return plans


def _is_chorus(tag):
    return "chorus" in (tag or "").lower()


def fit_lyrics(lyric_sections, section_plans):
    """Align lyric sections onto score section plans in order.

    Instrumental plans emit a bare tag. Each sung plan consumes the next
    lyric section: lines pack 1:1 onto phrases; short choruses repeat
    their hook cyclically; overflow lines drop with a warning; shortfalls
    outside choruses warn (words are never invented). Leftover lyric
    sections warn. Raises ValueError when there are no lyric lines at all.
    """
    if not any(sec.lines for sec in lyric_sections):
        raise ValueError("lyrics have no singable lines")
    out_lines = []
    warnings = []
    dropped = []
    repeated = 0
    covered = 0
    sung_total = 0
    lyric_idx = 0
    n_lyric = len(lyric_sections)

    for plan in section_plans:
        if plan.instrumental and not plan.name:
            continue  # header region before the first % line: no tag at all
        out_lines.append(f"[{plan.name}]" if plan.name else "[Verse]")
        if plan.instrumental:
            out_lines.append("")
            continue
        sung_total += len(plan.phrases)
        if lyric_idx >= n_lyric:
            warnings.append(
                f"[{plan.name or 'Verse'}]: no words left, "
                f"{len(plan.phrases)} phrases unsung")
            out_lines.append("")
            continue
        section = lyric_sections[lyric_idx]
        lyric_idx += 1
        lines = list(section.lines)
        placed = 0
        for phrase_idx, phrase in enumerate(plan.phrases):
            if phrase_idx < len(lines):
                line = lines[phrase_idx]
            elif _is_chorus(section.tag) and lines:
                line = lines[phrase_idx % len(lines)]
                repeated += 1
            else:
                break
            if line.syllables > phrase.slots * 1.5 and phrase.slots > 0:
                warnings.append(
                    f"[{plan.name or 'Verse'}]: {line.syllables} syllables "
                    f"on a {phrase.slots}-note phrase "
                    f"({line.text[:40]!r}…) may cram")
            out_lines.append(line.text)
            placed += 1
        covered += placed
        if len(lines) > len(plan.phrases):
            # Distinct lines past the phrase count drop even in choruses
            # (cyclic repeat only fills shortfalls, never absorbs surplus).
            overflow = lines[len(plan.phrases):]
            dropped.extend(t.text for t in overflow)
            warnings.append(
                f"[{plan.name or 'Verse'}]: dropped "
                f"{len(overflow)} line(s) with no phrase to ride on")
        elif placed < len(plan.phrases) and not _is_chorus(section.tag):
            warnings.append(
                f"[{plan.name or 'Verse'}]: {len(plan.phrases) - placed} "
                f"phrase(s) unsung -- add lines or shorten the score")
        out_lines.append("")

    if lyric_idx < n_lyric:
        rest = [s.tag for s in lyric_sections[lyric_idx:]]
        warnings.append(f"unused lyric sections: {', '.join(rest)}")

    coverage = (covered / sung_total) if sung_total else 1.0
    report_bits = [
        f"sections: {len(section_plans)} score / {n_lyric} lyrics",
        f"coverage: {covered}/{sung_total} phrases ({coverage:.0%})",
        f"repeated hook lines: {repeated}",
        f"dropped lines: {len(dropped)}",
    ]
    report_bits.extend(f"! {w}" for w in warnings)
    text = "\n".join(out_lines).rstrip("\n") + "\n"
    return FitResult(text=text, report="\n".join(report_bits) + "\n",
                     coverage=coverage, repeated=repeated,
                     dropped=tuple(dropped), warnings=tuple(warnings))


def fit_timed_words_to_score(words_or_text, score,
                             sample_rate: float = WORDS_SAMPLE_RATE,
                             gap_s: float = 0.8):
    """Align word timestamps onto the score's sung phrases 1:1."""
    if isinstance(words_or_text, str):
        words = parse_words_json(words_or_text, sample_rate)
    elif isinstance(words_or_text, (list, tuple)) and words_or_text and isinstance(
            words_or_text[0], WordTimestamp):
        words = sorted(words_or_text, key=lambda w: w.start_s)
    else:
        words = parse_words_json(words_or_text, sample_rate)
    if not words:
        raise ValueError("no words")
    bpm = float(getattr(score, "bpm", 0.0) or 0.0) or 120.0
    sec_per_beat = 60.0 / bpm
    plans = score_sections(score)
    marks = list(score.sections)
    spans = []
    prev_start = 0.0
    for mark in marks:
        spans.append((prev_start, mark.start))
        prev_start = mark.start
    spans.append((prev_start, None))
    total = float(getattr(score, "total_beats", 0.0) or 0.0)
    detail = []
    for start_b, end_b in spans:
        span = [n for n in score.notes
                if n.start >= start_b - 1e-9
                and (end_b is None or n.start < end_b - 1e-9)]
        phrases = []
        run: list = []
        run_start = 0.0
        prev_end = None
        for note in span:
            if run and note.start - prev_end >= PHRASE_GAP_BEATS - 1e-9:
                phrases.append((run_start, prev_end, len(run)))
                run = []
            if not run:
                run_start = note.start
            run.append(note)
            prev_end = note.start + note.dur
        if run:
            phrases.append((run_start, prev_end, len(run)))
        sec_end = total if end_b is None else float(end_b)
        detail.append((float(start_b), sec_end, phrases))
    buckets: list = [[] for _ in detail]
    for w in words:
        t_mid = (w.start_s + w.end_s) / 2.0
        placed = None
        for idx, (s_b, e_b, _ph) in enumerate(detail):
            s0, s1 = s_b * sec_per_beat, e_b * sec_per_beat
            if s0 - 0.25 <= t_mid < s1 + 0.25:
                # Last match wins so a word in the tolerance overlap (or in
                # a zero-length opening span) lands in the later section.
                placed = idx
        if placed is None:
            placed = 0 if t_mid < detail[0][0] else len(detail) - 1
        buckets[placed].append(w)
    lyric_sections = []
    for idx, plan in enumerate(plans):
        if plan.instrumental:
            continue
        _, _, phrases = detail[idx]
        if not phrases:
            continue
        section_words = buckets[idx] if idx < len(buckets) else []
        if not section_words:
            lyric_sections.append(LyricSection(plan.name or "Verse", ()))
            continue
        if len(phrases) == 1:
            text = " ".join(w.word for w in section_words)
            lines = [text] if text.strip() else []
        else:
            splits = [((p0[1] + p1[0]) / 2.0) * sec_per_beat
                      for p0, p1 in zip(phrases, phrases[1:])]
            groups: list = [[] for _ in phrases]
            for w in section_words:
                t_mid = (w.start_s + w.end_s) / 2.0
                li = 0
                while li < len(splits) and t_mid >= splits[li]:
                    li += 1
                groups[li].append(w)
            lines = [" ".join(w.word for w in g) for g in groups if g]
            if not lines:
                lines = group_words_to_lines(section_words, gap_s=gap_s)
        lyric_sections.append(LyricSection(
            plan.name or "Verse",
            tuple(LyricLine(t, count_syllables(t)) for t in lines)))
    return fit_lyrics(lyric_sections, plans)


def fit_lyrics_to_score(lyrics_text, score):
    """Parse + align in one call. The node's worker entry point."""
    if isinstance(lyrics_text, str) and is_words_json(lyrics_text):
        return fit_timed_words_to_score(lyrics_text, score)
    return fit_lyrics(parse_lyrics(lyrics_text), score_sections(score))


#: Rest gap (beats) that forces a phrase break in fit templates. Grounded in
#: validated covers: Singer's z12 (1.5 beats) ends phrases, Temple's z4
#: (1.0 beat) and Afterburn's z8 (1.0 beat) are breaths sung straight
#: through. Deliberately wider than PHRASE_GAP_BEATS (1.0), which exists for
#: the node's coarser 1-line-per-phrase packing.
TEMPLATE_BREAK_BEATS = 1.5

#: Smaller gaps are breaths: singable across, but never split a clause there.
TEMPLATE_BREATH_BEATS = 0.5

#: Notes this long (beats) or longer are holds: park sustained words on them.
TEMPLATE_HOLD_BEATS = 2.0

#: A phrase whose next phrase/section end lies this far ahead (beats) ends
#: into an instrumental bed: leaving it empty beats a lone cry (observed:
#: the model fills long post-cry silences with babble).
BED_BEATS = 8.0

#: Tags the YuE2 planner knows. Anything else in a score (e.g. % pre-chorus)
#: needs retagging when the template is filled in.
CANONICAL_TAGS = ("intro", "verse", "chorus", "bridge", "interlude", "outro")


@dataclass(frozen=True)
class TemplatePhrase:
    start: float  # beats from score start
    end: float  # beats (last note end)
    slots: int  # sung-note attacks ~ syllable budget
    gap_before: float  # rest beats since previous phrase end/section start
    breaths: tuple  # (offset_beats, dur_beats) sub-break gaps, sung across
    holds: tuple  # (offset_beats, dur_beats) notes >= hold_beats


@dataclass(frozen=True)
class TemplateSection:
    name: str  # score % name ("" for the opening stretch)
    instrumental: bool
    duration: float  # beats
    seconds: float
    phrases: tuple  # TemplatePhrase
    meter_changes: tuple  # "M:2/4 @ line 94" within this section


@dataclass(frozen=True)
class FitTemplate:
    title: str
    key: str
    bpm: float
    total_seconds: float
    sections: tuple  # TemplateSection


def _meter_events(abc_text):
    """``M:`` field lines as (section_name, occurrence, value, lineno).

    `occurrence` counts same-name ``%`` sections from the top, so an event is
    later attributed to the template section it sits in. Only lines after
    the first ``%`` section count -- the header ``M:`` is the default meter,
    not a change. Both staves of dual-staff transcriptions repeat the same
    switches, so consecutive identical (section, occurrence, value) triples
    collapse to the first (a genuine 4/4→1/4→4/4 round-trip survives, since
    consecutive values differ).
    """
    events = []
    section, occurrence = None, 0
    seen_sections: dict = {}
    for lineno, raw_line in enumerate((abc_text or "").splitlines(), 1):
        line = raw_line.strip()
        if line.startswith("%"):
            name = line[1:].strip()
            if name:
                section = name
                occurrence = seen_sections.get(name.lower(), 0)
                seen_sections[name.lower()] = occurrence + 1
            continue
        if (section is not None and len(line) > 1 and line[1] == ":"
                and line[0] in "mM" and line[0].isalpha()):
            value = line.partition(":")[2].strip()
            if not events or events[-1][:3] != (section, occurrence, value):
                events.append((section, occurrence, value, lineno))
    return events


def plan_template(abc_text, voice="Vocal",
                  break_beats=TEMPLATE_BREAK_BEATS,
                  breath_beats=TEMPLATE_BREATH_BEATS,
                  hold_beats=TEMPLATE_HOLD_BEATS):
    """Reduce an ABC score to a lyric fill-in plan on one staff's timeline.

    Same spans as :func:`score_sections`, but each phrase records its end,
    preceding rest gap, sub-break breaths, and held notes, and each section
    records its duration plus any mid-section meter changes (kink-bar rule:
    never straddle a short meter-change bar with a full lyric line).
    """
    from abc_score import parse_score, select_voice
    score = parse_score(select_voice(abc_text, voice))
    marks = list(score.sections)
    spans = []
    prev_name, prev_start = "", 0.0
    for mark in marks:
        spans.append((prev_name, prev_start, mark.start))
        prev_name, prev_start = mark.name, mark.start
    spans.append((prev_name, prev_start, None))

    meter = _meter_events(abc_text)
    plans = []
    for name, start, end in spans:
        span = [n for n in score.notes
                if n.start >= start - 1e-9 and (end is None or n.start < end - 1e-9)]
        phrases = []
        run, run_start, prev_end, breaths, holds = [], None, None, [], []
        gap_before = 0.0
        for note in span:
            if run and note.start - prev_end >= break_beats - 1e-9:
                phrases.append(TemplatePhrase(
                    run_start, prev_end, len(run), gap_before,
                    tuple(breaths), tuple(holds)))
                gap_before = note.start - prev_end
                run, breaths, holds = [], [], []
            elif run and note.start - prev_end >= breath_beats - 1e-9:
                breaths.append((round(prev_end - run_start, 3),
                                round(note.start - prev_end, 3)))
            if not run:
                run_start = note.start
                if len(phrases) == 0 and prev_end is None:
                    gap_before = round(note.start - start, 3)
            run.append(note)
            if note.dur >= hold_beats - 1e-9:
                holds.append((round(note.start - run_start, 3),
                              round(note.dur, 3)))
            prev_end = note.start + note.dur
        if run:
            phrases.append(TemplatePhrase(
                run_start, prev_end, len(run), gap_before,
                tuple(breaths), tuple(holds)))
        section_end = end if end is not None else score.total_beats
        beats = max(section_end - start, 0.0)
        seconds = beats * 60.0 / score.bpm if score.bpm > 0 else 0.0
        plans.append([name, not phrases, beats, seconds, tuple(phrases), []])
    # Attribute each M: event to the template section it sits in: the event
    # carries its % section's occurrence index, matched against same-name
    # template sections in order. Both lists derive from the same % lines,
    # so every event lands.
    for sec_name, occurrence, value, lineno in meter:
        key = sec_name.lower()
        count = -1
        for plan in plans:
            if plan[0].lower() == key:
                count += 1
                if count == occurrence:
                    plan[5].append(f"M:{value} @ line {lineno}")
                    break
    sections = tuple(
        TemplateSection(name, instrumental, beats, seconds, phrases,
                        tuple(changes))
        for name, instrumental, beats, seconds, phrases, changes in plans)
    return FitTemplate(score.title, score.key, score.bpm,
                       score.total_seconds, sections)


def suggested_lines(slots):
    """Blank-line count for a phrase budget. Rough (~1 line per 9 notes, the
    observed verse density); reshape freely to phrasing."""
    return max(1, round(slots / 9))


def render_template(template):
    """Render a FitTemplate as fill-in-the-blanks lyric skeleton text."""
    lines = [
        f"Fit template (vocal staff only): {template.total_seconds:.1f}s "
        f"total at {template.bpm:g} BPM, key {template.key or '?'}",
        "Budgets are sung-note attacks ~= singable syllables; bias long "
        "lines -1 (melisma absorbs slack, cramming kills). Break lines only "
        "at phrase starts; never split a clause across a breath mark (*).",
        "",
    ]
    for section in template.sections:
        tag = section.name if section.name else "Verse"
        if section.instrumental and not section.name:
            continue  # header region before the first % line
        if tag.lower() in CANONICAL_TAGS:
            # Capitalize ([intro] -> [Intro]) so the skeleton matches YuE2
            # file convention straight away.
            tag = tag[:1].upper() + tag[1:].lower()
        lines.append(f"[{tag}]")
        if section.name and section.name.lower() not in CANONICAL_TAGS:
            lines.append(f"  ! '{section.name}' is not a YuE2 tag -- retag as "
                         f"[Verse] (or [Bridge] for contrasting middles)")
        if section.instrumental:
            lines.append("*(instrumental)*")
            lines.append("")
            continue
        lines.append(f"  ({section.seconds:.1f}s, "
                     f"{sum(p.slots for p in section.phrases)} notes)")
        for change in section.meter_changes:
            lines.append(f"  ! meter change {change} -- end a line before it, "
                         f"or cover it with one complete short exclamation")
        section_start = None
        if section.phrases:
            first = section.phrases[0]
            section_start = first.start - first.gap_before
        section_end = (section_start + section.duration
                       if section_start is not None else None)
        for index, phrase in enumerate(section.phrases):
            start_s = phrase.start * 60.0 / template.bpm if template.bpm > 0 else 0.0
            dur_s = (phrase.end - phrase.start) * 60.0 / template.bpm if template.bpm > 0 else 0.0
            bits = [f"{phrase.slots} notes / {dur_s:.1f}s"]
            if phrase.gap_before >= TEMPLATE_BREAK_BEATS - 1e-9:
                bits.append(f"after {phrase.gap_before:g}-beat rest")
            for offset, dur in phrase.breaths:
                bits.append(f"*breath {dur:g}b @+{offset:g}b (sing across, "
                            f"don't split clauses here)")
            for offset, dur in phrase.holds:
                bits.append(f"HOLD {dur:g}b @+{offset:g}b (park a sustained "
                            f"word here)")
            if section_end is not None:
                following = (section.phrases[index + 1].start
                             if index + 1 < len(section.phrases)
                             else section_end)
                trailing = following - phrase.end
                if trailing >= BED_BEATS - 1e-9:
                    bits.append(f"leads into {trailing:.0f}-beat instrumental "
                                f"bed -- prefer leaving this empty (a lone cry "
                                f"risks fill-in babble)")
            lines.append(f"  phrase @{start_s:.1f}s: {'; '.join(bits)}")
            for num in range(1, suggested_lines(phrase.slots) + 1):
                lines.append(f"    {num}. ... (~{phrase.slots // suggested_lines(phrase.slots)} syllables)")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"

"""ABC subset parser for scores ANode produces and consumes.

Dialect: SheetSage2 transcriptions and YuE2 cover scores — ``X/T/M/L/Q/K/V``
headers, ``%`` section comments, note runs with ``^_=`` accidentals,
``,``/``'`` octaves and fractional lengths, ``z`` rests, ``|`` bar lines,
``"chord"`` symbols, ``[CEG]`` clusters, ``w:`` lyric lines. Staves (``V:``)
are merged sequentially in file order onto one shared timeline.
Everything else
(decorations, grace notes, broken-rhythm ``>``/``<``, slurs) is skipped
leniently and documented below.

Pure stdlib, no node state: the fit-warning helpers, the score viewer, and
the ABC player all build on this instead of regexing ABC themselves.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class NoteEvent:
    midi: int  # clamped 0..127
    start: float  # beats (quarter-note units) from score start
    dur: float  # beats


@dataclass(frozen=True)
class ChordMark:
    beat: float
    name: str


@dataclass(frozen=True)
class SectionMark:
    name: str  # "" for the opening stretch before the first % line
    start: float  # beats


@dataclass(frozen=True)
class Score:
    notes: tuple  # NoteEvent, ascending start
    chords: tuple  # ChordMark, ascending beat
    sections: tuple  # SectionMark, ascending start
    total_beats: float
    bpm: float  # quarter-note beats per minute
    title: str
    key: str  # raw K: header value

    @property
    def total_seconds(self):
        return self.total_beats * 60.0 / self.bpm if self.bpm > 0 else 0.0


# pitch helpers -----------------------------------------------------------

_LETTER_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
# Major key root -> (sharps, flats). Enharmonic spellings resolve to the
# flatter common key (Db over C#, etc.); exotic keys fall back to C.
_MAJOR_ACCIDENTALS = {
    "C": (0, 0), "G": (1, 0), "D": (2, 0), "A": (3, 0), "E": (4, 0),
    "B": (5, 0), "F#": (6, 0), "C#": (7, 0),
    "F": (0, 1), "Bb": (0, 2), "Eb": (0, 3), "Ab": (0, 4),
    "Db": (0, 5), "Gb": (0, 6), "Cb": (0, 7),
}
_PC_TO_MAJOR = {0: "C", 7: "G", 2: "D", 9: "A", 4: "E", 11: "B", 5: "F",
                1: "Db", 3: "Eb", 6: "F#", 8: "Ab", 10: "Bb"}
_PC_OF_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _key_accidentals(key):
    """Map pitch letter -> accidental offset for a K: header value."""
    token = (key or "").split()[0].strip()
    if not token or token.lower() == "none":
        return {}
    # Trailing "m" marks minor keys (Am, F#m, ...): work in the relative
    # major (up a minor third).
    minor = token.endswith("m") and len(token) > 1
    core = token[:-1] if minor else token
    if minor:
        root_pc = (_PC_OF_LETTER.get(core[0].upper(), 0)
                   + (1 if "#" in core else -1 if "b" in core else 0)) % 12
        core = _PC_TO_MAJOR[(root_pc + 3) % 12]
    sharps, flats = _MAJOR_ACCIDENTALS.get(core, (0, 0))
    out = {}
    for letter in _SHARP_ORDER[:sharps]:
        out[letter] = 1
    for letter in _FLAT_ORDER[:flats]:
        out[letter] = -1
    return out


def _parse_length_unit(header_value, default):
    """Parse an L:/M:-style fraction ("1/8", "3/4"), else default."""
    try:
        text = (header_value or "").strip()
        if "/" in text:
            num, _, den = text.partition("/")
            return Fraction(int(num), int(den))
        return Fraction(int(text), 1)
    except (ValueError, ZeroDivisionError):
        return default


def _parse_tempo(header_value):
    """Q: header -> quarter-note bpm. Accepts "140", "1/4=140", "C=120"."""
    text = (header_value or "").strip()
    try:
        if "=" in text:
            unit_text, _, bpm_text = text.partition("=")
            unit = _parse_length_unit(unit_text.strip("C"), Fraction(1, 4))
            return float(Fraction(bpm_text.strip()) * unit / Fraction(1, 4))
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 120.0


# tokenizer ---------------------------------------------------------------

def _read_length(text, pos):
    """Read an ABC length suffix at pos -> (Fraction multiplier, new pos).

    "4" = 4, "/" = 1/2, "/2" = 1/2, "2/" = 1, "3/2" = 3/2. Bare
    (no digits, no slash) = 1.
    """
    start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1
    digits = text[start:pos]
    if pos < len(text) and text[pos] == "/":
        pos += 1
        start2 = pos
        while pos < len(text) and text[pos].isdigit():
            pos += 1
        denom = text[start2:pos]
        num = int(digits) if digits else 1
        return Fraction(num, int(denom) if denom else 2), pos
    if digits:
        return Fraction(int(digits), 1), pos
    return Fraction(1, 1), pos


def _read_note(text, pos):
    """Read one note/rest token -> (kind, payload, new pos).

    kind "note": payload (letter, acc_offset_or_None, octave_shift, length).
    kind "rest": payload length. kind "chord": payload text.
    Returns (None, None, pos) for skippable noise (single char consumed or
    skipped wholesale for {...}/!...!/[...] decorations... no: clusters are
    handled by the caller).
    """
    ch = text[pos]
    if ch in "^_=":
        # Accidental prefix: count carets/underscores, = is natural.
        acc = 0
        natural = False
        while pos < len(text) and text[pos] in "^_=":
            if text[pos] == "^":
                acc += 1
            elif text[pos] == "_":
                acc -= 1
            else:
                natural = True
            pos += 1
        if pos >= len(text):
            return None, None, pos
        ch = text[pos]
        if ch.upper() not in _LETTER_SEMI and ch not in "zZx":
            return None, None, pos
        kind, payload, pos = _read_note_body(text, pos)
        if kind == "note":
            letter, _, octave, length = payload
            payload = (letter, 0 if natural else (acc if acc else None),
                       octave, length)
        return kind, payload, pos
    return _read_note_body(text, pos)


def _read_note_body(text, pos):
    ch = text[pos]
    if ch in "zZx":
        length, pos = _read_length(text, pos + 1)
        return "rest", length, pos
    if ch.upper() in _LETTER_SEMI:
        letter = ch.upper()
        octave = 5 if ch.islower() else 4
        pos += 1
        while pos < len(text) and text[pos] in ",'":
            octave += 1 if text[pos] == "'" else -1
            pos += 1
        length, pos = _read_length(text, pos)
        return "note", (letter, None, octave, length), pos
    return None, None, pos + 1


# main parse --------------------------------------------------------------

_MUSIC_SKIPS = set("|()$.") | {">", "<", "]"}


def parse_score(abc_text):
    """Parse ABC text into a Score. Never raises on malformed input: unknown
    headers, decorations, and unparseable tokens are skipped leniently."""
    unit = Fraction(1, 8)
    bpm = 120.0
    title = ""
    key = "C"
    sections = []
    notes = []
    chords = []
    # Single beat cursor: staves merge sequentially in file order (one shared
    # timeline). Deterministic and predictable for auditioning; the player
    # exposes sections, not voices.
    pos_beats = Fraction(0)

    for raw_line in (abc_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            name = line[1:].strip()
            if name:
                sections.append(SectionMark(name, float(pos_beats)))
            continue
        if len(line) > 1 and line[1] == ":" and line[0].isalpha():
            field, _, value = line.partition(":")
            field = field.upper()
            value = value.strip()
            if field == "L":
                unit = _parse_length_unit(value, unit)
            elif field == "Q":
                bpm = _parse_tempo(value)
            elif field == "K":
                key = value.split()[0] if value.split() else "C"
            elif field == "T":
                title = value
            # X/M/V/w/P and anything else: no beat content.
            continue
        # Music line: tokenize. A "%" starts an inline comment.
        unit_beats = Fraction(4, 1) * unit  # quarter-note beats per L: unit
        key_acc = _key_accidentals(key)
        pos = 0
        while pos < len(line):
            ch = line[pos]
            if ch == "%":
                break
            if ch.isspace() or ch in _MUSIC_SKIPS:
                pos += 1
                continue
            if ch == '"':
                end = line.find('"', pos + 1)
                if end < 0:
                    break
                chords.append(ChordMark(float(pos_beats), line[pos + 1:end]))
                pos = end + 1
                continue
            if ch == "[" and pos + 1 < len(line) and line[pos + 1] != "|":
                # Chord cluster: simultaneous notes, one shared length.
                pos += 1
                members = []
                while pos < len(line) and line[pos] != "]":
                    kind, payload, pos = _read_note(line, pos)
                    if kind == "note":
                        members.append(payload)
                    elif kind == "rest":
                        members.append(None)
                pos += 1  # consume ]
                length, pos = _read_length(line, pos)
                for member in members:
                    if member is None:
                        continue
                    letter, acc, octave, _ = member
                    midi = _to_midi(letter, acc, octave, key_acc)
                    notes.append(NoteEvent(midi, float(pos_beats),
                                           float(length * unit_beats)))
                pos_beats += length * unit_beats
                continue
            if ch == "{":
                end = line.find("}", pos + 1)
                pos = len(line) if end < 0 else end + 1
                continue
            if ch == "!":
                end = line.find("!", pos + 1)
                pos = len(line) if end < 0 else end + 1
                continue
            if ch == "+":
                end = line.find("+", pos + 1)
                pos = len(line) if end < 0 else end + 1
                continue
            kind, payload, pos = _read_note(line, pos)
            if kind == "note":
                letter, acc, octave, length = payload
                midi = _to_midi(letter, acc, octave, key_acc)
                notes.append(NoteEvent(midi, float(pos_beats),
                                       float(length * unit_beats)))
                pos_beats += length * unit_beats
            elif kind == "rest":
                pos_beats += payload * unit_beats
            # kind None: single noisy char already consumed.

    notes.sort(key=lambda e: (e.start, e.midi))
    chords.sort(key=lambda e: e.beat)
    sections.sort(key=lambda e: e.start)
    return Score(notes=tuple(notes), chords=tuple(chords),
                 sections=tuple(sections), total_beats=float(pos_beats),
                 bpm=bpm, title=title, key=key)


def _to_midi(letter, acc, octave, key_acc):
    """Letter + explicit accidental (or key signature) + octave -> MIDI."""
    if acc is None:
        acc = key_acc.get(letter, 0)
    midi = 12 * (octave + 1) + _LETTER_SEMI[letter] + acc
    return max(0, min(127, midi))


def load_score(path):
    """Read an ABC file and parse it. Raises OSError when unreadable and
    ValueError when it holds no playable notes."""
    text = Path(path).read_text(encoding="utf-8")
    score = parse_score(text)
    if not score.notes:
        raise ValueError(f"score has no playable notes: {path}")
    return score


def select_voice(abc_text, voice="Vocal"):
    """Return ABC text with music lines restricted to one ``V:`` staff.

    SheetSage2 transcriptions carry two interleaved staves (``V: Vocal`` and
    ``V: Ins``); :func:`parse_score` merges staves sequentially onto one
    shared timeline, which suits auditioning the full arrangement but doubles
    the timeline for lyric work (words ride one staff only). This keeps
    headers, ``%`` sections, meter/key changes and blank lines, and drops
    music lines sitting under other staves, so the result parses (via
    :func:`parse_score`) to that staff's own timeline. Scores without any
    ``V:`` lines pass through unchanged. Matching is case-insensitive on the
    voice id (first token after ``V:``); an empty `voice` keeps everything.
    """
    if not abc_text:
        return abc_text
    want = (voice or "").strip().lower()
    out = []
    current = None
    seen_voice = False
    for raw_line in abc_text.splitlines():
        line = raw_line.strip()
        if len(line) > 2 and line[0] == "V" and line[1] == ":":
            seen_voice = True
            token = line[2:].strip().split()
            current = token[0].lower() if token else ""
            out.append(raw_line)  # V: lines carry no beats; keep them all
            continue
        if (not line or line.startswith("%")
                or (len(line) > 1 and line[1] == ":" and line[0].isalpha())):
            out.append(raw_line)
            continue
        # Music line: keep for single-staff scores, else only under `voice`.
        if not seen_voice or (current is not None and current.startswith(want)):
            out.append(raw_line)
    return "\n".join(out) + "\n"

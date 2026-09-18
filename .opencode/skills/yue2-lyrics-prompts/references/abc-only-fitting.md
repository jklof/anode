# Fitting lyrics to a melody from ABC only (no reference words)

Use this when you have a melody (ABC or equivalent) but no existing words to
imitate. It derives syllable budgets, stress positions, and section structure
purely from note data. Companion to
[cover-refit-method.md](cover-refit-method.md), which covers the case where
reference lyrics exist; the fitting loop and validation are the same, only
the target pattern's source differs. In ANode, `python
tools/abc_fit_template.py <score.abc>` automates steps 1–5 below.

## 0. What you're extracting

Per lyric line: **(a)** how many syllable slots it has, and **(b)** which
slots fall on strong beats / long notes, so stress placement is right, not
just the count.

## 1. Parse the ABC header (context only)

Read `X:`, `T:`, `M:` (meter), `L:` (default note length), `Q:` (tempo), `K:`
(key). `M:` and `L:` matter most: together they tell you how many
default-length notes make a bar, which locates strong beats. On ANode
dual-staff transcriptions (`V: Vocal` + `V: Ins`), plan on the **vocal staff
only** — the instrumental staff roughly doubles the timeline and its phrases
are not singable slots.

## 2. Tokenize the note stream

Scan the tune body left to right:

| Token | Meaning | Syllable slots? |
|---|---|---|
| Letter `A–G`/`a–g` (±`^ _ =`, ±`,/'`, ±duration) | a note | **Yes — 1** |
| Same pitch tied with `-` into same pitch again | tie (one sustained sound) | **No new slot** — belongs to the previous slot |
| `z`/`x` + duration | rest | **None** — marks breath/phrase gap, not words |
| `[CEG]` bracket | chord | **1** (one syllable against the whole chord) |
| `{...}` before a note | grace notes | **None** — attach to the following main note |
| `>`/`<` between notes | broken rhythm | Each note keeps its own slot; duration only |
| `(3` tuplets | triplet etc. | Each note 1 slot, as normal |
| `\|`, `||`, `|:`, `:|` | bar lines | No slots — but use them to segment bars (step 3) |
| `w:` line, if present | existing lyric alignment | Ground truth for that line — trust it directly |

Rule of thumb: every distinct new attack = 1 syllable. Rests mark where a
line/phrase ends or breathes.

ANode parser notes (know what the tooling actually does): `abc_score` skips
`{...}`, `>`/`<`, and decorations leniently (counts unaffected, durations of
broken pairs approximate — fine for budgets). It does **not** merge ties, so
tied notes count as separate attacks and budgets overstate by the tie count —
the bias-−1 rule below covers this; treat a tie as one fewer usable slot when
you spot one. Monophonic vocal staves make the chord-cluster case moot in
practice. Repeat/volta markers are not expanded (SheetSage scores don't use
them); mirror repeats in words per step 4 instead.

## 3. Group notes into bars, then into lines

Split music into bars on `|`; count slots per bar. Rests ≥ ~1.5 beats end a
lyric phrase; shorter rests are breaths to sing across, never clause-split
points (`tools/abc_fit_template.py` marks both). Then:

- One lyric line per melodic **gesture** (rest-delimited phrase), 4–11
  syllables each; bias long lines −1 (melisma absorbs slack, cramming kills).
- Record strong-beat slots per line: in simple meters the first note of each
  bar is strong (compound: first note of each pulse group); any unusually
  long note is a de facto strong/held slot even mid-bar — park a stressed or
  vowel-heavy syllable there, never a clipped consonant cluster.

Source-line caveat (ANode amendment — getting this wrong breaks fits): on
**composed lead sheets**, one ABC source line usually is one lyric line, as
Claude's original text assumes. On **transcription dumps** (SheetSage
`score-melody.abc`), a `V:` source line is a fixed bar group (typically 4
bars ≈ 2–4 lyric lines), so source-line breaks carry **no** phrasing meaning
— derive lines from rest gaps only, never from source-line breaks.

Output per line: slot count plus a stress sketch, e.g. `[strong, weak, weak,
strong, weak, held(long), weak]`.

## 4. Repeat structure

`|:`, `:|`, `[1`, `[2`, `%` comments, or repeated note sequences mark
recurring sections — reuse fitted lines for repeats unless the file signals a
written variation. Build the section list from `%` comments where present
(retagging non-canonical ones like `% pre-chorus` to `[Verse]`/`[Bridge]` for
YuE2); without markers, treat clearly distinct melodic material (new phrase
shape, different bar count, `||`) as a new section.

## 5. Build the target pattern

A table like:

```
Section       Lines   Slots per line                  Notes
Verse 1       8       8-7-11-9-8-7-9-7                line 3 holds on beat 3: stretchable vowel
Chorus 1      9       11-4-9-6-8-4-10-6-7             lines 2 & 6 are 4-note fragments: hook words only
...
```

## 6. Fit the new lyrics

1. Break content into beats/images independent of line breaks.
2. Per target line, draft, count, adjust to the slot count (add
   modifiers/conjunctions when short; cut articles, contract, or shorten
   synonyms when long).
3. Vowel-friendly holdable words (`soul`, `fire`, `day`) on held/long slots.
4. Hook/rhyme/peak words on strong-beat slots.
5. Keep repeats matched to step-4 structure; keep onset words stressed and
   sonorous after rests, and end lines on strong syllables (drops cluster at
   line edges — see cover-refit-method.md step 6).

## 7. Output format

**A. Plain structured lyric text** (what YuE2 takes): canonical section tags
(`[Verse]`/`[Chorus]`/`[Bridge]`/`[Intro]`/`[Interlude]`/`[Outro]`), one
lyric line per line, words and tags only, repeats written out, instrumental
stretches marked `*(instrumental)*`.

**B. Annotated ABC with `w:` lines** (optional): syllables space-separated,
`-` splitting one word across notes, `_` for ties. Rigorous and
machine-checkable — but note the ANode stack does **not** consume it (lyrics
and ABC travel to the sidecar separately; the parser skips `w:` lines), so
treat B as documentation-only here unless the target tool reads alignment.

## Checklist before handing off

- [ ] Slot counts derived from actual notes, not feel
- [ ] Ties, rests, chords, grace notes handled per the step-2 table (ties =
      one fewer usable slot than the raw attack count)
- [ ] Strong-beat and held-note positions mapped; important words on them
- [ ] Line breaks sit on melodic rests, never mid-clause (transcription
      source-line breaks ignored)
- [ ] Section repeats mirrored in words; asymmetry matched
- [ ] Counts match within ±1 per line (bias −1 on longs), or exceptions
      flagged with reasons
- [ ] Same-seed A/B render against previous best; bracketed drops fixed,
      champion file kept intact

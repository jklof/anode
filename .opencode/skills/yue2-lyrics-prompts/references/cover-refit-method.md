# Cover lyric refit: matching new words to a fixed melody

When new lyric content must sing over an existing song's melody (a cover with a
fixed ABC score, `cot="melody"`), section-level mapping (see
[abc-lyrics-fit.md](abc-lyrics-fit.md)) is necessary but not sufficient. This
doc covers the line-level work: making each line's syllables, breaks, and
sounds land on the melody's gestures. Method contributed from a Freydis /
Heartstone cover over "Temple of the King"; the ANode-marked amendments are
grounded in measured render runs on that cover, not theory.

## Inputs you need

1. **The new lyric content** — the story/images/hooks to keep, even if the
   current line lengths are wrong.
2. **A syllable-accurate reference for the target melody**, in order of
   preference:
   - **Sheet music / ABC / MIDI / MusicXML** — best, because you count note
     onsets per phrase directly instead of guessing from text.
   - **The original song's lyrics**, if you have no notation — assume roughly
     one syllable per melodic note, so the original is a decent proxy for the
     note-rhythm. (Spot-checked on the Temple cover: 0.90–1.09
     syllables/note across sections.)
   - Ideally both — original lyrics for section structure, notation to
     resolve melisma, held notes, and pickup bars.

## Step 1 — Map the song's skeleton

Before touching words, list sections in order with **line counts each**,
including asymmetry: songs are often not symmetric (here chorus 1 was 9 lines,
chorus 2 only 4), and matching that asymmetry matters as much as syllables.

With notation, count note-onsets per vocal phrase (new pitch/attack = one
syllable slot; ties/slurs = same syllable continuing; rests = phrase gaps, not
slots). In ANode, `abc_score.parse_score()` gives per-section note counts on
the vocal staff — use them as the syllable budget. Flag mid-section meter
changes (`M:2/4`, `M:3/4` bars, especially tie-in + rest combinations): a lyric
line straddling a short kink bar gets amputated mid-phrase at the rest and the
lyric pointer derails for everything downstream (observed: collapse from the
exact midpoint of a verse). End a line before the kink, or cover the kink and
its aftermath with one complete short exclamation.

## Step 2 — Exact syllable count per reference line

Count mechanically per line (dictionary syllabification, not guesswork:
"mortal" = 2, "wisdom" = 2, "begun" = 2) and record a compact pattern per
section, e.g. `8-7-11-9-8-7-9-7`.

Counting pitfalls in English: silent e, -ed endings, and diphthongs
(`fire` = 1, not 2; `unclaimed` = 3, not 4). Crude vowel-group counters
overcount these — fine for relative deltas, misleading as absolutes.

Tolerance: match each line within ±1 syllable of its counterpart, and **bias
long lines −1**. A syllable short is absorbed by melisma (natural in ballads);
a syllable over forces cramming, dropped words, and lyric-pointer drift that
compounds into later sections. Section totals should land at or just under the
original's.

## Step 3 — Extract content units from the new lyrics

Break the draft into content beats — short phrases/images independent of line
breaks ("crimson sail," "cursed shore, black winds," "temple ahead"). You will
redistribute these across a different number of lines, so don't get attached to
the draft's breaks.

## Step 4 — Rebuild line by line against the target pattern

For each target line: carry the next content beat(s), hit the syllable count,
preserve the slot's shape (short 4–6 syllable slots are hooks/punches; 10–11
syllable slots carry full clauses). Draft the sentence, count, then adjust by
a word (add a modifier/conjunction/intensifier when short; cut an article,
contract, or swap in a shorter synonym when long). Solve one line at a time.

Hard rule (ANode amendment — violated once, rendered worse from bar one):
**one lyric line per melodic gesture; put line breaks only where the melody
breathes** (rests, held notes, phrase ends). Splitting a clause across two
short lines to hit a syllable total breaks the line↔gesture mapping and the
model fumbles the fragment onsets. If a line must be short, make it a whole
utterance ("Power beyond"), never half a clause ("On a cursed shore /
Where black winds cry" split from one phrase).

## Step 5 — Preserve structural repeats and asymmetry

Mirror the original's repeats (full 9-line chorus once, trimmed 4-line version
later) rather than writing fresh content per repeat. Reuse refitted lines the
way the original reuses its lines.

## Step 6 — Stress, onsets, tails, and singable sounds

Matching counts doesn't guarantee singability. Check each line against the
melody: stressed syllables on strong beats/held notes; melisma landing on
vowel-heavy words, not clipped consonants.

Onsets and tails (ANode amendment — every observed drop in our runs sat at a
line edge, never mid-line): after a melodic rest the model re-enters late and
swallows unstressed openers, and phrase-final tails clip. Defenses, in order
of preference:
- Delete unstressed line-initial fillers (`And`, `with`, `to` at a rest
  boundary carry nothing — cut them).
- If the slot needs the syllables, open with a sacrificial pickup (an
  exclamation or `So`/`Oh` the song won't miss if dropped).
- End lines on sonorous, stressed syllables: pure vowels survive short final
  notes better than diphthongs (`fire` → `roar`); plosives beat sibilants for
  final stops (`yours to claim`, not `not yours`); front-load key nouns so a
  clipped tail costs a filler, not the point.

Phonetic load (ANode amendment): rare proper nouns, consonant clusters, and
alliteration (`long lost to the tides`, `before her burning eyes`) garble
first on small models. Keep story-critical names, simplify the rest
(`Tranicos` → `dead kings`), and break up alliterative stacks.

## Step 7 — Instrumental stretches (ANode amendment)

Rests-only stretches take no words — but mark them explicitly rather than
leaving bare tags. Measured ranking on the Temple cover: an explicit
`*(instrumental)*` line under the tag (best) > empty tag (fine) > a sung cry
into a minute-long solo/fade (the model keeps vocalizing into the silence and
babbles). Never leave a cry line as the only lyric in a long instrumental
bed; move that word into the nearest sung section. Exception: a multi-note
pickup figure that leads directly into the next section (e.g. four pickup
notes into a finale chorus) must carry words — a short sealing line that
closes the old section. Left unsung, the model fills it with echoes of
earlier sections and enters the next chorus off-pointer, with dropped lines
as cascade damage.

## Step 8 — Validate by render, bracket the drops (ANode amendment)

Render same-seed A/B against the previous best (`cot="melody"`, frozen
style): only the lyrics may change between runs — verify seed, style, and ABC
from the kept `request.json` before comparing, or the comparison is void.
Listen with the lyric sheet and bracket every missing/garbled word `[like
this]`. Drops cluster diagnostically — onsets mean rest-boundary/timing
trouble (Step 6 defenses), tails mean the line is still too long for its
gesture (Step 4), whole-section drift means section totals still exceed budget
(Step 2). Fix only the bracketed spots.

Non-compositionality warning (observed, same seed/style/ABC): even small
word swaps can reshuffle the *entire* render, degrading sections whose words
didn't change — the lyric plan is global, not per-line. So never edit the
champion file in place: keep it intact, save each attempt as a new version,
and promote a challenger only on a full-listen win. If nearby lyric versions
diverge globally run after run, stop word-surgery and render the champion at
2+ fresh seeds instead — seed variance alone may hold a clean take. Once drops
are down to a few negligible words, further lyric tweaks will be lost in
run-to-run noise: switch from editing to selecting (render N seeds, keep the
best take).

Two model behaviors to expect, not fix: short fragment lines (2–4 syllables
on their own gesture) are sometimes repeated to fill time ("power beyond,
power beyond") — harmless, and preferable to a dropped onset, so leave them.
And vocal character from the style prompt (including singer gender) is
probabilistic per run, not guaranteed — a voice-flipped take is a
reseed/discard situation, never a lyric problem. Judge lyric adherence only
across takes with the intended voice.

## Worked example (compressed)

Reference: `"Came a time remembered well"` → 7 (Came-1, a-1, time-1,
re-mem-bered-3, well-1). Beat: "crimson sail, moving fast." Draft: "A crimson
sail rides wild and free" → 8, one over. Fix: drop the article → "Crimson
sail flies wild and free" → 7. Matches. Loop per line, per section.

## ANode repo notes

- Canonical tags only (`[Verse]`/`[Chorus]`/`[Interlude]`/`[Intro]`/`[Outro]`);
  no `[Verse 1]`, `[Pre-Chorus]`, `[No Vocals]` banners — the planner only
  knows the canonical set.
- Cover recipe: chord-free ABC + `cot="melody"`; the node passes the ABC file
  straight to the sidecar, so `strip_abc_chords()`-style preprocessing is the
  transcriber's job, already done in kept `score-melody.abc` files.
- Fit warning (`score_lyrics_fit_warning`) checks section counts only — a
  `None` warning does **not** mean the words fit; run the per-line/per-note
  density check from this doc before rendering.
- `python tools/abc_fit_template.py <score.abc>` emits the map this doc
  reasons about (vocal staff only): sections with phrase starts, note
  budgets, breath/hold marks, mid-section meter changes, and numbered blank
  lines to fill in.

# ABC↔lyrics fit

When a fixed ABC melody is supplied (`cot="melody"`, chord-free score),
the lyrics must map onto the score's **musical** structure — not onto the
song's abstract verse/chorus idea. A 9-line stanza cannot ride a 1-phrase
interlude; the model will stretch/cram syllables and the vocal garbles.

## 1. Read the score first

- `%` comment lines, in order, are the section map. Keep them in the same
  order in the lyrics, including instrumental ones.
- Within a section, count **vocal phrases**: note runs separated by rests
  (`z`). Rests-only stretches are instrumental — no words there.
- Note the tempo (`Q:`) and density: brisk scores want 2–4 short lyric
  lines per phrase; sparse ballads take longer lines.
- `V:` staves, `w:` lyric lines, chord `"symbols"`, bar lines: ignore for
  mapping (with `cot="melody"` the file should already be chord-free).

## 2. Map lyric sections onto phrases

- Budget ~2–4 lyric lines per sung phrase.
- Repeats are free material: a 2-gesture chorus takes the hook twice, a
  6-gesture finale takes hooks plus ad-libs.
- One-phrase interludes take one cry (`How can we understand`), not a
  stanza. Stanzas with no vocal underneath get cut — say so explicitly
  rather than silently dropping words.
- Ad-lib/outro material with no vocal in the score moves to the last
  sung section (usually the final chorus), where notes exist.
- Empty `[Intro]`/`[Interlude]`/`[Outro]` headers render instrumental
  (duration comes from the score). If a run ever mishandles empties, drop
  those tags — a clean shorter sing beats a confused full-length one.

## 3. Sanity checks before rendering

- Every score section has a lyric tag, in score order.
- No lyric section is empty unless its score stretch is rests-only.
- Section counts roughly agree (a fit warning on divergence is a real
  garble risk, not noise — reshape words until it clears).
- Hook lines are identical wherever repeated.

## Worked example: rock cover over a transcribed 12-section score

Score map: intro(rests) – verse(3 phrases) – chorus(2) – verse(3) –
chorus(~5) – interlude(rests) – verse(3) – interlude(1 phrase +
instrumental) – chorus(~6) – interlude(rests) – chorus(~8, finale) –
outro(rests).

Result: empty `[Intro]`; 9-line verses (3 lines/phrase); hook ×2, then
×5–6 for the big choruses; empty interludes; one cry line in the sung
interlude; traveler ad-libs + variant last line folded into the finale
chorus; empty `[Outro]` for the instrumental fade. The 7-line bridge from
the original had no vocal underneath anywhere, so it was cut to its
opening cry — stated, not hidden.

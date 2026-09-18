# Lyrics format

## Rules

- Tags on their own lines: `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`,
  `[Outro]`, `[Interlude]`. Blank line between sections, blank line after
  the tag is optional but be consistent within a file.
- Reuse the same tag for repeated sections with identical words; paste the
  words out every time (no `×4`, no "repeat chorus").
- ~30 seconds of singable content per section. A 9-line verse is fine; an
  18-line verse should become two sections.
- Start with `[Verse]` or `[Chorus]`. `[Intro]` first is unstable.
- Words and tags only. No commentary, no `(shout!)`, no chord names, no
  phoneme respellings, no production notes.
- Light punctuation is fine; heavy punctuation and ALL CAPS do not help.
- Keep line breaks where breaths go: short lines sing better than long
  sentences. One idea per line.

## Minimal shape

```text
[Verse]
line one
line two
line three

[Chorus]
hook line
hook line
```

## Tag choice

- `[Chorus]`: the repeated hook, even if it is a single line.
- `[Bridge]`: contrasting middle material (different rhyme, feel, or
  perspective). Not a third verse.
- `[Interlude]`: short connective tissue or a single cry between big
  sections — 1–3 lines, not a full stanza.
- `[Intro]` / `[Outro]`: opening/closing material. May be left empty when
  the score opens or fades instrumentally (see abc-lyrics-fit.md).
- `[Verse]`: everything strophic. Do not invent `[Pre-Chorus]` /
  `[Refrain]` / `[Part 2]` — the planner knows the canonical set.

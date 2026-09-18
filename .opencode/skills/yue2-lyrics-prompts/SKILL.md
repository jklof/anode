---
name: yue2-lyrics-prompts
description: Write YuE2 style prompts and format lyrics for lyrics-to-song generation. Use when turning a musical idea into a style prompt, tagging/structuring a lyrics file, adapting words to a fixed ABC melody, or checking that lyrics and a score fit before rendering. Covers cot full/melody/off selection and the lyrics↔ABC compatibility check.
---

# YuE2 Lyrics & Prompts

Turn a musical idea into a style prompt plus a lyrics file that renders
cleanly. This skill covers **words in, words out**: the style prompt, the
lyrics file, and (for covers) fitting lyrics to a fixed ABC melody. It does
not cover running inference, picking hardware, or editing scores — for the
full YuE2 workflow (planning, synthesis, score editing, listening
comparisons) see the official `yue2-music` skill in the YuE repository.

## Route the request first

| Request | Do this |
|---|---|
| New song, no score | Write style + lyrics → `cot="full"` |
| Cover with a user-supplied or transcribed ABC melody | Inspect the score → adapt lyrics to its phrases → `cot="melody"`, chord-free ABC |
| Fast draft, no editable plan needed | Same inputs → `cot="off"` (least intelligible; never use for covers) |
| Lyrics feel wrong against a score | Run the fit check below; reshape words, not the score |

Read [references/style-prompts.md](references/style-prompts.md) when writing
the style, [references/lyrics-format.md](references/lyrics-format.md) when
writing or tagging lyrics, and
[references/abc-lyrics-fit.md](references/abc-lyrics-fit.md) whenever an ABC
score is in play.

`prompt.md` (next to this file) is a condensed single-shot version of these
rules for small local models: inject it before the user brief when driving
generation from a local LLM instead of an agent. It mirrors this file by
design; change both together.

## Style prompt rules

- One line: genre, instrumentation, vocal character, mood, tempo, language.
  Concrete musical descriptors beat prose (`warm Rhodes, round bass, brushed
  drums`, not `a nice groovy vibe`).
- Name the language when lyrics are non-English; name the tempo (`96 BPM`)
  when it matters. There is no `bpm` or `negative_prompt` field — tempo
  lives in the style text (and in the ABC for covers).
- Never write lyrics, section tags, or production commentary into the
  style. Never paste instructions meant for the planner.
- Keep it stable across re-renders of the same song; change words or score
  when iterating, not the style, so comparisons mean something.

## Lyrics rules

- Section tags on their own lines: `[Verse]`, `[Chorus]`, `[Bridge]`,
  `[Intro]`, `[Outro]`, `[Interlude]`. Blank line between sections.
- One tag per musical section; number repeated verses only if they need
  different words (`[Verse]` reused is fine).
- ~30 seconds of singable content per section; split long verses rather
  than cramming. Start with `[Verse]` or `[Chorus]`, not `[Intro]`.
- Lyrics field carries words and tags only: no commentary, no production
  notes, no chord symbols, no phoneme hints.
- Repeats are written out (`hook ×4` is not a thing — paste the lines).
  YuE2 sings what is on the page.

## ABC↔lyrics fit (covers)

Do not require identical section names/numbers as a rigid rule. Instead:

1. Read the score's **musical structure**: `%` section lines in order,
   vocal phrases per section (note runs separated by rests), rests-only
   stretches (instrumental — leave them lyric-free).
2. Map lyric sections onto those phrases: roughly 2–4 lyric lines per
   sung phrase at brisk tempos. A 9-line verse fits a 3-phrase verse;
   a 1-phrase interlude fits one cry, not seven lines.
3. Stretching/cramming syllables onto a mismatched plan garbles vocals.
   Reshape the words (repeat hooks, move ad-libs, drop orphan stanzas)
   rather than forcing the fit.
4. Keep every score section tagged in the lyrics in score order —
   including empty `[Intro]`/`[Interlude]`/`[Outro]` for instrumental
   passages — so nothing shifts out of alignment.

The full procedure with a worked example is in
[references/abc-lyrics-fit.md](references/abc-lyrics-fit.md).

## ANode repo notes

- Node inputs: `style` string param, `lyrics_file` path param,
  `abc_file` path (or wired `abc_uri`), `cot` menu, `seed` int.
- The generator warns (never rejects) when score and lyric section
  counts diverge — treat the warning as a real garble risk, not noise.
- Keeps store `request.json` (style, full lyric/score texts, seed,
  profile, runtime pin): enough to recreate the song. Record which file
  you rendered from.
- Passing `--backend`/precision tweaks or shortening songs to dodge OOM
  is out of scope for this skill; report hardware problems, don't hide
  them in the words.

# Style prompts

One line of concrete musical description. The model reads genre,
instruments, voices, mood, tempo and language from it — nothing else.

## Anatomy (in this order)

1. **Language** (when not obvious): `English`, `Mandarin`, …
2. **Genre core**: `indie pop`, `jazz funk`, `classic rock`, `piano ballad`
3. **Instruments, 2–5 named**: `bright acoustic guitar, soft drums`,
   `warm Rhodes, round bass, light drums`, `driving rhythm guitar, warm organ`
4. **Vocal character**: `warm lead vocal`, `clear lead vocal, gentle
   gang-vocal chorus`, `relaxed vocal, clean live band feel`
5. **Tempo/feel when load-bearing**: `96 BPM`, `anthemic`, `laid-back`
6. **Mix hint (optional, short)**: `polished demo mix`, `clean live feel`

## Good examples

```text
English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix
```

```text
English, jazz funk cover, warm Rhodes, round bass, light drums, relaxed vocal, clean live band feel
```

```text
English, classic rock, driving rhythm guitar, warm organ, anthemic gang vocal chorus
```

## Anti-patterns

- Prose or vibe talk: `a nice song about freedom with good energy`.
- Lyrics or tags in the style field. Structure belongs in lyrics.
- Planner instructions (`make the chorus bigger`, `key of G` as prose).
  Key/meter for covers come from the ABC, not the style.
- Changing the style between takes of the same song and comparing the
  audio as if only the words changed.
- Stuffing the style to fix a lyrics problem (garbled hook → fix the
  lyric phrasing/sections, not the adjectives).

# Vocal Broadcast Chain (`vocal_broadcast_chain.json`)

A loadable 5-node live vocal strip: cleanup → gate → transform → de-ess → limit.
Load via the app's patch-load path (`Graph.to_json` format; no clock source).

```text
Mic → Mic HPF → Mic Gate → Voice → De-Ess → Out Limiter → Out
```

## Stage settings

| # | Node | Key settings | Role |
|---|---|---|---|
| 1 | BiquadFilter (High Pass) | cutoff **80 Hz**, Q 0.707 | Removes rumble/air handling before tracking and dry |
| 2 | NoiseGate | thresh **−50 dB**, ratio 10:1, attack 1 ms, hold 50 ms | Kills bleed/pauses; sidechain free for other keying |
| 3 | WorldVoiceTransformer | pitch 0 st, formant 0 st, mix 1.0 | Neutral base — transpose as needed (±12 st) |
| 4 | DeEsser | freq **6500 Hz**, thresh **−18 dB**, depth **6 dB**, att 1 ms / rel 60 ms | Tames sibilance **after** shifting (shift-brightened ess lives here) |
| 5 | BrickwallLimiter | threshold/ceiling **−0.1 dBFS** | Safety net; transparent until peaks hit |

## Placement notes

* **De-esser defaults to post-WORLD.** Pitch/formant shifting brightens ess, so the
  harshness to tame is created inside stage 3 — treat post placement as canonical.
* **Pre-WORLD variant** (protects F0 tracking on extremely sibilant mics):
  move De-Ess between Gate and Voice. Expect slightly duller consonants into the
  analysis; A/B both orders on the actual voice.
* If the WORLD stage runs wet/dry `mix < 1.0`, keep the De-Esser after the blend
  so both paths are treated equally.

## No-code stopgap (no DeEsser node needed)

A BiquadFilter keying a Compressor sidechain approximates de-essing today:

```text
Mic ──┬──→ BiquadFilter (High Pass ~6 kHz) ──→ Compressor.sidechain
      └──→ Compressor.in (thresh −40 dB, ratio 10:1) ──→ Out
```

Measured −16.8 dB ess-HF reduction on bandpassed-noise /s/. Compromise vs the
split-band node: **broadband pumping** (vowel body ducks with loud ess) instead
of HF-only attenuation. Use the stopgap to confirm a voice needs de-essing;
reach for the node for transparency.

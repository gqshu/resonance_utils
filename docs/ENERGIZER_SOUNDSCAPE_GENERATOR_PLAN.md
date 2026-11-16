# Implementation Plan — Hybrid Energizer Soundscape Generator (ElevenLabs + Procedural)

*Implementation spec for Claude Code. Work milestone by milestone (§12). Each milestone has acceptance criteria.*

**Scope: soundscape only.** The app's separate music engine owns every lever (AM, binaural, monaural/isochronic) and their depth/level trajectories. This generator produces lever-ready, pleasant beds plus the metadata the engine consumes. It never applies a lever.

**Design in one sentence:** ElevenLabs supplies the musicality (melodic/harmonic content, opening motif, textures). Code supplies the layers the engine depends on: the steady `pad` (AM carrier) re-synthesised from the AI track's chords, the exact-grid `pulse` and `bass`, the tempo ramp, and the clear lever slots.

Context docs: Focus-First "Ramp-to-Work" research report; `BGMUSIC_TAXONOMY_CHANGES.md` (energizer pack, `am_compatible` gate); `Monaural_and_Isochronic_Beats_Implementation.md`. Existing tools: `audio_features.py`, `eeg_report.py`, `resonance_utils`.

Evidence labels: **[E]** established, **[P]** preliminary, **[H]** engineering hypothesis. Every **[H]** number is a **config constant** fixed by listening tests or EEG/behavioural runs, never hard-coded.

---

## 1. Layer ownership

| Stem | Source | Role | Constraints |
|---|---|---|---|
| `music` | Eleven Music → stem separation (melodic/"other" stems; drums and vocals discarded) | Musical interest: keys, plucks, arps, light instrumentation | No vocals; slots notched; level in the AM band capped (§7.3); time-mapped to the ramp |
| `motif` | Eleven Music, separate short call at entry tempo | Melodic awakening line, first ~30 s only ([P] McFarlane 2020) | Fades out before the plateau (no salient melody during work [E]) |
| `pad` | **Procedural**, following chords detected in `music` | **AM carrier** for the engine | Sustained, transient-free, energy in the AM band, controlled native modulation (§6.1) |
| `pulse` | **Procedural** | Kick/hats on the exact tempo grid | Onsets ≤ 16th notes; no 32nd rolls |
| `bass` | **Procedural**, following the chord roots | Low-end floor | < 180 Hz, soft attacks, no pumping |
| `texture` | ElevenLabs Sound Effects (looping), CC0, or own recordings; procedural `air` fallback | Ambience | Passes the native-modulation gate; slots notched |

The `pad` comes from code because the engine's AM needs a steady carrier without hidden beating. The `pulse` and `bass` come from code because the engine tempo-locks AM to the grid, and an AI drum stem neither follows an exact tempo ramp nor sits exactly on a grid.

---

## 2. Interface contract with the music engine

### 2.1 Input owned by the engine (read-only here)
```yaml
# lever_slots.yaml — owned by the music engine; copy lives in config/, never edited by this package
am_band:      { lo_hz: 200, hi_hz: 1000 }               # engine applies AM to the `pad` stem only
mb_slot:      { f1_hz: 1200, max_beat_hz: 40, guard_hz: 50 }
bb_slot:      { carrier_lo_hz: 200, carrier_hi_hz: 500, guard_hz: 25 }
native_mod_bands_hz: [[12, 20], [38, 42]]
```

### 2.2 Output per bed
```
beds/<bed_id>/
  manifest.yaml        # params, seeds, generator version, API request hashes, licence + AI flags, stem gains, QA tags
  stems/pad.wav pulse.wav bass.wav music.wav motif.wav texture.wav   # 48 kHz, 24-bit, equal length, sample-aligned
  tempo_map.json       # bpm(t) at 100 Hz; beat, 16th and 32nd onset times (s)
  arc.json             # phase markers {entry, ramp, plateau, handoff}; per-stem gain/brightness curves (already baked in)
  chords.json          # bar-quantised chord timeline (for the engine/UI if useful; also an audit trail)
  preview_mix.wav      # reference mix at manifest stem gains
  qa_report.json
```

### 2.3 Guarantees (checked by the QA gate, §9)
1. `pad` is sustained with its energy in `am_band`, and its native modulation at `native_mod_bands_hz` is below threshold.
2. No bed energy in `mb_slot` ± guard (notch ≥ 18 dB on every stem).
3. No tonal peaks from non-pad stems in the BB carrier range above threshold.
4. Bed and mix native modulation at 12–20 Hz and 38–42 Hz is below threshold.
5. `pulse`/`bass` onsets lie on the `tempo_map.json` grid (≤ 1 ms).
6. The `music` stem follows the same tempo map (beat alignment ≤ 15 ms after time-mapping).

---

## 3. Arc (default profile `wake_to_work_4m`; 2:30 and 5:00 scale proportionally)

| Phase | Time | Tempo | Stems active | Brightness | Loudness |
|---|---|---|---|---|---|
| Entry | 0:00–0:30 | entry tempo = `ENTRY_TEMPO_RATIO` × plateau | pad, texture, motif | dark | −8 → −5 dB |
| Ramp | 0:30–2:30 | linear in BPM → plateau | + music (from 0:30, sparse section), + pulse (0:45), + bass (1:00) | opens | → 0 dB |
| Plateau | 2:30–3:40 | plateau | full; motif gone | held | 0 dB |
| Hand-off | 3:40–4:00 | plateau | music and pulse thin out | slight close | −3 dB |

- `plateau_bpm` defaults to 118 (range 105–125 for focus priority).
- `ENTRY_TEMPO_RATIO` defaults to **0.85** in hybrid mode (≈ 100 BPM). It was 0.76 in the all-procedural design, but the AI `music` stem must be time-stretched across the ramp and quality degrades beyond `MAX_STRETCH = 1.18` [H]. Arousal comes from tempo **and** density, brightness and loudness, so the smaller tempo range is compensated by a stronger density/brightness build.
- The `pad` level stays near-constant from 0:30 (steady AM carrier). Brightness may change; the pad must not swell.

Tempo math (`tempo.py`): tempo is defined on the beat axis, and `t(b) = ∫₀ᵇ 60/bpm(β) dβ` is integrated at 1/960-beat resolution. All procedural events are scheduled in beats and converted to time with this map.

---

## 4. Parameters

```python
@dataclass
class EnergizerParams:
    # identity
    seed: int = 0
    arc_profile: str = "wake_to_work_4m"     # wake_to_work_2m30 | _4m | _5m
    plateau_bpm: float = 118
    entry_tempo_ratio: float = 0.85
    key_root: str = "D"
    mode: str = "major"                      # major | lydian | mixolydian | dorian
    # AI musical layer
    music_source: str = "elevenlabs"         # elevenlabs | none (all-procedural fallback/control)
    music_style: str = "ambient_electronic_keys"   # preset id → prompt template (§5.2)
    motif: bool = True
    # procedural layers
    pad_timbre: str = "warm_additive"        # warm_additive | glass | string_ensemble
    pad_tuning: str = "just_per_chord"       # just_per_chord | equal
    pulse_style: str = "soft_four"           # none | soft_four | broken_soft | shaker_only
    # texture
    texture_source: str = "elevenlabs_sfx"   # elevenlabs_sfx | cc0_file | procedural_air | none
    texture_prompt: str = "soft steady rain on leaves, no thunder, no birds"
    length_s: float = 240
```

`bed_id = sha1(params, generator_version, lever_slots.yaml, API model ids)[:12]`.

---

## 5. ElevenLabs integration (`el_client.py`, `el_music.py`, `el_sfx.py`)

> **Verify every endpoint, field name and limit against the current ElevenLabs API reference** (`https://elevenlabs.io/docs/api-reference/music/compose`, `.../music/separate-stems`, sound-effects reference) before coding. Use the official `elevenlabs` Python SDK where it covers the call, and raw HTTP otherwise. Facts confirmed at planning time:
> - Music compose accepts a prompt **or** a `composition_plan` whose sections each have name, positive/negative styles, duration and lyrics. Total length is 3 s – 10 min; each section is 3 s – 2 min.
> - Stem separation is `POST /v1/music/stem-separation` with `two_stems_v1` or `six_stems_v1`; it returns a zip and can be slow. `pcm_44100` output requires the Pro tier.
> - Sound effects: max 30 s per generation; `loop: true` for seamless loops; WAV 48 kHz only for non-looping (looping returns MP3).
> - Licence: broad commercial use on paid plans; film/TV/large studio games need Enterprise. **Confirm in writing that in-app streaming of modified tracks is covered** before shipping.

### 5.1 Client rules
- API key from env `ELEVENLABS_API_KEY`; never logged.
- **Content-addressed cache:** `cache/el/<sha1(request_json)>/` stores the response audio and metadata. Identical requests never hit the API twice (cost control and reproducibility). `--no-cache` forces a call.
- Retry with exponential backoff on 429/5xx (max 5). A cost/usage log is appended per call (`cache/el/usage.jsonl`: endpoint, seconds generated, timestamp, request hash).
- Request the highest-quality uncompressed format the plan allows (PCM 44.1 kHz). Resample to 48 kHz with `soxr` (VHQ).
- Pass a seed where the API supports one, so the same plan + seed gives identical audio. Store it in the manifest.

### 5.2 Music generation: two calls per bed
**Call A: main track** at **plateau tempo**, length = ramp + plateau + hand-off + 8 s pre-roll (≈ 3:38 for the 4:00 profile). Composition plan built by `el_music.build_plan(params)`:

| Section | Duration | Positive styles (template) | Negative styles (all sections) |
|---|---|---|---|
| `build_sparse` | 0:30–1:30 share | "instrumental, {style}, {bpm} BPM, {key} {mode}, sparse soft keys, gentle, warm, steady" | "vocals, voice, choir, spoken word, lyrics, drums, percussion, drops, risers, build-ups, sidechain, pumping, tremolo, stutter, glitch, sudden changes, minor key" |
| `build_fuller` | 1:30–2:30 share | "+ light plucked arpeggio, brighter, more motion, steady energy" | same |
| `plateau` | 2:30–3:40 share | "steady, hypnotic, consistent, no lead melody, background music for focus" | same |
| `handoff` | 3:40–4:00 share | "gently thinning, softer" | same |

- Lyrics are empty for every section.
- "drums, percussion" are negative because the AI drums are replaced by procedural `pulse`. Separation then has less to remove, which reduces leakage.
- Style presets (`music_style`) live in `config/music_styles.yaml`: `ambient_electronic_keys`, `warm_lofi_keys_no_beat`, `minimal_marimba_pulse`, `soft_post_rock_guitars`, `bright_piano_minimalism`, plus more as they prove themselves in listening tests.

**Call B: motif**, ~36 s at **entry tempo**, same key and mode: "instrumental, single gentle bright melodic line, {timbre e.g. celesta / soft bells / felt piano}, simple rising phrase, {entry_bpm} BPM, {key} {mode}, solo, no accompaniment". Negative styles as above plus "chords, pads, bass".

### 5.3 Stem separation
- Run `six_stems_v1` on Call A. Keep the melodic/"other"-type stems (piano/guitar/other, whichever the variant returns). Discard vocals and drums; keep the AI bass only as an analysis reference, never in the output.
- The kept stems are summed into the `music` stem. Map stem names from the actual zip contents (`stem_map` in config). Log unknown names.
- Call B is used unseparated (it's a solo line). If QA finds accompaniment leakage, separate it too.

### 5.4 Textures (`el_sfx.py`)
- Generate 30 s with `loop: true` from `texture_prompt`, then tile with a 2 s equal-power crossfade to bed length, and randomise the tile order between two or three seeded variants to avoid audible repetition.
- Fallbacks: `cc0_file` (path in params; licence stored in the manifest) or `procedural_air`.

---

## 6. Procedural layers (`synths/`)

All NumPy/SciPy, deterministic, 48 kHz, float64 internally. Every partial is explicit, so slot avoidance is exact.

### 6.1 `pad`: the AM carrier
- **Harmony input:** `chords.json` from §7.2. The pad follows the AI track's harmony, so it sounds like part of the music rather than a drone underneath it.
- **Voicing:** 3–5 chord tones with most energy in `am_band`, and minimal-movement voice leading.
- **Additive partials:** harmonics with a timbre-specific tilt. Drop partials in `mb_slot` ± guard and above 2 × `am_band.hi_hz`.
- **Beating control:**
  - Detune spread specified in Hz, with every pairwise difference ≤ `PAD_MAX_BEAT_HZ` (6 Hz [H]).
  - `just_per_chord` tuning removes equal-temperament interval beating on sustained chords. Common tones drift ≤ 5 cents at chord changes, which happen only during crossfades.
  - Synthesis-time audit: no pairwise partial difference above −40 dB inside `native_mod_bands_hz`. Re-voice or drop partials on failure.
  - Note: just-tuned pad tones sit a few cents away from the AI track's equal-tempered notes. The pad is quiet and sustained, so this should be inaudible; check it in M2 listening and fall back to `equal` if needed.
- **Movement:** tilt LFO ≤ 0.2 Hz; random-walk partial amplitudes with a 2–5 s time constant; chord changes by equal-power crossfade ≥ 400 ms.
- **Low-confidence harmony fallback:** if chord detection confidence < `CHORD_CONF_MIN`, the pad plays a key root + fifth (+ ninth) sustained voicing, which fits any diatonic harmony.

### 6.2 `bass`
Sine + 2nd/3rd harmonic, low-passed at 180 Hz. Plays the chord root per `chords.json`, holding through the ramp and adding an optional 8th-note pattern in the plateau. Attacks ≥ 15 ms.

### 6.3 `pulse`
- Kick: sine sweep 120 → 50 Hz with a high-passed click (> 2 kHz).
- Hats/shakers: band-passed noise above 4 kHz on 8ths or 16ths.
- **No 32nd notes.** At 118 BPM, 32nds = 15.7 Hz, which is the engine's AM rate.
- Velocity variation ±2 dB, timing jitter 0. Styles: `soft_four`, `broken_soft`, `shaker_only`, `none`.

### 6.4 `procedural_air` texture
High-shelved pink noise with ≤ 0.1 Hz amplitude drift.

### 6.5 FX
Deterministic FDN reverb (no modulated tails) on `pad`. Tilt EQ for the brightness arc.

---

## 7. Analysis and conforming of AI audio (`analyze.py`, `conform.py`)

### 7.1 Tempo and grid of the AI track
- Beat-track the summed `music` stem with `librosa.beat.beat_track` and refine with onset-strength autocorrelation. Expected tempo = `plateau_bpm`. Tolerate ±3 %; outside that, discard and regenerate with a new seed.
- Fit a constant-tempo grid (least squares on beat times) and record the downbeat phase.

### 7.2 Key and chords
- Key: chroma (CQT) + Krumhansl–Schmuckler profiles. It must match the requested key/mode (or its relative). Otherwise regenerate.
- Chords: beat-synchronous chroma → template matching against the diatonic triads/sevenths of the detected key → HMM/Viterbi smoothing → quantised to bars, with a minimum duration of 1 bar.
- Output `chords.json` with per-bar label, root, pitch classes and confidence.
- Don't use madmom's pretrained chord models (their model licence is non-commercial). This template approach is self-contained.

### 7.3 Conforming `music` and `motif`
1. **Time-map for the ramp:** build a time-map from the AI track's constant-tempo grid to the bed's tempo curve (ramp → plateau) and apply it with Rubber Band (`pyrubberband`, `--timemap`, crisp/finer engine).
   - Rubber Band is GPL. It's fine in this **internal offline** tool (the output audio isn't covered), but never ship it in the app.
   - Enforce `MAX_STRETCH`.
2. **Motif:** already at entry tempo. Trim to phrase boundaries (onset/energy-based), fade out by 0:30, align its first downbeat to bar 1 of the bed grid.
3. **Band handling for `music`:**
   - FIR notch at `mb_slot` (≥ 18 dB).
   - In `am_band`, apply a gain offset of `AI_IN_AM_BAND_GAIN_DB` (default −6 dB [H]). AI content inside the AM band competes with the AM carrier; how much is tolerable is the project's open *richness ceiling* constant, now measurable per bed.
   - Attenuate tonal peaks found in the BB carrier range (narrow dynamic notches).
4. **Resample** everything to 48 kHz, apply the arc gain/brightness curves, and loudness-normalise per stem to the LUFS targets in the manifest.
5. **Beat alignment check:** re-detect beats on the conformed `music` and compare them with `tempo_map.json`. Tolerance is 15 ms (median) and 30 ms (p95); on failure, re-stretch with refined anchors or regenerate.

---

## 8. Render (`render.py`)

Order: `plan` → API calls (cached) → separate → analyse → `tempo_map` → synthesise `pad`/`bass`/`pulse` from `chords.json` on the map → conform `music`/`motif`/`texture` → apply the arc → per-stem normalise → `preview_mix` → sidecars → QA.

- **Determinism:** with a warm cache, re-rendering a `bed_id` must be bit-identical.
- **Performance target:** < 3 min per 4-minute bed on the M4 Pro once the cache is warm (excluding API latency).

---

## 9. QA gate (`qa.py`, built on `audio_features.py`)

All thresholds are config constants [H].

| Metric | On | Gate |
|---|---|---|
| `native_mod_beta` (envelope modulation depth 12–20 Hz) | `pad` in `am_band`; `music` in `am_band`; preview mix | `< NATIVE_MOD_BETA_MAX` (separate limits for pad and music) |
| `native_mod_40` (38–42 Hz) | every stem; preview mix | `< NATIVE_MOD_40_MAX` |
| `pad_onset_rate` | `pad` | ≈ 0 (chord changes only) |
| `mb_slot_residual_db` | preview mix | `< MB_SLOT_RESIDUAL_MAX_DB` |
| `bb_tonal_peaks` | non-pad stems | none above threshold in the BB range |
| `music_am_band_share` | `music` energy in `am_band` relative to `pad` | `< AI_AM_BAND_SHARE_MAX` |
| `grid_conformance` | `pulse`/`bass` onsets; `music` beats | ≤ 1 ms / ≤ 15 ms median |
| `vocals_detected` | `music`, `motif` (vocal-stem energy from a 2-stem separation or a VAD) | false |
| `melodic_salience` (plateau window only) | preview mix | `< SALIENCE_MAX` |
| `stretch_artifacts` | `music` (transient smearing / spectral flux anomaly vs. un-stretched source) | `< STRETCH_ARTIFACT_MAX` |
| `loudness_arc`, true peak | preview mix | within ±1 dB of `arc.json`; ≤ −1 dBTP |

Tags: `am_compatible`, `mb_compatible`, `bb_compatible`, and `lever_compatible` = all three. **`lever_compatible` gates energizer-pack membership** (extends change #4 in `BGMUSIC_TAXONOMY_CHANGES.md`).

---

## 10. Package layout (`resonance_utils/energizer/`)

```
energizer/
  __init__.py
  config.py           # dataclasses, thresholds, loads lever_slots.yaml + music_styles.yaml
  cli.py              # energizer render | batch | compare | qa
  tempo.py            # bpm curve on beat axis, beat↔time, grid export
  arc.py              # arc profiles → phase times, per-stem gain/brightness curves
  theory.py           # modes, diatonic chords, voicing, voice leading, just-intonation tables
  el_client.py        # auth, cache, retry, usage log
  el_music.py         # composition-plan builder, compose (main + motif), stem separation, stem mapping
  el_sfx.py           # looping texture generation + tiling
  analyze.py          # beat/tempo, key, chord recognition → chords.json
  conform.py          # time-map stretch, band handling, notches, alignment check
  synths/
    additive.py       # partial-bank renderer (phase-continuous, vectorised)
    pad.py bass.py pulse.py air.py
  slots.py            # slot partial filtering, pairwise-beat audit, FIR notch design
  fx.py               # FDN reverb, tilt EQ, filters
  render.py           # full pipeline → bed folder
  qa.py               # gate + tags
  batch.py            # seeded sweeps, skip existing bed_ids
config/
  lever_slots.yaml  music_styles.yaml  thresholds.yaml  arc_profiles.yaml  stem_map.yaml
tests/
  fixtures/           # short synthetic + recorded audio, mocked EL responses (zip with stems)
  test_tempo.py test_slots.py test_pad_beating.py test_analyze.py
  test_conform.py test_el_client_cache.py test_render_determinism.py test_qa.py
```

Dependencies: `numpy`, `scipy`, `soundfile`, `soxr`, `pyloudnorm`, `librosa`, `pyrubberband` (+ rubberband CLI), `PyYAML`, `elevenlabs`, `requests`, `pytest`. No madmom models.

**Testing rule:** no test calls the real API. `el_client` is mocked with fixtures. One opt-in smoke test is marked `@pytest.mark.live` and skipped unless `ELEVENLABS_API_KEY` and `--live` are set.

---

## 11. Library production
- `batch.py` runs seeded sweeps over `music_style` × `plateau_bpm` {108, 114, 118, 122} × `mode` × `pulse_style` × `texture` × `arc_profile`.
- QA-pass beds go to a human listening pass. Keep only approved beds and log the decision in the manifest (curation record; also strengthens the authorship position).
- **China:** beds with `music_source = elevenlabs` or an AI texture carry `ai_generated: true`. The app must show the explicit AI-content label and embed the implicit metadata label (GB 45438-2025).
- If the engine changes `lever_slots.yaml`, re-run `batch.py`. With a warm cache this costs no API calls.

---

## 12. Milestones and acceptance criteria

**M1: Procedural core**
- Build `tempo.py`, `arc.py`, `theory.py`, `synths/additive.py`, `pad.py`, `bass.py`, `pulse.py`, `air.py`, `slots.py`, `fx.py`, and a minimal `render.py` (all-procedural, `music_source = none`).
- ✅ When:
  - A 4-minute bed renders with all sidecars.
  - Slot and pairwise-beat tests pass, and pulse onsets are within 1 ms of the grid.
  - Two renders are bit-identical.

**M2: ElevenLabs layer + A/B prototype (the pleasantness decision)**
- Build `el_client.py` (cache/retry/usage), `el_music.py`, `el_sfx.py`, `analyze.py`, `conform.py`.
- `energizer compare --seed N` renders the same params three ways:
  - (a) all-procedural;
  - (b) hybrid (this plan);
  - (c) reference: the un-separated Eleven Music track with only the arc gain applied (not lever-ready, for listening only).
- ✅ When:
  - All three render for 3 different `music_style` presets.
  - Hybrid passes beat alignment (≤ 15 ms median) and key match.
  - George does a blind listening comparison and either approves the hybrid's quality or flags specific problems (stretch artifacts, pad/music clash, separation leakage) to fix before M3.

**M3: QA gate**
- Build `qa.py` on top of `audio_features.py`, plus a synthetic failure suite (32nd-note hats, equal-temperament thirds, detuned saw pad, vocal fixture, tremolo fixture). The suite must be rejected with the correct reasons.
- ✅ When the suite fails for the right reasons and the M2 hybrid beds get tagged.

**M4: Batch + library v0**
- Build `batch.py` and the style presets; render 30 beds.
- ✅ When the QA pass rate is recorded per `music_style`, prompt templates are tuned until ≥ 60 % pass, and the usage log reports cost per bed.

**M5: Engine hand-off**
- The engine consumes 5 beds (stems + `tempo_map.json` + `arc.json`) with all lever arms.
- ✅ When the engine's verification confirms the delivered AM depth on `pad` and the 40 Hz monaural slot is clean on every bed.

**M6: Empirical constants (with `eeg_report.py` + KSS/PVT)**

| Run | Factor | Decides |
|---|---|---|
| S1 | `AI_IN_AM_BAND_GAIN_DB` {0, −6, −12} / `AI_AM_BAND_SHARE_MAX` | Richness ceiling for the product-AM path |
| S2 | `PAD_MAX_BEAT_HZ` {3, 6, 10} × tuning | Pad richness the AM carrier tolerates |
| S3 | `plateau_bpm` {108, 118, 125} × `pulse_style` | Focus-vs-energize sweet spot (PVT lapses vs. self-rated tension) |
| S4 | Arc vs. flat at the same plateau | Whether the ramp adds value ([H], untested) |
| S5 | Beds spanning `native_mod_*` | QA thresholds |

- ✅ When the constants are frozen in `thresholds.yaml` → **energizer pack v1**.

---

## 13. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Time-stretch artifacts on the `music` stem | `ENTRY_TEMPO_RATIO` 0.85 and `MAX_STRETCH` 1.18; crisp Rubber Band engine; `stretch_artifacts` QA metric; music enters at 0:30 when stretch is smallest during the sparse section |
| Separation leakage (drum ghosts, phasiness) | Negative-prompt drums out of the AI track so there's less to remove; QA native-mod and transient checks on `music` |
| Pad and AI harmony clash | Chords come from the AI track itself; confidence fallback to a root–fifth voicing; M2 listening check of just vs. equal tuning |
| AI ignores the tempo or key prompt | Detect, tolerate ±3 %, otherwise regenerate with a new seed (cached attempts counted in the usage log) |
| API cost and rate limits | Content-addressed cache, batch throttling, usage log, cost-per-bed KPI in M4 |
| Licence scope (in-app streaming of modified tracks) | Written confirmation from ElevenLabs or an Enterprise plan before shipping; keep `music_source = none` beds as a fully owned fallback catalogue |
| AI-label duties in China | Manifest `ai_generated` flag drives the app's labels |
| Brain.fm patent (AM on a music sub-band) | Engine-side concern, flagged here because `am_band` is defined by the engine: FTO review before the US launch |

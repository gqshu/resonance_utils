# resonance_utils

Soundscape generators for the resonance EEG-adaptive app. One command, `ss`, drives every
**soundscape type**. Types share audio I/O, analysis, loop assembly, the carrier set, the
ElevenLabs client and the deploy framework (`core/`), but each has its own pipeline for
generating, QA and deployment.

| Type | What | Status |
|---|---|---|
| `inducer` | power-nap inducer: seamless texture loops with a baked 0.1–0.25 Hz swell, optional tone layer | in production |
| `energizer` | wake-to-work ramp: hybrid ElevenLabs + procedural beds with lever-ready stems ([plan](docs/ENERGIZER_SOUNDSCAPE_GENERATOR_PLAN.md)) | skeleton |

```bash
ss list                       # the types
ss list inducer               # what a type can generate
ss gen <type> ... -h          # build into build/<type>/<id>/
ss deploy <type>|all          # ready builds -> deploy/<type>/manifest.json + deploy/index.json
ss migrate [--dry-run]        # move a pre-0.3 build/ and deploy/ into the per-type layout
```

**Carriers.** The inducer is built against the shared carrier set in `core/slots.py` (default 110,
220 and 440 Hz, `--carrier` to change it). The energizer uses one fixed monaural carrier, 220 Hz,
with beats at 20 and 40 Hz (`energizer/config/lever_slots.yaml`). Only the three tones, 220, 240 and
260 Hz, are kept clear (±20 Hz, so 200–280 Hz), not the whole beat range. That leaves the bass
(below 180 Hz) and most of the AM band (280–1000 Hz) free. No energizer partial may also beat
against a lever tone at 12–20 or 38–42 Hz, so 180 Hz and 300 Hz are out too
(`energizer.config.partial_clash`). 220 Hz is in both sets, so the app can use it with every pack.
AM needs no slot.

The rest of this README covers the **inducer**. The binaural beat / AM is **not added here**: the
app's music driver adds it on its own bus.

```
 scenario prompt ──► ElevenLabs takes ──► screen ──► join ──► condition ──► <scenario>_flat.wav
   (cached in raw/)                        │
                                           ▼
                         swell @ 0.1 / 0.2 / 0.25 Hz  (+ optional tone layer that breathes with it)
                                           │
                                           ▼
                     <scenario>_0p1Hz[_tone].wav   _0p2Hz   _0p25Hz     (+ .json, _envelope.csv)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # numpy + scipy: everything the inducer needs
pip install -e '.[energizer]'    # + librosa, rubberband, soxr, ... for the energizer
pip install -e '.[dev]'          # + pytest; run `pytest` (no test calls the real API)
```

Requires Python ≥ 3.9. `ffmpeg` is needed only if the API returns MP3 (its fallback when PCM output
isn't available on your plan).

## Usage

```bash
export ELEVENLABS_API_KEY=sk_...
ss gen inducer --list                           # scenarios and their prompts
ss gen inducer train_smooth rain_soft --takes 8 --tone  # two scenarios, with the tone layer
ss gen inducer all --dry-run                    # show the API calls, spend nothing
ss gen inducer river --tone --stems             # also write bed and tone stems
ss gen inducer --from-dir my_takes/ --name cabin --profile train_smooth   # your own takes, no API
ss gen inducer all --probe 2 --dry-run          # cost of testing every prompt variant
ss gen inducer rain_soft --probe 3              # 3 takes per variant -> pass-rate table
ss gen inducer rain_soft --variant c --tone     # build from the variant that won
```

`ss_gen` still works as an alias for `ss gen inducer`, with the same options.

The default model is `eleven_text_to_sound_v3` (v2 is deprecated and will be removed on 2027-03-01).
Each 30 s take costs about 1,200 credits.

**Takes are cached.** They are saved in `<out>/inducer/<scenario>/raw/` and reused on every later run, so
re-running to tune the swell or tone costs nothing. Raising `--takes` buys only the missing ones.
A take is reused only if its sidecar records the same prompt and model, so editing a prompt or
switching models never silently reuses old takes. Delete `raw/` to start a scenario over.

Outputs per scenario, in `<out>/inducer/<scenario>/` (default `build/`):

| File | What |
|---|---|
| `<scenario>_flat.wav` | conditioned flat bed: no swell, no tone |
| `<scenario>_<rate>Hz[_tone].wav` | one loop per rate, 48 kHz / 24-bit, −26 dBFS RMS |
| `…json` | swell, tone and profile parameters plus QA (EEG-band excess, seam, levels) |
| `…_envelope.csv` | swell phase, envelope, gain and pan at 20 Hz, for EEG alignment |
| `…_bed.wav`, `…_tonelayer.wav` | with `--stems`: the two layers at the same scale as the mix |
| `report.json` | screening results for every take, and the take order in the loop |

## Deploy to the app

```bash
ss deploy inducer                           # every ready scenario: build/inducer/ -> deploy/inducer/
ss deploy inducer car_night rain_soft --stems   # only these, with bed + tone stems
ss deploy inducer --clean --out ../app/assets/soundscapes
```

`ss_deploy` is an alias for `ss deploy inducer`. It copies every **ready** scenario into one directory, with a single `manifest.json`
for the resonance app. A scenario is ready when its last build succeeded, every listed output and
its JSON and envelope exist, every output's QA says the EEG band is clean, and all outputs agree
on loop length and carriers. Anything else is skipped with the reason, and `build/` is never
modified. Re-deploying a subset keeps the other scenarios already in the manifest. `--clean`
empties `<out>/inducer/` first, but only if that directory already holds a manifest. Every deploy
also rewrites `<out>/index.json`, which lists the packs (types) present and their manifests.

`inducer/manifest.json` (schema 1, unchanged by the move to per-type folders: paths are relative
to `inducer/`) contains:

- the shared audio format (WAV, 48 kHz, 24-bit, stereo, −26 dBFS RMS, seamless) and the
  `carriers_hz`, every binaural carrier the soundscapes are safe with (the app picks one);
- per scenario: `id`, `title`, `loop_s`, the `flat` bed, the tone and profile parameters and the
  source (model, prompts, takes);
- per rate in `loops`: the audio file, its envelope CSV, the swell parameters, QA, and
  `troughs_s` / `crests_s`, the times of each wave's trough and crest. Switch rates or scenarios
  at a trough. Time 0 is a trough at every rate;
- with `--stems`, the bed and tone-layer stems, so the app can set the tone level itself.

Every file entry has a relative `path`, `bytes` and `sha256`. The full set of 7 scenarios is about
0.8 GB, or about 2 GB with stems.

## Inducer: the three stages

### 1. Bed (`bed.py`)

The generator supplies **texture only**: every prompt ends with a "constant level, no swelling" line.
Each take is screened, and every check is measured as **excess over a phase-randomised surrogate**
(a stationary copy with the same spectrum), so ordinary noise fluctuation scores about 0. A take is
rejected for:

- a **regular** rhythm in the 1–12 Hz EEG band (one sharp line, e.g. a train's clack): strict;
- **random** 1–12 Hz flicker above 0.8 dB. This was 0.3 dB, an untuned guess that rejected
  smooth rain. Random patter does not drive a steady-state EEG response;
- excess transients, a tonal line within ±20 Hz of any carrier, near-silence or clipping.

Each comparison averages 3 surrogates, so borderline takes don't flip between pass and reject
on luck. Thresholds live in `bed.THRESH`.

Screening results are stored in each take's sidecar JSON, keyed by the thresholds and the
measurement version, so a rebuild re-screens only new takes or takes judged under other settings.
The result does not depend on the carrier: hum is measured separately and cached per carrier, so
adding a carrier costs one spectrum per take, not a re-screen.

Passing takes are shuffled and joined with equal-power crossfades into a loop whose length is a
whole multiple of 20 s (the shortest length holding whole waves at 0.1, 0.2 and 0.25 Hz). Then
the bed's slow level is flattened in 6 frequency bands (0.5 s RMS each, so a built-in swell loses
its brightness swing too), transients are capped at +6 dB, a −6 dB ⅓-octave notch is cut at each
carrier, and the level is normalised. All of it is circular, so the loop stays seamless.

A take whose only problem is its own swell (in loudness or brightness) is not rejected outright.
It is conditioned and screened again, and kept if both swells fall below 0.75 dB.

Takes can pass screening and still sound like different places. A bed is joined only from the
largest set of passing takes whose octave-band balance matches within 10 dB in every **audible**
octave (one holding more than −30 dB of the take's power). The others are reported as tonal
outliers: one ocean take was pure low rumble, 50 dB short on treble. A static EQ, at most 6 dB
per octave, then pulls each kept take toward the group's median balance and level, so the
crossfades don't change timbre. The first rule (5 dB in all octaves, no EQ) rejected 13 of the 18
passing train, car and wind takes, mostly over treble 45 dB down. Tonal settings are not part of
the screening cache key, so tuning them never re-screens.
`--variant a,b,c` draws on several prompt variants at once, and this matching decides which takes join.

### Prompting for pass rate

Each scenario has a few prompt variants (`ss_gen --list`). Run `--probe N` to measure them, then
build from the winner with `--variant`. Probe takes stay cached, so they count towards the build.
What the screening has shown so far:

- **Close, discrete sources fail.** Rain "on a tent roof" failed 6/6 on transients (+50–120/min).
  Distance and an enclosure ("heard through a closed window") brought transients to about 0.
- **Periodic sources keep their rhythm.** Surf comes back with its own ~0.15 Hz swell. The
  conditioning removes the loudness part, but only part of the brightness part.
- **Grainy textures flicker a little.** Smooth v3 rain takes measure 0.2–0.8 dB of random
  1–12 Hz flicker, with no periodic line. That is why the flicker limit is 0.8 dB.
- **Name only what you want to hear.** Text-to-audio models tend to add what a prompt mentions,
  even when it is negated ("no seagulls"). Variants `b`/`c` describe the setting and atmosphere
  positively, which is what the ElevenLabs v3 guidance recommends.

### 2. Swell (`swell.py`)

| Rate | Depth | Brightness dip | Sway | Rise share |
|---|---|---|---|---|
| 0.10 Hz (6 breaths/min) | 8 dB | 6 dB | 0 | 40 % |
| 0.20 Hz | 6 dB | 5 dB | ±0.15 | 38 % |
| 0.25 Hz (rocking rate) | 5 dB | 4 dB | ±0.25 | 35 % |

These are scaled per scenario by its profile (`scenarios.py`: `depth`, `tilt`, `pan`, `xover`).
The swell is asymmetric (the rise is shorter than the fall, like an exhale). Treble above `xover`
dips in each trough. The period drifts by about 3 % and each wave's depth by about 10 %, but the
wave count over the loop stays exact. All rates use the **same bed**, so the app can crossfade
between rates at a trough.

### 3. Tone layer (`tone.py`, `--tone`)

A warm, open drone of stacked just fifths and octaves (default root 88 Hz, chord `1,3/2,2,3,4`),
with soft 1/h² harmonics and three copies per partial detuned by 0.06 Hz, spread across the stereo
field for slow shimmer. It swells with the bed (`--tone-swell`, default 0.6× the bed's depth), and
its upper harmonics fade in toward each crest. The default level is −12 dB under the bed (`--tone-db`).

Safety rules, enforced automatically (any dropped partial is listed in the JSON):

- no partial within ±20 Hz of any binaural carrier the app may use;
- no two partials 0.5–20 Hz apart: that spacing beats as amplitude modulation in the EEG band,
  sleep spindles included. Coinciding partials are merged;
- every frequency is snapped to a whole number of cycles per loop, so the layer loops sample-exactly.

Changing `--tone-root` or `--tone-chord` is safe: offending partials are dropped and reported.
The default root changed from 55 Hz (A) to 88 Hz for the default carriers 110, 220 and 440 Hz.
Those carriers are octaves of A, so an A root puts partials right on them. With the default chord,
only roots of about 43.5–44 Hz or 87–90 Hz keep every partial at least 20 Hz from all three. 88 Hz
has the widest margin (22 Hz) and stays audible on small speakers.

## Design rules

- **No added modulation in 1–12 Hz.** A 1–12 Hz envelope evokes steady-state responses in exactly the
  theta/alpha bins the controller reads. Swells at 0.1–0.25 Hz sit far below the 2 Hz band-power floor.
- **Keep every possible carrier clear.** `--carrier` lists every binaural carrier the app may use
  (default `110 220 440`), so it can choose one per session from a single set of files. The bed is
  notched −6 dB in a ⅓-octave around each. That removes 0.4–1.7 dB of the bed's power, against
  0.1–0.8 dB for one notch, and the level is normalised back. Takes with tonal lines near any carrier
  are rejected, and the tone layer avoids all of them. A carrier outside the list may beat against
  the tone layer: rebuild with it added (no credits, just CPU).
- **No crashes.** The swell's attack is at least 1 s, its depth is bounded and transients are capped.
  Sudden events can cause arousals.
- **Loop-safe everything.** Circular processing, whole-wave loop lengths, equal-power crossfades.
  Ship WAV/FLAC, not MP3, because encoder padding breaks gapless loops.
- **Log the stimulus.** Every output has its exact envelope in the CSV, for EEG alignment.

## Evidence and claims

| Claim | Status |
|---|---|
| Slow rocking at **0.25 Hz** speeds sleep onset and lengthens N2 in naps | Shown with a *physically rocking bed* (Bayer et al., Curr Biol 2011). The audio-only analogue (swell + sway) is a **hypothesis**. |
| 0.1 Hz swell paces breathing toward ~6 breaths/min | Plausible from the resonance-breathing / HRV literature; not sleep-validated. |
| Broadband nature beds help sleep | Evidence is low quality and mostly from masking (Riedy et al. 2021). All-night loud broadband noise reduced REM (Basner et al. 2026). In a ~30 min nap, REM barely occurs. |

Treat rate, sway and tone on/off as **randomised bandit arms**, and judge them against sleep-onset
latency, N3 creep and post-nap alertness. Make no entrainment claims from these tools.

## Licensing of sound sources

- **ElevenLabs:** commercial use requires generating on a **paid plan**.
- **Freesound:** licences are per file (CC0 / CC-BY / CC-BY-NC). Never use NC files.
- **Stock libraries:** many forbid standalone distribution, and some explicitly ban sleep or noise
  apps. Get written confirmation before using them with `--from-dir`.
- **Own recordings:** cleanest.

## Repository layout

```
src/resonance_utils/
├── cli.py            # `ss`: list | gen | deploy | migrate, dispatching to the type registry
├── core/             # shared by every type
│   ├── types.py      # SoundscapeType contract + lazy registry
│   ├── audio_io.py   # WAV I/O, decoding API audio, normalisation
│   ├── analysis.py   # block envelopes, band stats, surrogates, octave balance
│   ├── loop.py       # equal-power crossfades, exact-length seamless loops
│   ├── slots.py      # shared carriers + guards, carrier notch
│   ├── elevenlabs.py # API client: auth, retry, PCM fallback signal, usage log (cache/el/usage.jsonl)
│   ├── cache.py      # content-addressed request cache (energizer)
│   ├── layout.py     # build/<type>/, deploy/<type>/
│   └── deploy.py     # hashed file entries, guarded --clean, manifest merge, deploy/index.json
├── inducer/          # gen.py (ss_gen), deploy.py (ss_deploy), bed.py, swell.py, tone.py, scenarios.py
├── energizer/        # config.py (params, bed_id), type.py, config/*.yaml; pipeline per the plan
└── legacy/           # earlier tools, kept for reference: real-time EEG SwellEngine
                      # (swell_engine.py), ocean_gen.py, elevenlabs_bed_builder.py
tests/                # golden inducer build (bit-identical outputs), CLI, core client/cache
```

Adding a type: create `resonance_utils/<type>/type.py` with a `SoundscapeType` subclass and add it
to `core/types.REGISTRY`. Its builds go to `build/<type>/<id>/` and its deploy writes
`deploy/<type>/manifest.json`.

## Status

v0.2.0. Run against the live API on `rain_soft` and `ocean_distant` (model v2). The API returns
**interleaved stereo** PCM, which the first version decoded as mono, so those takes played at half
speed, an octave low. Fixed: PCM channels are now inferred from the sample count, and WAV is read
from its header. The cached takes were repaired in place, and the originals are kept in
`raw/misdecoded/`.

Built on v3 for carriers 110, 220 and 440 Hz, all seven scenarios: `river`, `rain_soft` (80 s
loops), `ocean_distant`, `airplane`, `train_smooth`, `car_night` (100 s) and `wind_pines` (120 s).
Rain lost one take, which had a hum line at 237 Hz, next to the 220 Hz carrier.

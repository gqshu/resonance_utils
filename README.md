# resonance_utils

Soundscape generator for an EEG-adaptive power-nap app. One command, `ss_gen`, turns a scenario
(train, rain, river, ...) into seamless loops with a slow, sedative swell baked in, optionally with
an ambient tone layer. The **binaural beat / AM is not added here**: the app's music driver adds it
on its own bus.

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
pip install -e .            # numpy + scipy
```

Requires Python ≥ 3.9. `ffmpeg` is needed only if the API returns MP3 (its fallback when PCM output
isn't available on your plan).

## Usage

```bash
export ELEVENLABS_API_KEY=sk_...
ss_gen --list                                   # scenarios and their prompts
ss_gen train_smooth rain_soft --takes 8 --tone  # two scenarios, with the tone layer
ss_gen all --dry-run                            # show the API calls, spend nothing
ss_gen river --tone --stems                     # also write bed and tone stems
ss_gen --from-dir my_takes/ --name cabin --profile train_smooth   # your own takes, no API
ss_gen all --probe 2 --dry-run                  # cost of testing every prompt variant
ss_gen rain_soft --probe 3                      # 3 takes per variant -> pass-rate table
ss_gen rain_soft --variant c --tone             # build from the variant that won
```

The default model is `eleven_text_to_sound_v3` (v2 is deprecated and will be removed on 2027-03-01).
Each 30 s take costs about 1,200 credits.

**Takes are cached.** They are saved in `<out>/<scenario>/raw/` and reused on every later run, so
re-running to tune the swell or tone costs nothing. Raising `--takes` buys only the missing ones.
A take is reused only if its sidecar records the same prompt and model, so editing a prompt or
switching models never silently reuses old takes. Delete `raw/` to start a scenario over.

Outputs per scenario, in `<out>/<scenario>/` (default `build/`):

| File | What |
|---|---|
| `<scenario>_flat.wav` | conditioned flat bed: no swell, no tone |
| `<scenario>_<rate>Hz[_tone].wav` | one loop per rate, 48 kHz / 24-bit, −26 dBFS RMS |
| `…json` | swell, tone and profile parameters plus QA (EEG-band excess, seam, levels) |
| `…_envelope.csv` | swell phase, envelope, gain and pan at 20 Hz, for EEG alignment |
| `…_bed.wav`, `…_tonelayer.wav` | with `--stems`: the two layers at the same scale as the mix |
| `report.json` | screening results for every take, and the take order in the loop |

## The three stages

### 1. Bed (`bed.py`)

The generator supplies **texture only**: every prompt ends with a "constant level, no swelling" line.
Each take is screened, and every check is measured as **excess over a phase-randomised surrogate**
(a stationary copy with the same spectrum), so ordinary noise fluctuation scores about 0. A take is
rejected for:

- a **regular** rhythm in the 1–12 Hz EEG band (one sharp line, e.g. a train's clack): strict;
- **random** 1–12 Hz flicker above 0.8 dB. This was 0.3 dB, an untuned guess that rejected
  smooth rain. Random patter does not drive a steady-state EEG response;
- excess transients, a tonal line within ±20 Hz of the carrier, near-silence or clipping.

Each comparison averages 3 surrogates, so borderline takes don't flip between pass and reject
on luck. Thresholds live in `bed.THRESH`.

Screening results are stored in each take's sidecar JSON, keyed by the thresholds and the
measurement version, so a rebuild re-screens only new takes or takes judged under other settings.

Passing takes are shuffled and joined with equal-power crossfades into a loop whose length is a
whole multiple of 20 s (the shortest length holding whole waves at 0.1, 0.2 and 0.25 Hz). Then
the bed's slow level is flattened in 6 frequency bands (0.5 s RMS each, so a built-in swell loses
its brightness swing too), transients are capped at +6 dB, a −6 dB ⅓-octave notch is cut at the
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

A warm, open drone of stacked just fifths and octaves (default root 55 Hz, chord `1,3/2,2,3,4`),
with soft 1/h² harmonics and three copies per partial detuned by 0.06 Hz, spread across the stereo
field for slow shimmer. It swells with the bed (`--tone-swell`, default 0.6× the bed's depth), and
its upper harmonics fade in toward each crest. The default level is −12 dB under the bed (`--tone-db`).

Safety rules, enforced automatically (any dropped partial is listed in the JSON):

- no partial within ±20 Hz of the binaural carrier;
- no two partials 0.5–20 Hz apart: that spacing beats as amplitude modulation in the EEG band,
  sleep spindles included. Coinciding partials are merged;
- every frequency is snapped to a whole number of cycles per loop, so the layer loops sample-exactly.

Changing `--tone-root` or `--tone-chord` is safe: offending partials are dropped and reported.
With the default chord and a 144 Hz carrier, only roots of about 55–62 Hz or 110–120 Hz keep every
partial. Most other roots between 40 and 110 Hz put some partial inside the carrier guard band.

## Design rules

- **No added modulation in 1–12 Hz.** A 1–12 Hz envelope evokes steady-state responses in exactly the
  theta/alpha bins the controller reads. Swells at 0.1–0.25 Hz sit far below the 2 Hz band-power floor.
- **Keep the binaural carrier clear.** The bed is notched −6 dB in a ⅓-octave around it (`--carrier`,
  default 144 Hz). Takes with tonal lines near it are rejected, and the tone layer avoids it.
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
├── ss_gen.py      # the CLI: bed -> swell -> tone, outputs + QA
├── bed.py         # ElevenLabs generation (cached), screening, loop join, conditioning
├── swell.py       # offline loop-exact swell
├── tone.py        # ambient tone layer
├── scenarios.py   # prompts + per-texture swell profiles
├── audio_io.py    # WAV I/O, normalisation
└── legacy/        # earlier tools, kept for reference: real-time EEG SwellEngine
                   # (swell_engine.py), ocean_gen.py, elevenlabs_bed_builder.py
```

## Status

v0.2.0. Run against the live API on `rain_soft` and `ocean_distant` (model v2). The API returns
**interleaved stereo** PCM, which the first version decoded as mono, so those takes played at half
speed, an octave low. Fixed: PCM channels are now inferred from the sample count, and WAV is read
from its header. The cached takes were repaired in place, and the originals are kept in
`raw/misdecoded/`.

Built on v3, all seven scenarios: `river` (80 s loop), `ocean_distant`, `rain_soft`, `airplane`,
`train_smooth`, `car_night` (100 s) and `wind_pines` (120 s).

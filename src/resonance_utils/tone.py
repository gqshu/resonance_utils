"""Stage 3 (optional): a sedative ambient tone layer that breathes with the swell.

A warm, open drone: stacked just fifths and octaves (no third, no second, so no pull toward
resolution and no low-register roughness),
soft harmonics (1/h^2), and three slightly detuned copies per partial spread across the
stereo field for slow shimmer. Upper harmonics fade in toward each crest and out in the trough.

Safety rules, so the layer can't interfere with the app's beat/AM or with the EEG:
  * no partial within +/-CARRIER_GUARD_HZ of the binaural carrier (it would beat against it)
  * no two partials 0.5-20 Hz apart: that difference beats as amplitude modulation in the
    EEG band, including 12-16 Hz sleep spindles. Just ratios make harmonics coincide exactly
    (coinciding partials are merged); any clash drops the weaker partial
  * detune is in Hz, not cents, so copy-against-copy beating stays below 0.5 Hz
  * every frequency is snapped to k / loop_length, so the layer loops sample-exactly
"""

from fractions import Fraction

import numpy as np

from .audio_io import FS

DEFAULT_ROOT_HZ = 55.0                 # A1; see plan_partials for why it clears a 144 Hz carrier
DEFAULT_CHORD = "1,3/2,2,3,4"          # root, fifth, octave, twelfth, double octave (just)
HARMONICS = 4
F_MAX_HZ = 1200.0
DETUNE_HZ = 0.06                       # copy spacing; outer copies beat at ~0.12 Hz
CARRIER_GUARD_HZ = 20.0
BEAT_GUARD_HZ = (0.5, 20.0)           # forbidden partial spacing (beats as AM)
MERGE_HZ = 0.5
DRIFT_DEPTH = 0.2                      # each voice's level drifts +/-20 % over 1-3 cycles per loop


def parse_chord(s):
    return [float(Fraction(t.strip())) for t in s.split(",") if t.strip()]


def plan_partials(root, ratios, carrier):
    """List the partials to synthesise, plus the ones dropped and why."""
    parts, dropped = [], []
    for v, r in enumerate(ratios):
        for h in range(1, HARMONICS + 1):
            f = root * r * h
            if f > F_MAX_HZ:
                continue
            p = dict(voice=v, h=h, f=round(f, 3), amp=(1.0 / (1.0 + 0.3 * v)) / h ** 2)
            if carrier and abs(f - carrier) < CARRIER_GUARD_HZ:
                dropped.append({**p, "why": f"within {CARRIER_GUARD_HZ:g} Hz of the {carrier:g} Hz carrier"})
                continue
            parts.append(p)
    kept = []
    for p in sorted(parts, key=lambda q: -q["amp"]):
        same = next((q for q in kept if abs(p["f"] - q["f"]) < MERGE_HZ), None)
        if same:                                       # coinciding harmonics: one partial, power sum
            same["amp"] = float(np.hypot(same["amp"], p["amp"]))
            continue
        clash = next((q for q in kept if BEAT_GUARD_HZ[0] <= abs(p["f"] - q["f"]) <= BEAT_GUARD_HZ[1]), None)
        if clash:
            dropped.append({**p, "why": f"beats at {abs(p['f'] - clash['f']):.2f} Hz with {clash['f']} Hz"})
        else:
            kept.append(p)
    return sorted(kept, key=lambda q: q["f"]), dropped


def render_tone_stems(n, root=DEFAULT_ROOT_HZ, chord=DEFAULT_CHORD, carrier=None, seed=0, chunk_s=10.0):
    """Render the layer once as two static loops: fundamentals and upper harmonics.
    mix_tone() then shapes them with each rate's swell envelope. Returns (fund, upper, info)."""
    kept, dropped = plan_partials(root, parse_chord(chord), carrier)
    rng = np.random.default_rng(seed + 5000)
    dk = max(1, round(DETUNE_HZ * n / FS))             # detune in FFT bins of the loop
    drift = {v: (int(rng.integers(1, 4)), rng.uniform(0, 2 * np.pi)) for v in {p["voice"] for p in kept}}
    osc = []
    for p in kept:
        k0 = round(p["f"] * n / FS)                    # snap: exactly k0 cycles per loop
        for off, pan in ((-dk, -0.5), (0, 0.0), (dk, 0.5)):
            th = (pan + 1) * np.pi / 4
            osc.append((k0 + off, rng.uniform(0, 2 * np.pi), p["amp"] / np.sqrt(3),
                        np.cos(th), np.sin(th), p["voice"], p["h"] > 1))

    fund = np.zeros((n, 2), np.float32)
    upper = np.zeros((n, 2), np.float32)
    step = int(chunk_s * FS)
    for s in range(0, n, step):
        i = np.arange(s, min(n, s + step), dtype=np.int64)
        dr = {v: 1 + DRIFT_DEPTH * np.sin(2 * np.pi * ((m * i) % n) / n + ph) for v, (m, ph) in drift.items()}
        for k, ph, a, gl, gr, v, up in osc:
            w = a * dr[v] * np.sin(2 * np.pi * ((k * i) % n) / n + ph)   # integer phase: exact loop
            tgt = upper if up else fund
            tgt[s:s + len(i), 0] += gl * w
            tgt[s:s + len(i), 1] += gr * w
    info = dict(root_hz=root, chord=chord, partials_hz=[p["f"] for p in kept],
                dropped=[{"f": p["f"], "why": p["why"]} for p in dropped],
                detune_hz=round(dk * FS / n, 4), carrier_guard_hz=CARRIER_GUARD_HZ)
    return fund, upper, info


def mix_tone(fund, upper, env, depth_db):
    """Shape the static stems with a swell envelope: the level swells by depth_db and
    upper harmonics fade from 30 % in the trough to 100 % at the crest."""
    g = 10 ** (-depth_db * (1.0 - env) / 20.0)
    bright = 0.3 + 0.7 * env
    return (fund + upper * bright[:, None]) * g[:, None]

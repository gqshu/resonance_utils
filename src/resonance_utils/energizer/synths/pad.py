"""The `pad`: the engine's AM carrier (plan §6.1).

Follows the chord timeline with 3-5 just-tuned chord tones voiced inside the AM band (above the
monaural slot), minimal-movement voice leading, explicit harmonics. Every partial is checked:
none in a monaural keep-clear range, none beating against a lever tone in a native-mod band,
none above 2 x am_band.hi, and no pair of partials beating inside native_mod_bands_hz
(the weaker one is dropped). Movement is slow only: per-voice level random walk (3 s), a tilt
LFO at 0.03 Hz and the arc's brightness; chord changes cross over in >= 0.4 s."""

import numpy as np

from ..config import partial_clash
from ..theory import JUST, midi_hz
from . import additive

TIMBRES = {
    "warm_additive":   dict(h=6, slope=1.6, copies=2, detune_hz=0.25),
    "glass":           dict(h=5, slope=1.0, copies=2, detune_hz=0.18, odd_boost=0.6),
    "string_ensemble": dict(h=8, slope=1.2, copies=3, detune_hz=0.4),
}
MERGE_HZ = 0.5
AUDIT_FLOOR_DB = -40.0
XFADE_S = 0.5
TRANSITION_MARGIN_HZ = 3.0     # a beat that lasts only one crossover smears ~ +/-2 Hz: widen the bands
TILT_REF_HZ = 300.0


def harm_amp(tb, h):
    a = 1.0 / h ** tb["slope"]
    if "odd_boost" in tb and h % 2 == 0:
        a *= tb["odd_boost"]
    return a


def voice_range(slots):
    lo = max(slots["am_band"]["lo_hz"], max(hi for _, hi in slots["mb_keep_clear_hz"])) + 12.0
    return lo, min(760.0, slots["am_band"]["hi_hz"] * 0.76)


def chord_tones(chord, n_voices):
    """(semitones over the chord root) for 3-5 voices: root, third, fifth, then 9th, octave."""
    if chord["quality"] == "fallback":                # low-confidence harmony: root, fifth, ninth
        return [0, 7, 14][:max(3, n_voices)]
    tones = [0, chord["third"], 7]
    if n_voices >= 4:
        tones.append(14 if chord["quality"] == "major" else 12)
    if n_voices >= 5:
        tones.append(12 if 14 in tones else 19)
    return tones


def tone_hz(chord, semis, tuning, ref_midi):
    root = midi_hz(ref_midi)
    if tuning == "equal":
        return root * 2 ** (semis / 12)
    return root * JUST.get(semis, 2 ** (semis / 12))


def voice_chord(chord, prev, slots, tuning, n_voices):
    """Fundamentals for this chord, each tone in the octave closest to the previous voicing."""
    lo, hi = voice_range(slots)
    ref = 48 + chord["root_pc"]                        # chord root in octave 3; octaves added below
    centre = prev if prev else [np.sqrt(lo * hi)]
    out = []
    for s in chord_tones(chord, n_voices):
        base = tone_hz(chord, s, tuning, ref)
        cands = [base * 2 ** k for k in range(-2, 4) if lo <= base * 2 ** k <= hi]
        cands = [f for f in cands if partial_clash(f, slots) is None and all(abs(f - g) > 3 for g in out)]
        if not cands:
            continue
        out.append(min(cands, key=lambda f: min(abs(np.log2(f / c)) for c in centre)))
    return sorted(out)


def chord_partials(voicing, tb, slots, extra=()):
    """Partials for one chord, plus the dropped ones and why. extra: partials of other stems
    sounding at the same time (e.g. bass), audited against but never dropped here."""
    fmax = 2 * slots["am_band"]["hi_hz"]
    parts, dropped = [], []
    for v, f0 in enumerate(voicing):
        for h in range(1, tb["h"] + 1):
            f = f0 * h
            if f > fmax:
                break
            why = partial_clash(f, slots)
            p = dict(f=round(f, 4), amp=harm_amp(tb, h) * (1.0 if v else 1.15), voice=v, h=h)
            if why:
                dropped.append({**p, "why": why})
            else:
                parts.append(p)
    kept = []
    for p in sorted(parts, key=lambda q: -q["amp"]):
        same = next((q for q in kept if abs(q["f"] - p["f"]) < MERGE_HZ), None)
        if same:
            same["amp"] = float(np.hypot(same["amp"], p["amp"]))
            continue
        kept.append(p)
    ref = max(p["amp"] for p in kept)
    audit, ok = [], []
    for p in sorted(kept, key=lambda q: -q["amp"]):
        others = ok + list(extra)
        bad = None
        for q in others:
            d = abs(p["f"] - q["f"])
            weak = min(p["amp"], q["amp"]) / ref
            if 20 * np.log10(weak + 1e-12) > AUDIT_FLOOR_DB and any(lo <= d <= hi for lo, hi in slots["native_mod_bands_hz"]):
                bad = q
                break
        if bad is not None:
            dropped.append({**p, "why": f"beats at {abs(p['f'] - bad['f']):.2f} Hz with {bad['f']} Hz"})
            audit.append((p["f"], bad["f"]))
        else:
            ok.append(p)
    return sorted(ok, key=lambda q: q["f"]), dropped


def plan(chords, slots, timbre, tuning, plateau_s, bass_partials=None):
    """Voicing + partials per chord. 3 voices before the plateau, 4 from it (fuller)."""
    tb = TIMBRES[timbre]
    prev, rows = None, []
    for i, c in enumerate(chords):
        nv = 4 if c["start_s"] >= plateau_s[0] - 1e-6 else 3
        voicing = voice_chord(c, prev, slots, tuning, nv)
        extra = (bass_partials or {}).get(i, [])
        parts, dropped = chord_partials(voicing, tb, slots, extra)
        rows.append(dict(chord=i, label=c["label"], voicing_hz=[round(f, 3) for f in voicing],
                         partials=parts, dropped=dropped, native_mod_bands_hz=slots["native_mod_bands_hz"]))
        prev = voicing
    return rows


def slots_bands(rows):
    return rows[0]["native_mod_bands_hz"] if rows else []


def transition_clashes(rows, bands):
    """{(chord, f)} partials that must leave early / arrive late at a chord change, because
    they would beat inside a native-mod band with a partial of the neighbouring chord."""
    out_late, in_early = set(), set()
    for i in range(len(rows) - 1):
        a, b = rows[i]["partials"], rows[i + 1]["partials"]
        ref = max([p["amp"] for p in a + b] or [1.0])
        for p in a:
            if any(abs(p["f"] - q["f"]) < MERGE_HZ for q in b):
                continue                                  # persists: one continuous sine
            for q in b:
                d = abs(p["f"] - q["f"])
                m = TRANSITION_MARGIN_HZ
                if any(lo - m <= d <= hi + m for lo, hi in bands) and \
                        20 * np.log10(min(p["amp"], q["amp"]) / ref + 1e-12) > AUDIT_FLOOR_DB:
                    out_late.add((i, p["f"]))
                    in_early.add((i + 1, q["f"]))
    return out_late, in_early


def render(chords, rows, n, n_cr, timbre, brightness_db, seed):
    """Pad stem (dry). brightness_db: CR curve of tilt in dB/octave."""
    tb = TIMBRES[timbre]
    rng = np.random.default_rng([seed, 21])
    walk = [additive.slow_noise(rng, n_cr, 3.0, 1.5) for _ in range(5)]
    lfo = 0.75 * np.sin(2 * np.pi * 0.03 * np.arange(n_cr) / additive.CR + rng.uniform(0, 2 * np.pi))
    tilt = brightness_db + lfo
    out_late, in_early = transition_clashes(rows, slots_bands(rows))
    half = int(round(XFADE_S / 2 * additive.CR))
    tracks = {}
    for i, (c, row) in enumerate(zip(chords, rows)):
        c0 = int(round(c["start_s"] * additive.CR))
        c1 = min(n_cr, int(round(c["end_s"] * additive.CR)))
        for p in row["partials"]:
            # a partial that would beat with one of the next/previous chord's partials during the
            # crossover hands over in sequence: out before the boundary, in after it, no overlap
            a0 = c0 + (half if (i, p["f"]) in in_early else 0)
            a1 = c1 - (half if (i, p["f"]) in out_late else 0)
            nv = max(1, len(row["voicing_hz"]))
            for k in range(tb["copies"]):
                off = (k - (tb["copies"] - 1) / 2) * tb["detune_hz"] / max(1, tb["copies"] - 1)
                key = (round(p["f"], 2), k)
                tr = tracks.get(key)
                if tr is None:
                    th = np.pi / 4 + (0.5 if k % 2 else -0.5) * (0.35 + 0.35 * p["voice"] / nv)
                    tr = tracks[key] = dict(f=p["f"] + off, phase=float(rng.uniform(0, 2 * np.pi)),
                                            gl=np.cos(th), gr=np.sin(th), step=np.zeros(n_cr))
                g = 10 ** (tilt[a0:a1] * np.log2(p["f"] / TILT_REF_HZ) / 20)
                tr["step"][a0:a1] = p["amp"] / np.sqrt(tb["copies"]) * walk[p["voice"] % 5][a0:a1] * g
    out = []
    for tr in tracks.values():
        tr["amp"] = additive.smooth_power(tr.pop("step"), XFADE_S)
        out.append(tr)
    return additive.render(out, n)

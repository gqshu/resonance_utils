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


def invert(voicing, slots):
    """Next inversion: lowest voice up an octave if it fits, else the top voice down."""
    lo, hi = voice_range(slots)
    v = sorted(voicing)
    if v and v[0] * 2 <= hi and partial_clash(v[0] * 2, slots) is None and all(abs(v[0] * 2 - g) > 3 for g in v):
        return sorted(v[1:] + [v[0] * 2])
    if v and v[-1] / 2 >= lo and partial_clash(v[-1] / 2, slots) is None and all(abs(v[-1] / 2 - g) > 3 for g in v):
        return sorted([v[-1] / 2] + v[:-1])
    return v


ENV_CHECK_FS = 8000
ENV_LIMIT_DB = -43.0           # per chord, 3 dB under the bed's provisional PAD_NATIVE_MOD_MAX_DB


def env_mod_db(parts, slots, seeds=3, seconds=2.0):
    """Amplitude-envelope modulation (dB re 100 %) of a sustained chord inside the AM band, in each
    native-mod band, max over a few random phase sets. Pairwise spacing misses this: a just minor
    triad (10:12:15) repeats at root/10 (19.6 Hz for G), so its envelope pulses there even though
    no two partials are 12-20 Hz apart."""
    from ...core.analysis import band_std
    am = slots["am_band"]
    fs = [p for p in parts if am["lo_hz"] <= p["f"] <= am["hi_hz"]]
    if len(fs) < 2:
        return -120.0
    t = np.arange(int(seconds * ENV_CHECK_FS)) / ENV_CHECK_FS
    blk = ENV_CHECK_FS // 1000
    worst = -120.0
    for s in range(seeds):
        ph = np.random.default_rng(s).uniform(0, 2 * np.pi, len(fs))
        x = sum(p["amp"] * np.sin(2 * np.pi * p["f"] * t + q) for p, q in zip(fs, ph))
        env = np.sqrt((x[: len(x) // blk * blk] ** 2).reshape(-1, blk).mean(1))
        for lo, hi in slots["native_mod_bands_hz"]:
            d = band_std(env, 1000, lo, hi) * np.sqrt(2) / (env.mean() + 1e-12)
            worst = max(worst, 20 * np.log10(d + 1e-9))
    return float(worst)


THIRD_TUNINGS = {3: (6 / 5, 2 ** (3 / 12), 7 / 6), 4: (5 / 4, 2 ** (4 / 12))}   # just first


def retune_third(voicing, chord, ratio):
    """The voicing with every chord third (any octave) retuned to `ratio` over the chord root."""
    if chord.get("third") not in THIRD_TUNINGS:
        return voicing
    just = THIRD_TUNINGS[chord["third"]][0]
    root = midi_hz(48 + chord["root_pc"])
    out = []
    for f in voicing:
        r = f / root
        k = np.floor(np.log2(r))
        out.append(f * ratio / just if abs(r / 2 ** k - just) < 2e-3 else f)
    return sorted(out)


def clear_envelope(voicing, tb, slots, extra, limit=ENV_LIMIT_DB, chord=None):
    """Best of a few voicings by envelope modulation: inversions, and for each the chord's third in
    just, equal-tempered and (minor) septimal tuning. A just triad repeats at root/10 (minor) or
    root/4 (major); when that lands in a native-mod band only the third's tuning moves it. If no
    candidate is under the limit, drop upper harmonics (highest h first) of the best until it is.
    Returns (voicing, parts, dropped, db)."""
    base = [voicing]
    v = voicing
    for _ in range(min(3, len(voicing))):
        v = invert(v, slots)
        if v not in base:
            base.append(v)
    tunings = THIRD_TUNINGS.get((chord or {}).get("third"), (None,))
    cands = []
    for r in tunings:
        for vc in base:
            vv = vc if r is None or chord is None else retune_third(vc, chord, r)
            if all(partial_clash(f, slots) is None for f in vv) and vv not in cands:
                cands.append(vv)
    scored = []
    for vc in cands:
        parts, dropped = chord_partials(vc, tb, slots, extra)
        scored.append((env_mod_db(parts, slots), vc, parts, dropped))
        if scored[-1][0] <= limit:
            break
    db, vc, parts, dropped = min(scored, key=lambda r: r[0])
    while db > limit:
        upper = [p for p in parts if p["h"] > 1]
        if not upper:
            break
        worst = max(upper, key=lambda p: (p["h"], -p["amp"]))
        parts = [p for p in parts if p is not worst]
        dropped = dropped + [{**worst, "why": "envelope periodicity in a native-mod band"}]
        db = env_mod_db(parts, slots)
    return vc, parts, dropped, round(db, 1)


def plan(chords, slots, timbre, tuning, plateau_s, bass_partials=None, sections=None):
    """Voicing + partials per chord. 3 voices before the plateau, 4 from it (fuller). With
    sections, a chord that comes back within a section alternates between its voicing and the
    next inversion, so a looping progression doesn't sound identical every time."""
    from .pulse import section_at
    tb = TIMBRES[timbre]
    prev, rows = None, []
    seen = {}
    for i, c in enumerate(chords):
        nv = 4 if c["start_s"] >= plateau_s[0] - 1e-6 else 3
        voicing = voice_chord(c, prev, slots, tuning, nv)
        if sections:
            sec = section_at(sections, c["start_beat"])
            key = (sec["index"] if sec else None, c["label"])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] % 2 == 0:
                voicing = invert(voicing, slots)
        extra = (bass_partials or {}).get(i, [])
        voicing, parts, dropped, env_db = clear_envelope(voicing, tb, slots, extra, chord=c)
        rows.append(dict(chord=i, label=c["label"], voicing_hz=[round(f, 3) for f in voicing],
                         partials=parts, dropped=dropped, env_mod_db=env_db,
                         native_mod_bands_hz=slots["native_mod_bands_hz"]))
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


def render(chords, rows, n, n_cr, timbre, brightness_db, seed, split=False):
    """Pad stem (dry). brightness_db: CR curve of tilt in dB/octave. With split, returns
    (main, handover): the partials that hand over in sequence at a chord change go to a separate
    bus that must stay out of the reverb, or the old partial's tail rings under the new one (a
    whole-tone step around 330-370 Hz is a 40 Hz beat)."""
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
    flagged = {round(f, 2) for _, f in out_late | in_early}
    main, hand = [], []
    for (fk, _), tr in tracks.items():
        tr["amp"] = additive.smooth_power(tr.pop("step"), XFADE_S)
        (hand if split and fk in flagged else main).append(tr)
    if not split:
        return additive.render(main, n)
    return additive.render(main, n), additive.render(hand, n)

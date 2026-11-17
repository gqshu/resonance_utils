"""The `pulse` (plan §6.3): soft kick and hats/shakers on the exact tempo grid.

No 32nd notes (at 118 BPM they fall at 15.7 Hz, the engine's AM rate). Velocity varies +/-2 dB,
timing jitter is 0. Layers build with the arc for density: kick from the pulse entry, offbeat
hats from the middle of the ramp, 16th shakers in the plateau; the hand-off thins them out."""

import numpy as np
from scipy import fft as sfft

from ...core.audio_io import FS

STYLES = {
    #               kick beats in bar        hat 8ths   16th shaker
    "soft_four":   dict(kick=[0, 1, 2, 3],    hats=True, shaker=True),
    "broken_soft": dict(kick=[0, 1.5, 2.5],   hats=True, shaker=True),
    "shaker_only": dict(kick=[],              hats=False, shaker=True),
    "none":        dict(kick=[],              hats=False, shaker=False),
}


def _bandpass(x, lo, hi):
    f = sfft.rfftfreq(len(x), 1 / FS)
    g = (1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 8)) / np.sqrt(1 + (f / hi) ** 8)
    return sfft.irfft(sfft.rfft(x) * g, len(x))


def kick_sample(rng):
    n = int(0.4 * FS)
    t = np.arange(n) / FS
    f = 50 + 70 * np.exp(-t / 0.04)
    body = np.sin(2 * np.pi * np.cumsum(f) / FS) * np.minimum(1, t / 0.003) * np.exp(-t / 0.16)
    click = _bandpass(rng.standard_normal(n), 2000, 9000) * np.exp(-t / 0.004) * 0.08
    return body + click


def noise_hit(rng, lo, hi, attack, decay, length=0.25):
    n = int(length * FS)
    t = np.arange(n) / FS
    env = np.minimum(1, t / attack) * np.exp(-t / decay)
    return _bandpass(rng.standard_normal(n), lo, hi) * env


def section_at(sections, beat):
    for s in sections or []:
        if s["start_beat"] <= beat < s["end_beat"]:
            return s
    return None


def events(style, tempo, entry_s, ramp_s, plateau_s, handoff_s, seed, sections=None):
    """[(beat, kind, vel)]: every onset, in beats.

    With sections (hybrid: the AI track's sections in bed time) the pulse follows the music:
    a breakdown drops kick and hats to soft shakers, each main section gets the next of three
    pattern variants (as styled / half-time kick / busier 16th ghosts), and the last bar before
    a section change is a fill (kick out on beat 4, 16th shakers rising), so the groove changes
    when the music does. Never anything finer than 16ths."""
    st = STYLES[style]
    mains = [s["index"] for s in (sections or []) if s["role"] == "main"]
    rng = np.random.default_rng([seed, 31])
    hats_from = (ramp_s[0] + ramp_s[1]) / 2
    ev = []
    b = np.ceil(tempo.beat_at(entry_s) / 4) * 4            # start on a bar line
    while b < tempo.total_beats - 1e-9:
        t_bar = float(tempo.time_of_beat(b))
        sec = section_at(sections, b)
        nxt = section_at(sections, b + 4)
        fill = sec is not None and nxt is not None and nxt is not sec and t_bar >= ramp_s[0]
        if sec is not None and sec["role"] == "breakdown":
            ev += [(b + k + o, "shaker", 0.3 if o else 0.4) for k in range(4) for o in (0.0, 0.5)]
            b += 4
            continue
        var = mains.index(sec["index"]) % 3 if sec is not None and sec["index"] in mains else 0
        kicks = [0, 2] if (var == 1 and st["kick"]) else st["kick"]
        if fill:
            kicks = [k for k in kicks if k < 3]
            ev += [(b + 2 + j * 0.25, "shaker", 0.3 + 0.05 * j) for j in range(8)]
        for k in kicks:
            ev.append((b + k, "kick", 1.0 if k == 0 else 0.85))
        if var == 2 and t_bar >= plateau_s[0] and t_bar < handoff_s[0]:
            ev += [(b + k + 0.75, "hat", 0.35) for k in range(4)]
        if st["hats"] and t_bar >= hats_from and t_bar < handoff_s[0] + (handoff_s[1] - handoff_s[0]) / 2:
            ev += [(b + k + 0.5, "hat", 0.7) for k in range(4)]
        if st["shaker"] and (t_bar >= plateau_s[0] or (not st["kick"] and t_bar >= ramp_s[0])) \
                and t_bar < handoff_s[0]:
            ev += [(b + k + o, "shaker", 0.45 if o == 0.25 else 0.35) for k in range(4) for o in (0.25, 0.75)]
            if not st["kick"]:
                ev += [(b + k, "shaker", 0.6) for k in range(4)]
        b += 4
    ev = [(bb, kind, v * 10 ** (rng.uniform(-2, 2) / 20)) for bb, kind, v in ev if bb < tempo.total_beats]
    return sorted(ev)


def render(ev, tempo, n, seed, brightness_db_at):
    rng = np.random.default_rng([seed, 32])
    kick = kick_sample(rng)
    hats = [noise_hit(rng, 4000, 12000, 0.001, 0.035) for _ in range(8)]
    shakers = [noise_hit(rng, 5000, 13000, 0.008, 0.05) for _ in range(8)]
    out = np.zeros((n, 2))
    onsets = []
    for i, (b, kind, vel) in enumerate(ev):
        t = float(tempo.time_of_beat(b))
        s0 = int(round(t * FS))
        if s0 >= n:
            continue
        onsets.append(dict(beat=round(b, 4), t_s=t, sample=s0, kind=kind))
        if kind == "kick":
            x, pan = kick * 0.9, 0.5
        else:
            x = (hats if kind == "hat" else shakers)[i % 8]
            x = x * (0.22 if kind == "hat" else 0.15) * 10 ** (min(0.0, brightness_db_at(t)) * 1.5 / 20)
            pan = 0.62 if kind == "hat" else (0.35 if i % 2 else 0.68)
        m = min(len(x), n - s0)
        out[s0:s0 + m, 0] += vel * np.cos(pan * np.pi / 2) * x[:m] * np.sqrt(2)
        out[s0:s0 + m, 1] += vel * np.sin(pan * np.pi / 2) * x[:m] * np.sqrt(2)
    return out, onsets

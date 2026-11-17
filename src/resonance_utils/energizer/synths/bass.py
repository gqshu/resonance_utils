"""The `bass` (plan §6.2): sine + 2nd/3rd harmonic under 180 Hz, playing the chord root.

Ramp: one held note per chord. Plateau: an 8th-note pattern (soft accents, legato-ish, no
pumping). Attacks >= 15 ms. Onsets are tempo-map times rounded to the sample (<= 0.01 ms)."""

import numpy as np

from ...core.audio_io import FS
from ..config import partial_clash
from ..theory import midi_hz

LOWPASS_HZ = 180.0
ROOT_RANGE_HZ = (41.0, 82.0)
HARMONICS = ((1, 1.0), (2, 0.35), (3, 0.15))
ATTACK_S, RELEASE_S = 0.04, 0.3
PAT_ATTACK_S, PAT_RELEASE_S = 0.015, 0.04
PATTERN = [(0.0, 1.0), (0.5, 0.55), (1.0, 0.8), (1.5, 0.55), (2.0, 0.9), (2.5, 0.55), (3.0, 0.8), (3.5, 0.6)]
# (beat, velocity, length in beats): a second plateau pattern, half notes with an 8th pickup
PATTERN_B = [(0.0, 1.0, 1.9), (2.0, 0.9, 1.4), (3.5, 0.6, 0.45)]


def root_hz(pc):
    f = midi_hz(36 + pc)                               # C2 + pc
    while f >= ROOT_RANGE_HZ[1]:
        f /= 2
    while f < ROOT_RANGE_HZ[0]:
        f *= 2
    return f


def partials(f0, slots):
    return [dict(f=round(f0 * h, 4), amp=a) for h, a in HARMONICS
            if f0 * h < LOWPASS_HZ and partial_clash(f0 * h, slots) is None]


def notes(chords, tempo, entry_s, plateau_s, handoff_s, sections=None):
    """Note list: dict(t0, t1, pc, vel, attack, release, beat). With sections, a breakdown holds
    soft notes and the main plateau sections alternate between the 8th pattern and PATTERN_B."""
    from .pulse import section_at
    mains = [s["index"] for s in (sections or []) if s["role"] == "main"]
    out = []
    for c in chords:
        sec = section_at(sections, c["start_beat"])
        if sec is not None and sec["role"] == "breakdown" and c["end_s"] > entry_s:
            t0 = max(c["start_s"], entry_s)
            out.append(dict(t0=t0, t1=c["end_s"], pc=c["root_pc"], vel=0.7, attack=ATTACK_S,
                            release=RELEASE_S, beat=float(tempo.beat_at(t0))))
            continue
        if c["end_s"] <= entry_s:
            continue
        if c["start_s"] < plateau_s[0]:
            t0 = max(c["start_s"], entry_s)
            out.append(dict(t0=t0, t1=c["end_s"], pc=c["root_pc"], vel=1.0, attack=ATTACK_S,
                            release=RELEASE_S, beat=float(tempo.beat_at(t0))))
            continue
        b = c["start_beat"]
        alt = sec is not None and sec["index"] in mains and mains.index(sec["index"]) % 2 == 1
        pattern = PATTERN_B if alt else [(o, v, 0.45) for o, v in PATTERN]
        while b < c["end_beat"] - 1e-9:
            for off, vel, ln in pattern:
                bb = b + off
                if bb >= c["end_beat"] - 1e-9:
                    break
                t0 = float(tempo.time_of_beat(bb))
                t1 = float(tempo.time_of_beat(min(bb + ln, c["end_beat"])))
                if t0 >= handoff_s[0]:                  # hand-off: back to held notes, thinner
                    vel *= 0.8
                out.append(dict(t0=t0, t1=t1, pc=c["root_pc"], vel=vel, attack=PAT_ATTACK_S,
                                release=PAT_RELEASE_S, beat=bb))
            b += 4
    return out


def render(note_list, n, slots):
    out = np.zeros(n)
    for nt in note_list:
        s0 = int(round(nt["t0"] * FS))
        s1 = min(n, int(round((nt["t1"] + nt["release"]) * FS)))
        if s0 >= n:
            continue
        t = np.arange(s1 - s0) / FS
        hold = nt["t1"] - nt["t0"]
        env = np.minimum(1.0, t / nt["attack"])
        env *= np.where(t <= hold, 1.0, np.clip(1 - (t - hold) / nt["release"], 0, 1))
        env = 0.5 - 0.5 * np.cos(np.pi * env)          # smooth shape, no corners
        f0 = root_hz(nt["pc"])
        v = sum(p["amp"] * np.sin(2 * np.pi * p["f"] * t) for p in partials(f0, slots))
        out[s0:s1] += nt["vel"] * env * v
    from scipy import fft as sfft
    f = sfft.rfftfreq(n, 1 / FS)
    lp = 1.0 / np.sqrt(1.0 + (f / LOWPASS_HZ) ** 8)
    y = sfft.irfft(sfft.rfft(out) * lp, n)
    return np.stack([y, y], 1)

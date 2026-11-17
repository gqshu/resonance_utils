"""Arc profiles -> phase times and per-stem gain / brightness curves (plan §3).

Curves are breakpoints in the 4-minute reference timeline, scaled to the profile length.
Gains are dB relative to each stem's plateau level; None = silent (the stem fades in from
there). Brightness is a spectral tilt in dB/octave (negative = darker), applied per stem.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import load_yaml

REF_LENGTH_S = 240.0
CR = 100                                   # control rate (frames/s) for every curve

# (t_s in the 240 s reference, dB). The pad level is held from 0:30 (steady AM carrier):
# the build comes from the other stems entering and the brightness opening.
GAIN_DB = {
    "pad":     [(0, -8), (30, -5), (220, -5), (240, -6.5)],
    "texture": [(0, -8), (30, -5), (150, -4), (220, -4), (240, -7)],
    "pulse":   [(45, None), (53, -8), (150, 0), (220, 0), (240, -12)],
    "bass":    [(60, None), (68, -6), (150, 0), (220, 0), (240, -4)],
    # AI layers (hybrid / reference): music from 0:30 (its 8 s pre-roll fades in), motif out by 0:30
    "music":   [(22, None), (30, -6), (90, -3), (150, 0), (220, 0), (240, -5)],
    "motif":   [(0, -2), (22, -2), (30, None)],
}
BRIGHTNESS_DB_PER_OCT = [(0, -4.0), (30, -4.0), (150, 0.0), (220, 0.0), (240, -1.0)]
SILENT_DB = -60.0
EDGE_FADE_S = (2.0, 1.5)                   # global fade-in / fade-out so the bed never starts on a step


@dataclass
class Arc:
    name: str
    length_s: float
    phases: dict
    entries: dict
    scale: float
    gains: dict = field(default_factory=dict)
    brightness: list = field(default_factory=list)

    def t(self, t_ref):
        return t_ref * self.scale

    def curve(self, points, n):
        """Breakpoints (reference time) -> array at CR with n frames, interpolated in dB."""
        tt = np.arange(n) / CR
        tp = [self.t(p[0]) for p in points]
        vp = [SILENT_DB if p[1] is None else p[1] for p in points]
        y = np.interp(tt, tp, vp)
        if points[0][1] is None:
            y[tt < tp[0]] = SILENT_DB
        return y

    def gain_curve(self, stem, n):
        """Linear gain at CR, with the global edge fades."""
        g = 10 ** (self.curve(self.gains[stem], n) / 20)
        g[g <= 10 ** (SILENT_DB / 20) * 1.0001] = 0.0
        tt = np.arange(n) / CR
        fi, fo = EDGE_FADE_S
        g *= np.clip(tt / fi, 0, 1) * np.clip((self.length_s - tt) / fo, 0, 1)
        return g

    def brightness_curve(self, n):
        return self.curve(self.brightness, n)

    def export(self):
        sc = lambda pts: [dict(t_s=round(self.t(t), 3), db=v) for t, v in pts]
        return dict(profile=self.name, length_s=self.length_s, control_rate_hz=CR,
                    phases={k: [round(a, 3), round(b, 3)] for k, (a, b) in self.phases.items()},
                    entries_s={k: round(v, 3) for k, v in self.entries.items()},
                    gain_db=dict((k, sc(v)) for k, v in self.gains.items()),
                    brightness_db_per_oct=sc(self.brightness),
                    note="curves are already baked into the stems; null db = silent")


def load(profile, length_s=None):
    profs = load_yaml("arc_profiles.yaml")
    p = profs[profile]
    base = profs[p["scale_from"]] if "scale_from" in p else p
    length = float(length_s or p["length_s"])
    k = length / float(base["length_s"])
    scale = length / REF_LENGTH_S
    phases = {n: (a * k, b * k) for n, (a, b) in base["phases"].items()}
    entries = {n: v * k for n, v in base["entries"].items()}
    return Arc(profile, length, phases, entries, scale, dict(GAIN_DB), list(BRIGHTNESS_DB_PER_OCT))

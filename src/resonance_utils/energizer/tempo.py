"""Tempo map: bpm(t) for the arc, and exact beat <-> time conversion (plan §3).

The ramp is linear in BPM over time between the arc's phase times (so the phases land exactly
at their table times). Beats are the exact integral b(t) = integral of bpm/60 dt, evaluated by the
trapezoid rule on a 1/4800 s grid with every breakpoint on that grid; bpm is piecewise linear,
so that rule is exact and t(b) is interpolated from it (error far below 1 us). Every procedural
event is scheduled in beats and converted to time here.
"""

import numpy as np

GRID_HZ = 4800


class TempoMap:
    def __init__(self, length_s, bpm_points):
        """bpm_points: [(t_s, bpm), ...] breakpoints of a piecewise-linear bpm(t)."""
        self.length_s = float(length_s)
        self.points = [(float(t), float(b)) for t, b in bpm_points]
        n = int(round(self.length_s * GRID_HZ))
        self._t = np.arange(n + 1) / GRID_HZ
        tp, bp = zip(*self.points)
        self._bpm = np.interp(self._t, tp, bp)
        inc = (self._bpm[1:] + self._bpm[:-1]) / 2 / 60 / GRID_HZ
        self._b = np.concatenate([[0.0], np.cumsum(inc)])
        self.total_beats = float(self._b[-1])

    def bpm_at(self, t):
        return np.interp(t, self._t, self._bpm)

    def beat_at(self, t):
        return np.interp(t, self._t, self._b)

    def time_of_beat(self, b):
        return np.interp(b, self._b, self._t)

    def grid(self, step_beats):
        """Times (s) of every grid point step_beats apart that falls inside the bed."""
        b = np.arange(0, self.total_beats, step_beats)
        return self.time_of_beat(b)

    def export(self):
        """tempo_map.json: bpm at 100 Hz plus beat, 16th and 32nd onset times (s)."""
        t100 = np.arange(0, self.length_s, 0.01)
        r = lambda a: [round(float(v), 6) for v in a]
        return dict(
            bpm_rate_hz=100, bpm=[round(float(v), 3) for v in self.bpm_at(t100)],
            breakpoints=[dict(t_s=t, bpm=b) for t, b in self.points],
            total_beats=round(self.total_beats, 4), beats_per_bar=4,
            beats_s=r(self.grid(1.0)), sixteenths_s=r(self.grid(0.25)), thirtyseconds_s=r(self.grid(0.125)),
        )


def from_arc(arc, plateau_bpm, entry_ratio):
    """Entry at entry tempo, linear BPM ramp across the ramp phase, plateau tempo after."""
    entry = plateau_bpm * entry_ratio
    r0, r1 = arc.phases["ramp"]
    return TempoMap(arc.length_s, [(0.0, entry), (r0, entry), (r1, plateau_bpm), (arc.length_s, plateau_bpm)])

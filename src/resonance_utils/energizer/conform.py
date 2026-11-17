"""Conforming the AI `music` and `motif` to the bed (plan §7.3).

1. Bar map. The AI track is at constant plateau tempo; the bed ramps up to it, so the ramp holds
   fewer bars than the AI's build sections. Each AI section (from the composition plan) is mapped
   onto its bed phase bar for bar; a surplus is dropped evenly across the section (a shortfall
   repeats bars), always whole bars.
2. Splice the chosen bars at AI tempo (equal-power 40 ms crossfades on the bar lines, only where
   the bar sequence jumps), so the edit is a plain constant-tempo track.
3. Time-map it onto the bed's tempo curve, one anchor per beat (stretch.py, Rubber Band), within
   MAX_STRETCH.
4. Band handling: band-stop the monaural keep-clear range (>= 18 dB), lower the AM band by
   AI_IN_AM_BAND_GAIN_DB (AI content there competes with the pad, the AM carrier).
"""

import numpy as np
from scipy import fft as sfft

from ..core.audio_io import FS
from ..core.slots import band_stop
from .stretch import stretch_to_map

SPLICE_XF_S = 0.04


def ai_sections(plan, preroll_s, phases=None):
    """(phase, start_s, end_s) in the AI track for the four bed phases, merging the plan's chunks
    (phases[i] is chunk i's phase; default: one chunk per phase, in order)."""
    phases = phases or ["sparse", "fuller", "plateau", "handoff"][:len(plan["chunks"])]
    t, out = 0.0, []
    for c, ph in zip(plan["chunks"], phases):
        d = c["duration_ms"] / 1000
        if out and out[-1][0] == ph:
            out[-1] = (ph, out[-1][1], t + d)
        else:
            out.append((ph, t, t + d))
        t += d
    return out


def bar_map(grid, sections, tempo, arc, preroll_s):
    """[(ai_bar, bed_bar)] for every bed bar the music covers. AI bar j starts at beat
    downbeat + 4j; bed bar m starts at bed beat 4m."""
    per, t0, db = grid["period_s"], grid["t0"], grid["downbeat"]
    n_ai_bars = (grid["n_beats"] - db) // 4
    ai_bar_t = lambda j: t0 + (db + 4 * j) * per
    bed_bar_at = lambda t: int(round(tempo.beat_at(t) / 4))
    ai_bar_at = lambda t: int(np.clip(round((t - t0) / per / 4 - db / 4), 0, n_ai_bars))
    n_bed_bars = int(tempo.total_beats // 4)
    # AI timeline sections -> bed phases: build sections share the ramp (with the pre-roll in front)
    music_start = arc.phases["ramp"][0] - preroll_s
    bounds_ai = [0.0] + [s[2] for s in sections]                     # AI section edges (s)
    r0, r1 = arc.phases["ramp"]
    bed_edges = [music_start, (r0 + r1) / 2, r1, arc.phases["plateau"][1], arc.length_s]
    pairs = []
    for k in range(len(sections)):
        a0, a1 = ai_bar_at(bounds_ai[k]), ai_bar_at(bounds_ai[k + 1])
        m0, m1 = bed_bar_at(bed_edges[k]), bed_bar_at(bed_edges[k + 1])
        if k == len(sections) - 1:
            a1, m1 = n_ai_bars, n_bed_bars + 1
        nb, na = m1 - m0, a1 - a0
        if nb <= 0 or na <= 0:
            continue
        pick = np.round(np.linspace(a0, a1 - 1, nb)).astype(int) if na != nb else np.arange(a0, a1)
        pairs += [(int(min(j, n_ai_bars - 1)), m0 + i) for i, j in enumerate(pick)]
    return pairs, ai_bar_t


def splice(x, pairs, ai_bar_t, per):
    """Constant-tempo edit of x holding the AI bars in bed order. Returns (y, bar_starts) with
    bar_starts[i] = sample where bed bar pairs[i][1] begins in y."""
    xf = int(SPLICE_XF_S * FS)
    w = np.sin(np.linspace(0, np.pi / 2, xf))[:, None]
    bar_len = 4 * per
    out, starts, pos = [], [], 0
    prev = None
    for ai_bar, _ in pairs:
        s0 = int(round(ai_bar_t(ai_bar) * FS))
        s1 = int(round((ai_bar_t(ai_bar) + bar_len) * FS))
        seg = x[max(0, s0):min(len(x), s1 + xf)]
        if len(seg) < s1 - s0 + xf:
            seg = np.concatenate([seg, np.zeros((s1 - s0 + xf - len(seg), x.shape[1]))])
        starts.append(pos)
        if prev is None:
            out.append(seg[: s1 - s0])
            tail = seg[s1 - s0:]
        elif ai_bar == prev + 1:                       # continuous: the tail IS the next bar's head
            out.append(seg[: s1 - s0])
            tail = seg[s1 - s0:]
        else:                                          # jump: crossfade the old tail into the new head
            head = seg[:xf] * w + tail * w[::-1]
            out.append(np.concatenate([head, seg[xf: s1 - s0]]))
            tail = seg[s1 - s0:]
        pos += s1 - s0
        prev = ai_bar
    return np.concatenate(out), np.asarray(starts)


def time_map(edit, bar_starts, pairs, per, tempo, n_out, max_stretch, guide=None):
    """Anchor every beat of the edit to its bed beat; returns the music stem in bed time.
    guide: the same edit of the AI drum stem. It is stretched together with the music (one
    4-channel stretcher, so both share exact timing) and only used to calibrate the latency:
    soft keys give no sharp envelope to correlate on."""
    ai_in, bed_out = [], []
    for (ai_bar, bed_bar), s in zip(pairs, bar_starts):
        for b in range(4):
            bed_beat = 4 * bed_bar + b
            if bed_beat > tempo.total_beats:
                break
            ai_in.append(s + b * per * FS)
            bed_out.append(float(tempo.time_of_beat(bed_beat)) * FS)
    ai_in, bed_out = np.asarray(ai_in), np.asarray(bed_out)
    keep = bed_out <= n_out
    ai_in, bed_out = ai_in[keep], bed_out[keep]
    if bed_out[-1] < n_out - 1 and ai_in[-1] + 1 < len(edit):  # last partial bar at plateau tempo
        extra = min(len(edit) - 1 - ai_in[-1], n_out - 1 - bed_out[-1])
        ai_in, bed_out = np.append(ai_in, ai_in[-1] + extra), np.append(bed_out, bed_out[-1] + extra)
    ch = edit.shape[1]
    src = edit if guide is None else np.concatenate([edit, guide[:len(edit)]], 1)
    y, info = stretch_to_map(src, ai_in, bed_out, max_stretch=max_stretch,
                             calib_channels=None if guide is None else slice(ch, None))
    y = y[:, :ch]
    full = np.zeros((n_out, ch))
    s0 = int(round(bed_out[0]))
    m = min(len(y), n_out - s0)
    full[s0:s0 + m] = y[:m]
    info["anchors"] = len(ai_in)
    info["bed_span_s"] = [round(bed_out[0] / FS, 3), round(bed_out[-1] / FS, 3)]
    return full, info


def band_gain(x, lo, hi, gain_db, edge_oct=0.25):
    """Static zero-phase gain inside [lo, hi] Hz with raised-cosine shoulders (in octaves)."""
    f = sfft.rfftfreq(len(x), 1 / FS)
    lf = np.log2(np.maximum(f, 1.0))
    d = np.maximum(np.log2(lo) - lf, lf - np.log2(hi))
    w = np.where(d <= 0, 1.0, np.where(d < edge_oct, 0.5 * (1 + np.cos(np.pi * d / edge_oct)), 0.0))
    g = 10 ** (gain_db * w / 20)
    return sfft.irfft(sfft.rfft(x, axis=0) * g[:, None], len(x), axis=0)


def band_handling(x, slots, th):
    am = slots["am_band"]
    y = band_gain(x, am["lo_hz"], am["hi_hz"], th["AI_IN_AM_BAND_GAIN_DB"])
    return band_stop(y, slots["mb_keep_clear_hz"], depth_db=th["MB_NOTCH_MIN_DB"] + 12)


def beat_alignment_ms(music, tempo, t_lo, t_hi):
    """Re-detect attacks on the conformed music near each bed beat in [t_lo, t_hi] (steepest rise
    of the 1 ms energy within +/- 60 ms) and report |offset| median and p95, over the beats with
    a clear attack (top half by strength)."""
    from .analyze import attack_times
    grid = np.asarray([t for t in tempo.grid(1.0) if t_lo <= t <= t_hi])
    if not len(grid):
        return dict(median=None, p95=None, n=0)
    at, sg = attack_times(music, grid, 0.06)
    good = sg > np.percentile(sg, 50)
    o = np.abs(at[good] - grid[good]) * 1000
    return dict(median=round(float(np.median(o)), 1), p95=round(float(np.percentile(o, 95)), 1), n=int(good.sum()))

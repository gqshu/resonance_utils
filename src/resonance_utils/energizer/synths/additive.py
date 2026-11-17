"""Partial-bank renderer: sum of sines, each with its own control-rate amplitude curve.

Phase is absolute (2*pi*f*t + phi from t = 0), so a partial that persists across a chord change
is one continuous sine, never two crossfaded copies that could cancel. Deterministic and chunked."""

import numpy as np

from ...core.audio_io import FS
from ..arc import CR


def smooth_power(step, win_s):
    """Equal-power transitions: sqrt of the Hann-smoothed squared step curve. A partial held across
    a boundary keeps its level; one that starts or stops crosses over in win_s."""
    k = max(3, int(round(win_s * CR)) | 1)
    w = np.hanning(k + 2)[1:-1]
    w /= w.sum()
    return np.sqrt(np.maximum(np.convolve(step ** 2, w, mode="same"), 0.0))


def slow_noise(rng, n, tau_s, depth_db):
    """Smooth random walk at CR (one-pole, both directions), scaled to +/- depth_db, as linear gain."""
    x = rng.standard_normal(n + 4 * int(tau_s * CR))
    a = np.exp(-1.0 / (tau_s * CR))
    from scipy.signal import lfilter
    y = lfilter([1 - a], [1, -a], x)
    y = lfilter([1 - a], [1, -a], y[::-1])[::-1][2 * int(tau_s * CR):][:n]
    y = y / (np.max(np.abs(y)) + 1e-12) * depth_db
    return 10 ** (y / 20)


def render(tracks, n, chunk_s=5.0):
    """tracks: dicts with f (Hz), phase (rad), gl/gr (pan gains), amp (CR curve, len >= n/FS*CR).
    Returns (n, 2) float64."""
    out = np.zeros((n, 2))
    chunk = int(chunk_s * FS)
    spans = []
    for tr in tracks:
        nz = np.flatnonzero(tr["amp"] > 1e-7)
        if len(nz):
            spans.append((int(nz[0] * FS / CR) - FS // CR, int((nz[-1] + 1) * FS / CR) + FS // CR, tr))
    for n0 in range(0, n, chunk):
        n1 = min(n, n0 + chunk)
        idx = np.arange(n0, n1)
        t = idx / FS
        tc = idx * (CR / FS)
        acc = np.zeros(n1 - n0)
        accr = np.zeros(n1 - n0)
        for s0, s1, tr in spans:
            if s1 <= n0 or s0 >= n1:
                continue
            a = np.interp(tc, np.arange(len(tr["amp"])), tr["amp"])
            v = a * np.sin(2 * np.pi * tr["f"] * t + tr["phase"])
            acc += tr["gl"] * v
            accr += tr["gr"] * v
        out[n0:n1, 0] = acc
        out[n0:n1, 1] = accr
    return out

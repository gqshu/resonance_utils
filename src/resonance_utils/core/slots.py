"""Frequency slots shared by every soundscape type: the binaural/monaural carriers the app may
use, and the guards every layer keeps around them.

Every soundscape type is built against the same carrier set, so the app can pick any carrier
in DEFAULT_CARRIERS_HZ for any soundscape. AM needs no slot (it modulates a stem's level)."""

import numpy as np
from scipy import fft as sfft

from .audio_io import FS

DEFAULT_CARRIERS_HZ = (110.0, 220.0, 440.0)
CARRIER_GUARD_HZ = 20.0               # no tonal partial within +/- this of any carrier
BEAT_GUARD_HZ = (0.5, 20.0)           # no two partials this far apart: they beat as AM in the EEG band


def clean_carriers(carriers):
    """Sorted list of the non-zero carriers (0 on the command line means none)."""
    return sorted(c for c in (carriers or []) if c and c > 0)


def near_carrier(f, carriers, guard=CARRIER_GUARD_HZ):
    """Carriers within guard Hz of frequency f."""
    return [c for c in carriers if abs(f - c) < guard]


def notch_gain(f, carrier, depth_db=6.0, bw_oct=1 / 3):
    z = np.log2(np.maximum(f, 1.0) / carrier) / (bw_oct / 2)
    return 10 ** ((-depth_db * np.exp(-0.5 * (z * 1.18) ** 2)) / 20)


def notch(x, carriers, depth_db=6.0):
    """Static circular -6 dB 1/3-octave dip at each binaural carrier the app may use, so its
    tones sit clear whichever one it picks."""
    carriers = [c for c in (carriers or []) if c]
    if not carriers:
        return x
    f = sfft.rfftfreq(len(x), 1 / FS)
    g = np.prod([notch_gain(f, c, depth_db) for c in carriers], axis=0)
    return sfft.irfft(sfft.rfft(x, axis=0) * g[:, None], len(x), axis=0)


def band_stop(x, ranges, depth_db=30.0, edge_hz=8.0):
    """Zero-phase static band-stop (whole-signal FFT): at least depth_db down inside every
    [lo, hi] range, with raised-cosine shoulders edge_hz wide just outside it."""
    if not ranges:
        return x
    f = sfft.rfftfreq(len(x), 1 / FS)
    g = np.ones_like(f)
    floor = 10 ** (-depth_db / 20)
    for lo, hi in ranges:
        d = np.maximum(lo - f, f - hi)                 # <= 0 inside the range
        w = np.where(d <= 0, 1.0, np.where(d < edge_hz, 0.5 * (1 + np.cos(np.pi * d / edge_hz)), 0.0))
        g *= 1 - w * (1 - floor)
    X = sfft.rfft(x, axis=0)
    return sfft.irfft(X * (g[:, None] if x.ndim == 2 else g), len(x), axis=0)

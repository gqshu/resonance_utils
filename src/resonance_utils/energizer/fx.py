"""Effects: deterministic reverb (no modulated tails), brightness blend, stem notch."""

import numpy as np
from scipy import fft as sfft
from scipy.signal import oaconvolve

from ..core.audio_io import FS
from .arc import CR


REVERB_WET = 0.3
REVERB_DRY = 1 - REVERB_WET          # level of the direct path, for partials kept out of the reverb


def reverb(x, seed, decay_s=2.4, wet=REVERB_WET, predelay_s=0.02):
    """LTI convolution with a seeded, decorrelated stereo noise IR (exponential decay, darker tail).
    Deterministic and unmodulated: a steady chord stays steady. (The plan names an FDN; any LTI
    reverb meets the 'no modulated tails' rule, and this one is vectorised.)"""
    rng = np.random.default_rng([seed, 51])
    m = int(decay_s * 1.5 * FS)
    t = np.arange(m) / FS
    irs = []
    for _ in range(2):
        ir = rng.standard_normal(m) * np.exp(-6.9 * t / decay_s)
        f = sfft.rfftfreq(m, 1 / FS)
        ir = sfft.irfft(sfft.rfft(ir) / np.sqrt(1 + (f / 5000) ** 2), m)
        ir = np.concatenate([np.zeros(int(predelay_s * FS)), ir])
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    mono = x.mean(1)
    wetsig = np.stack([oaconvolve(mono, ir)[: len(x)] for ir in irs], 1)
    return x * (1 - wet) + wetsig * wet * np.sqrt(2) * 0.7


def tilt_filter(x, db_per_oct, pivot_hz=500.0):
    f = sfft.rfftfreq(len(x), 1 / FS)
    g = 10 ** (db_per_oct * np.log2(np.maximum(f, pivot_hz) / pivot_hz) / 20)
    return sfft.irfft(sfft.rfft(x, axis=0) * g[:, None], len(x), axis=0)


def brightness_blend(x, tilt_db, dark_db=-4.0):
    """Time-varying brightness without a time-varying filter: crossfade the stem with a
    statically darkened copy. tilt_db is the CR curve (dB/oct, <= 0); dark_db is its minimum."""
    dark = tilt_filter(x, dark_db)
    w = np.clip(1 - tilt_db / dark_db, 0, 1)                    # 1 = bright, 0 = dark
    wn = np.interp(np.arange(len(x)) * (CR / FS), np.arange(len(w)), w)[:, None]
    return wn * x + (1 - wn) * dark


def apply_gain(x, g_cr):
    return x * np.interp(np.arange(len(x)) * (CR / FS), np.arange(len(g_cr)), g_cr)[:, None]

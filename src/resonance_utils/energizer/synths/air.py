"""`procedural_air` texture (plan §6.4): high-shelved pink noise with <= 0.1 Hz level drift."""

import numpy as np
from scipy import fft as sfft

from ...core.audio_io import FS
from . import additive


def render(n, n_cr, seed):
    rng = np.random.default_rng([seed, 41])
    w = rng.standard_normal((n, 2))
    f = sfft.rfftfreq(n, 1 / FS)
    g = 1 / np.sqrt(np.maximum(f, 20.0))                       # pink
    g *= 1 / np.sqrt(1 + (40 / np.maximum(f, 1)) ** 4)          # no sub rumble
    g *= 10 ** ((4 / 20) * (1 / (1 + (4000 / np.maximum(f, 1)) ** 2)))   # +4 dB high shelf
    g /= np.sqrt(1 + (f / 14000) ** 6)
    x = sfft.irfft(sfft.rfft(w, axis=0) * g[:, None], n, axis=0)
    drift = additive.slow_noise(rng, n_cr, 4.0, 1.0)           # time constant 4 s: <= ~0.05 Hz
    return x * np.interp(np.arange(n) * (additive.CR / FS), np.arange(n_cr), drift)[:, None]

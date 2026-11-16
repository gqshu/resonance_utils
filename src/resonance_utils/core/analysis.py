"""Signal-analysis building blocks shared by every soundscape type: block-power envelopes,
band-limited envelope statistics, phase-randomised surrogates, octave balance.

All arrays are 48 kHz float64; stereo is shape (n, 2)."""

import numpy as np
from scipy import fft as sfft
from scipy.signal import resample_poly

from .audio_io import FS

ENV_RATE = 1000                                       # envelope frames/s: 1 ms power blocks
_BLK = FS // ENV_RATE
OCTAVE_EDGES_HZ = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def power_frames(x):
    """Mean power per 1 ms block. Every envelope window here is >= 10 ms, so working on
    blocks instead of samples changes nothing measurable and is ~50x cheaper."""
    p = (x ** 2).mean(1) if x.ndim == 2 else x ** 2
    n = len(p) // _BLK * _BLK
    out = p[:n].reshape(-1, _BLK).mean(1)
    return np.append(out, p[n:].mean()) if n < len(p) else out


def frames_to_samples(g, n):
    """Upsample a smooth per-block curve to n samples (circular linear interpolation)."""
    return np.interp(np.arange(n), (np.arange(len(g)) + 0.5) * _BLK, g, period=n)


def env_db(x, win_s):
    """Centred moving-RMS envelope in dB, at ENV_RATE frames/s."""
    p = power_frames(x)
    k = max(1, int(round(win_s * ENV_RATE)))
    c = np.concatenate([[0.0], np.cumsum(p)])          # O(N) centred moving average
    lo = np.clip(np.arange(len(p)) - k // 2, 0, len(p))
    hi = np.clip(lo + k, 0, len(p))
    e = np.sqrt(np.maximum((c[hi] - c[lo]) / np.maximum(hi - lo, 1), 0)) + 1e-9
    return 20 * np.log10(e)


def band_std(sig, fs, lo, hi):
    """Std of a (dB) envelope after FFT band-pass."""
    s = sig - sig.mean()
    S = sfft.rfft(s * np.hanning(len(s)))
    f = sfft.rfftfreq(len(s), 1 / fs)
    S[(f < lo) | (f > hi)] = 0
    return float(np.std(sfft.irfft(S, len(s))) * np.sqrt(8 / 3))


def surrogate(m, seed=0):
    """Phase-randomised copy: same power spectrum, perfectly stationary. Its envelope
    statistics are the noise floor a clean bed should match, so every check is relative."""
    S = sfft.rfft(m)
    ph = np.exp(2j * np.pi * np.random.default_rng(seed).uniform(0, 1, len(S)))
    ph[0] = 1
    return sfft.irfft(np.abs(S) * ph, len(m))


def env_frames(m, win_s):
    """dB envelope at 100 frames/s, low-passed before decimating: plain slicing would fold
    beats between tonal partials (e.g. 110 Hz) onto false 1-12 Hz 'rhythm' (110 -> 10 Hz)."""
    return resample_poly(env_db(m, win_s), 1, ENV_RATE // 100)


def circ_conv_env(p, k):
    kern = np.zeros(len(p))
    kern[:k] = 1.0 / k
    kern = np.roll(kern, -k // 2)
    return sfft.irfft(sfft.rfft(p) * sfft.rfft(kern), len(p))


def octave_balance(x):
    """Long-term tonal balance: each octave band's share of the total power, in dB."""
    m = x.mean(1)
    f = sfft.rfftfreq(len(m), 1 / FS)
    P = np.abs(sfft.rfft(m)) ** 2
    e = OCTAVE_EDGES_HZ
    v = np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in zip(e[:-1], e[1:])]) + 1e-30
    return 10 * np.log10(v / v.sum())

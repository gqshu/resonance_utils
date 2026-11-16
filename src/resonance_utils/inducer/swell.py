"""Stage 2: bake a slow swell into a flat, looping bed. Offline, and the loop stays exact.

Per wave the bed gets:
  * an asymmetric amplitude swell (rise shorter than fall, like an exhale)
  * a brightness swell (treble dips in the trough, opens at the crest)
  * an optional equal-power left-right sway, one sway per wave

Natural feel without breaking the loop:
  * the wave period drifts a few percent, but the wave count over the loop is exact
  * each wave's depth varies a little (depth is scaled, the envelope is never clipped)

All processing is circular (FFT-domain), so sample n-1 flows into sample 0.
"""

from fractions import Fraction
from math import gcd

import numpy as np
from scipy import fft as sfft

from ..core.audio_io import FS

# Per-rate defaults. Slower = deeper and darker (paces breathing at 6/min);
# faster (0.25 Hz, the rocking rate) = shallower with more sway.
RATE_DEFAULTS = {
    0.10: dict(depth=8.0, tilt=6.0, pan=0.00, rise=0.40),
    0.20: dict(depth=6.0, tilt=5.0, pan=0.15, rise=0.38),
    0.25: dict(depth=5.0, tilt=4.0, pan=0.25, rise=0.35),
}


def rate_params(rate):
    rs = sorted(RATE_DEFAULTS)
    return {k: float(np.interp(rate, rs, [RATE_DEFAULTS[r][k] for r in rs]))
            for k in RATE_DEFAULTS[rs[0]]}


def loop_unit_s(rates):
    """Shortest loop length (s) holding a whole number of waves at every rate,
    and a whole number of samples. 0.1/0.2/0.25 Hz -> 20 s."""
    unit = Fraction(0)
    for r in rates:
        p = 1 / Fraction(str(r)).limit_denominator(100000)
        unit = p if unit == 0 else Fraction(unit.numerator * p.numerator // gcd(unit.numerator, p.numerator),
                                            gcd(unit.denominator, p.denominator))
    k = (unit * FS).denominator
    return float(unit * k)


def wave_shape(phase, rise):
    """Asymmetric raised cosine: 0 at the trough, 1 at the crest."""
    ph = np.mod(phase, 1.0)
    out = np.empty_like(ph)
    up = ph < rise
    out[up] = 0.5 - 0.5 * np.cos(np.pi * ph[up] / rise)
    dn = ~up
    out[dn] = 0.5 + 0.5 * np.cos(np.pi * (ph[dn] - rise) / (1.0 - rise))
    return out


def _slow_noise(rng, n, fc):
    """Zero-mean periodic noise band-limited to fc, peak-normalised to +/-1."""
    f = sfft.rfftfreq(n, 1 / FS)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(1 + (f / fc) ** 8)
    spec[0] = 0
    g = sfft.irfft(spec, n)
    return g / (np.max(np.abs(g)) + 1e-12)


def make_phase(n, rate, jitter, rng):
    """Wave phase (in cycles) with slow period drift; phase[n] would be exactly n_cycles."""
    n_cycles = round(rate * n / FS)
    if jitter <= 0:
        return np.arange(n) * (n_cycles / n), n_cycles
    inst = 1.0 + jitter * _slow_noise(rng, n, rate / 4.0)
    ph = np.concatenate([[0.0], np.cumsum(inst)[:-1]])
    return ph * (n_cycles / inst.sum()), n_cycles


def per_wave(rng, n_cycles, phase, amount):
    """Smooth per-wave multiplier 1 +/- amount, blended across each wave, circular."""
    if amount <= 0:
        return np.ones_like(phase)
    vals = 1.0 + amount * rng.uniform(-1, 1, n_cycles)
    k = np.floor(phase).astype(int) % n_cycles
    w = 0.5 - 0.5 * np.cos(np.pi * (phase - np.floor(phase)))
    return vals[k] * (1 - w) + vals[(k + 1) % n_cycles] * w


def apply_swell(bed, rate, profile, period_jitter=0.03, depth_jitter=0.10, seed=0):
    """bed: (n, 2) flat loop holding a whole number of waves. profile: dict with depth/tilt/pan
    scales and the brightness crossover 'xover' (Hz). Returns (y, info)."""
    n = len(bed)
    cycles = rate * n / FS
    if abs(cycles - round(cycles)) > 1e-6:
        raise ValueError(f"{n / FS:g} s loop is not a whole number of {rate:g} Hz waves")
    p = rate_params(rate)
    depth, tilt, width = p["depth"] * profile["depth"], p["tilt"] * profile["tilt"], p["pan"] * profile["pan"]

    phase, n_cycles = make_phase(n, rate, period_jitter, np.random.default_rng(seed + int(rate * 1000)))
    env = wave_shape(phase, p["rise"])
    d = depth * per_wave(np.random.default_rng(seed + 1), n_cycles, phase, depth_jitter)
    gain_db = -d * (1.0 - env)
    hi_db = -tilt * (1.0 - env)

    f = sfft.rfftfreq(n, 1 / FS)
    hp = 1.0 / np.sqrt(1.0 + (profile["xover"] / np.maximum(f, 1e-3)) ** 4)   # 2nd-order, zero-phase
    high = sfft.irfft(sfft.rfft(bed, axis=0) * hp[:, None], n, axis=0)
    y = (bed - high) + high * (10 ** (hi_db / 20))[:, None]
    y *= (10 ** (gain_db / 20))[:, None]

    pan = width * np.sin(2 * np.pi * phase)
    if width > 0:
        th = (pan + 1) * np.pi / 4
        mid, side = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])
        y = np.stack([np.sqrt(2) * np.cos(th) * mid + side, np.sqrt(2) * np.sin(th) * mid - side], 1)

    info = dict(rate_hz=rate, waves=n_cycles, depth_db=round(depth, 2), tilt_db=round(tilt, 2),
                pan_width=round(width, 3), rise_frac=round(p["rise"], 3), xover_hz=profile["xover"],
                period_jitter=period_jitter, depth_jitter=depth_jitter)
    return y, dict(phase=phase, env=env, gain_db=gain_db, pan=pan, params=info)

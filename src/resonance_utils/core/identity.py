"""Scene identity: does a bed sound like *its* scene, and unlike the others?

A fingerprint has two parts:
  spectrum  octave balance (core.analysis.octave_balance), dB share per octave 63 Hz-16 kHz
  texture   envelope fluctuation (dB std) in three bands (low / mid / high), at two rates:
              grain   2-20 Hz   patter, gurgle, crackle (random: allowed; a regular line is not)
              motion  0.3-2 Hz  gusts, washes, sway
Two scenes that differ in spectrum but not texture sound like two colours of the same noise.
That is what the shipped ElevenLabs scenes measured (2026-11-19): vehicles within 2.5-6.4 dB of
each other, water within 2.1-3.6 dB, and texture within 0.1-0.6 dB everywhere.

    python -m resonance_utils.core.identity a.wav b.wav ...     # table + pairwise distances
"""

import os
import sys

import numpy as np
from scipy import signal

from .analysis import octave_balance
from .audio_io import FS, load_any

TEXTURE_BANDS_HZ = ((40, 250), (250, 2000), (2000, 12000))
RATES = dict(grain=(2.0, 20.0), motion=(0.3, 2.0))
ENV_RATE = 100
AUDIBLE_DB = -30.0          # octaves below this share of the power don't count in distances


def _env_db(m, lo, hi):
    sos = signal.butter(4, [lo, hi], "band", fs=FS, output="sos")
    y = signal.sosfiltfilt(sos, m)
    hop = FS // ENV_RATE
    e = np.sqrt(np.mean(y[: len(y) // hop * hop].reshape(-1, hop) ** 2, 1)) + 1e-9
    return 20 * np.log10(e)


def _band_std(e, lo, hi):
    f, P = signal.welch(e - e.mean(), fs=ENV_RATE, nperseg=min(1024, len(e)))
    k = (f >= lo) & (f < hi)
    return float(np.sqrt(np.sum(P[k]) * (f[1] - f[0])))


def fingerprint(x):
    """x: (n, 2) at FS. Returns dict(spectrum=[8 dB], texture={band: {rate: dB}}, centroid_hz, tonal_db)."""
    m = x.mean(1)
    f, P = signal.welch(m, fs=FS, nperseg=8192)
    k = (f > 40) & (f < 3000)
    lp = 10 * np.log10(P[k] + 1e-20)
    tex = {}
    for lo, hi in TEXTURE_BANDS_HZ:
        e = _env_db(m, lo, hi)
        tex[f"{lo}-{hi}"] = {r: round(_band_std(e, *b), 2) for r, b in RATES.items()}
    return dict(spectrum=[round(float(v), 1) for v in octave_balance(x)],
                texture=tex,
                centroid_hz=round(float(np.sum(f * P) / np.sum(P))),
                tonal_db=round(float(np.max(lp - signal.medfilt(lp, 51))), 1))


def _tex_vec(fp):
    return np.array([v for b in fp["texture"].values() for v in b.values()])


def distance(a, b):
    """(spectral dB rms over audible octaves, texture dB rms)."""
    sa, sb = np.array(a["spectrum"]), np.array(b["spectrum"])
    w = np.maximum(sa, sb) > AUDIBLE_DB
    return (float(np.sqrt(np.mean((sa[w] - sb[w]) ** 2))),
            float(np.sqrt(np.mean((_tex_vec(a) - _tex_vec(b)) ** 2))))


def report(named):
    """named: [(name, fingerprint)]. Returns printable lines."""
    out = ["spectrum dB (63 125 250 500 1k 2k 4k 8k) | centroid | tonal | grain/motion dB low, mid, high"]
    for n, d in named:
        t = "  ".join(f"{v['grain']:.1f}/{v['motion']:.1f}" for v in d["texture"].values())
        out.append(f"{n[:22]:22s} " + " ".join(f"{v:4.0f}" for v in d["spectrum"][:8])
                   + f" | {d['centroid_hz']:6.0f} | {d['tonal_db']:4.1f} | {t}")
    out.append("\npairwise distance: spectral dB | texture dB")
    out.append(" " * 22 + "".join(f"{n[:11]:>13s}" for n, _ in named))
    for i, (n, a) in enumerate(named):
        cells = []
        for j, (_, b) in enumerate(named):
            s, t = distance(a, b)
            cells.append(f"{'—':>13s}" if i == j else f"{s:6.1f}|{t:<5.2f} ")
        out.append(f"{n[:22]:22s}" + "".join(cells))
    return out


def main(argv=None):
    paths = argv if argv is not None else sys.argv[1:]
    named = [(os.path.splitext(os.path.basename(p))[0], fingerprint(load_any(p))) for p in paths]
    print("\n".join(report(named)))


if __name__ == "__main__":
    main()

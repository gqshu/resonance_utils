"""M1 QA: the plan §9 measures the all-procedural path can already be checked. Thresholds are
provisional [H] (thresholds.yaml); the full gate with its failure suite is milestone M3."""

import numpy as np
from scipy import fft as sfft
from scipy.signal import resample_poly

from ..core.analysis import band_std, power_frames, ENV_RATE
from ..core.audio_io import FS


def band_limit(x, lo, hi):
    m = x.mean(1) if x.ndim == 2 else x
    f = sfft.rfftfreq(len(m), 1 / FS)
    return sfft.irfft(sfft.rfft(m) * ((f >= lo) & (f <= hi)), len(m))


def native_mod_db(x, band, mod_bands):
    """Envelope modulation depth (dB re 100 %) of x inside `band`, in each modulation band:
    the amplitude envelope (1 ms blocks), band-passed to the modulation band, as a fraction of
    its mean. A sine AM of depth m reads 20*log10(m)."""
    y = band_limit(x, *band) if band else (x.mean(1) if x.ndim == 2 else x)
    env = np.sqrt(power_frames(y))
    edge = ENV_RATE                                    # skip 1 s at each end (fades)
    env = env[edge:-edge]
    out = {}
    for lo, hi in mod_bands:
        d = band_std(env, ENV_RATE, lo, hi) * np.sqrt(2) / (env.mean() + 1e-12)
        out[f"{lo:g}-{hi:g}"] = round(float(20 * np.log10(d + 1e-9)), 1)
    return out


def slot_residual_db(x, ranges, flank_hz=60.0):
    """Mean power density inside each keep-clear range relative to its two flanks (dB)."""
    m = x.mean(1) if x.ndim == 2 else x
    nfft = 1 << 15
    segs = [m[i:i + nfft] * np.hanning(nfft) for i in range(0, len(m) - nfft, nfft)]
    P = np.mean([np.abs(sfft.rfft(s)) ** 2 for s in segs], 0)
    f = sfft.rfftfreq(nfft, 1 / FS)
    worst = -np.inf
    for lo, hi in ranges:
        inside = P[(f >= lo) & (f <= hi)].mean()
        fl = P[((f >= lo - flank_hz) & (f < lo - 5)) | ((f > hi + 5) & (f <= hi + flank_hz))].mean()
        worst = max(worst, 10 * np.log10(inside / (fl + 1e-30) + 1e-30))
    return round(float(worst), 1)


def true_peak_dbfs(x, chunk_s=10):
    pk = 0.0
    step = int(chunk_s * FS)
    for i in range(0, len(x), step):
        seg = x[max(0, i - 64): i + step + 64]
        pk = max(pk, float(np.max(np.abs(resample_poly(seg, 4, 1, axis=0)))))
    return round(20 * np.log10(pk + 1e-12), 2)


def rms_db(x):
    return round(float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)), 2)


def grid_error_ms(onsets, tempo):
    if not onsets:
        return 0.0
    err = [abs(o["sample"] / FS - float(tempo.time_of_beat(o["beat"]))) for o in onsets]
    return round(1000 * max(err), 4)


def run(stems, mix, tempo, arc, slots, onsets, th):
    nb = slots["native_mod_bands_hz"]
    am = (slots["am_band"]["lo_hz"], slots["am_band"]["hi_hz"])
    clear = slots["mb_keep_clear_hz"]
    phases = {k: (int(a * FS), int(b * FS)) for k, (a, b) in arc.phases.items()}
    q = dict(
        provisional=True,
        native_mod_db=dict(pad_am_band=native_mod_db(stems["pad"], am, nb),
                           mix=native_mod_db(mix, None, nb),
                           **{k: native_mod_db(v, None, nb) for k, v in stems.items() if k != "pad"}),
        mb_slot_residual_db=dict(mix=slot_residual_db(mix, clear),
                                 **{k: slot_residual_db(v, clear) for k, v in stems.items()}),
        grid_error_ms={k: grid_error_ms(v, tempo) for k, v in onsets.items()},
        true_peak_dbfs=true_peak_dbfs(mix),
        phase_rms_dbfs={k: rms_db(mix[a:b]) for k, (a, b) in phases.items()},
    )
    mb_ok = q["mb_slot_residual_db"]["mix"] <= -th["MB_NOTCH_MIN_DB"]
    am_ok = max(q["native_mod_db"]["pad_am_band"].values()) <= th["PAD_NATIVE_MOD_MAX_DB"]
    grid_ok = max(q["grid_error_ms"].values(), default=0) <= 1.0
    q["checks"] = dict(mb_slot=bool(mb_ok), pad_native_mod=bool(am_ok), grid=bool(grid_ok),
                       true_peak=bool(q["true_peak_dbfs"] <= -1.0))
    q["tags"] = dict(am_compatible=bool(am_ok), mb_compatible=bool(mb_ok), bb_compatible=bool(mb_ok),
                     lever_compatible=bool(am_ok and mb_ok and grid_ok))
    return q

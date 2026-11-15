#!/usr/bin/env python3
"""
ocean_gen.py  (v2) — "Ocean Swell" / "Ocean Soundscape" generator for an
EEG-adaptive binaural-beat / AM nap product.

Two ways to get the texture:
  1. PROCEDURAL (default)  — shaped noise + realism layers (bubble/foam grains,
     distant second wave, small room/air reverb, natural period drift).
  2. RECORDING  (--source calm_surf.wav) — "envelope transplant": take a real
     ocean field recording, flatten its own wave rhythm and crashes, then
     re-impose OUR rhythm. Keeps the real grain of water, removes its timing.
     This is the most realistic option.

Rhythm rules (both paths):
  * Loop = integer number of waves for every rate (120 s -> 12/24/30 waves at
    0.1/0.2/0.25 Hz); output loops sample-perfectly.
  * Same seed for every rate -> identical texture, only timing differs, so the
    app can switch rates at a trough almost invisibly.
  * Brightness sweeps with each wave (dark trough -> bright crest).
  * Asymmetric smooth envelope, attack >= 1 s, bounded depth (no crashes).
  * Optional natural period drift (--period-jitter, default 4 %) that keeps the
    AVERAGE rate and total wave count exact.
  * Nothing is added that flutters at 1-12 Hz (EEG band). Bubble grains are
    dense (hundreds/s), so they read as fizz, not as a rhythm.
  * -6 dB 1/3-octave notch at the binaural carrier (default 144 Hz).
  * Sidecar JSON (params + QA) and CSV (envelope @ 20 Hz) for EEG logging.

Requires numpy + scipy (scipy only for resampling a --source file).
Writes 48 kHz / 24-bit stereo WAV.

Examples
  python3 ocean_gen.py                                   # 6 procedural tracks
  python3 ocean_gen.py --source surf.wav --mode soundscape
  python3 ocean_gen.py --mode swell --rates 0.25 --period-jitter 0
  python3 ocean_gen.py --carrier 250 --seed 7 --out build/audio --stems
"""

import argparse
import json
import os
import wave

import numpy as np

FS = 48000

PRESETS = {
    "swell": {
        "depth_db": {0.1: 8.0, 0.2: 6.0, 0.25: 5.0},
        "rise_frac": 0.40,
        "amp_jitter": 0.10,
        "dark_lp": 700.0,
        "bright_lp": 2200.0,
        "slope_db_oct": -4.5,
        "hp": 60.0,
        "stereo_corr": 0.55,
        "pan_width": 0.0,
        "layers": {"rumble": -14.0, "wash": 0.0, "foam": None, "bubbles": None,
                   "distant": -16.0, "bed": -18.0, "reverb": -16.0},
    },
    "soundscape": {
        "depth_db": {0.1: 10.0, 0.2: 8.0, 0.25: 6.0},
        "rise_frac": 0.35,
        "amp_jitter": 0.15,
        "dark_lp": 500.0,
        "bright_lp": 3800.0,
        "slope_db_oct": -4.0,
        "hp": 40.0,
        "stereo_corr": 0.35,
        "pan_width": 0.20,
        "layers": {"rumble": -10.0, "wash": 0.0, "foam": -11.0, "bubbles": -15.0,
                   "distant": -13.0, "bed": -16.0, "reverb": -12.0},
    },
}

TARGET_RMS_DBFS = -26.0
PEAK_CEIL_DBFS = -3.0


# --------------------------------------------------------------------------
# Spectral helpers (rfft domain -> circular, loopable)
# --------------------------------------------------------------------------
def freqs(n):
    return np.fft.rfftfreq(n, 1.0 / FS)


def slope_gain(f, slope_db_oct, f_ref=1000.0):
    return 10 ** ((slope_db_oct * np.log2(np.maximum(f, 1.0) / f_ref)) / 20.0)


def butter_mag(f, fc, order, kind):
    f = np.maximum(f, 1e-3)
    r = (f / fc) if kind == "lp" else (fc / f)
    return 1.0 / np.sqrt(1.0 + r ** (2 * order))


def notch_gain(f, fc, depth_db=6.0, bw_oct=1.0 / 3.0):
    if not fc:
        return np.ones_like(f)
    x = np.log2(np.maximum(f, 1.0) / fc) / (bw_oct / 2.0)
    return 10 ** ((-depth_db * np.exp(-0.5 * (x * 1.18) ** 2)) / 20.0)


def shaped_noise(rng, n, gain):
    spec = (rng.standard_normal(len(gain)) + 1j * rng.standard_normal(len(gain))) * gain
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-12)


def stereo_noise(rng, n, gain, corr):
    c = shaped_noise(rng, n, gain)
    a, b = np.sqrt(corr), np.sqrt(1.0 - corr)
    return a * c + b * shaped_noise(rng, n, gain), a * c + b * shaped_noise(rng, n, gain)


def filt(x, g):
    """Zero-phase circular filter of a (n,) or (n,2) signal by rfft gain g."""
    if x.ndim == 1:
        return np.fft.irfft(np.fft.rfft(x) * g, len(x))
    return np.stack([filt(x[:, c], g) for c in range(x.shape[1])], 1)


def smooth_circ(x, win):
    """Circular moving average (loop-safe)."""
    k = np.zeros(len(x))
    k[:win] = 1.0 / win
    k = np.roll(k, -win // 2)
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(k), len(x))


# --------------------------------------------------------------------------
# Timing: phase with optional natural drift, envelope, per-wave size jitter
# --------------------------------------------------------------------------
def make_phase(rng, n, rate, period_jitter):
    """Instantaneous rate = rate*(1 + j*g(t)), g zero-mean, periodic, slow.
    Integral over the loop is unchanged -> exact wave count, loop-safe."""
    t = np.arange(n) / FS
    if period_jitter <= 0:
        return rate * t
    f = freqs(n)
    g = shaped_noise(rng, n, butter_mag(f, rate / 4.0, 4, "lp"))
    g -= g.mean()
    g /= (np.max(np.abs(g)) + 1e-12)
    inst = rate * (1.0 + period_jitter * g)
    phase = np.cumsum(inst) / FS
    phase -= phase[0]
    return phase * (rate * n / FS) / (phase[-1] + inst[-1] / FS)   # exact cycle count


def wave_shape(phase, rise):
    phase = np.mod(phase, 1.0)
    out = np.empty_like(phase)
    up = phase < rise
    out[up] = 0.5 - 0.5 * np.cos(np.pi * phase[up] / rise)
    dn = ~up
    out[dn] = 0.5 + 0.5 * np.cos(np.pi * (phase[dn] - rise) / (1.0 - rise))
    return out


def cycle_jitter(rng, n_cycles, phase, amount):
    if amount <= 0:
        return np.ones_like(phase)
    vals = 1.0 + amount * rng.uniform(-1, 1, n_cycles)
    k = np.floor(phase).astype(int) % n_cycles
    w = 0.5 - 0.5 * np.cos(np.pi * (phase - np.floor(phase)))
    return vals[k] * (1 - w) + vals[(k + 1) % n_cycles] * w


# --------------------------------------------------------------------------
# Realism layers
# --------------------------------------------------------------------------
def bubble_grains(rng, n, density_env, rate_per_s=400.0):
    """Foam = many tiny bubbles. Each grain: short decaying sine whose pitch
    rises slightly (bubble resonance), 1.5-8 kHz, 3-15 ms. Grain times are
    drawn in proportion to density_env, so fizz follows the wave. Hundreds of
    grains/s -> perceived as texture, no 1-12 Hz rhythm. Wraps circularly."""
    out = np.zeros((n, 2))
    total = int(rate_per_s * n / FS * density_env.mean())
    if total == 0:
        return out
    cdf = np.cumsum(density_env)
    cdf /= cdf[-1]
    pos = np.searchsorted(cdf, rng.uniform(0, 1, total))
    for p in pos:
        f0 = np.exp(rng.uniform(np.log(1500), np.log(8000)))
        dur = rng.uniform(0.003, 0.015)
        m = int(dur * FS * 4)
        tt = np.arange(m) / FS
        chirp = f0 * (1.0 + 0.15 * tt / dur)
        g = np.sin(2 * np.pi * np.cumsum(chirp) / FS) * np.exp(-tt / dur)
        g *= np.minimum(1.0, tt / 0.0005)                     # 0.5 ms fade-in, no click
        amp = np.exp(rng.normal(0, 0.6))
        pan = rng.uniform(0.15, 0.85)
        idx = (p + np.arange(m)) % n
        out[idx, 0] += amp * np.sqrt(1 - pan) * g
        out[idx, 1] += amp * np.sqrt(pan) * g
    return out / (np.std(out) + 1e-12)


def air_reverb(rng, x, seconds=0.9, predelay=0.012):
    """Open-air diffuse tail: decorrelated decaying noise IR, applied by
    circular convolution (loop-safe). Gives 'space' instead of 'in-ear hiss'."""
    n = len(x)
    m = int(seconds * FS)
    tt = np.arange(m) / FS
    out = np.zeros_like(x)
    for c in range(2):
        ir = np.zeros(n)
        d = int(predelay * FS) + c * int(0.007 * FS)
        tail = rng.standard_normal(m) * np.exp(-tt * 6.9 / seconds)
        tail = filt(tail, butter_mag(freqs(m), 3000, 2, "lp"))   # air absorbs highs
        ir[d:d + m] = tail[: max(0, min(m, n - d))]
        out[:, c] = np.fft.irfft(np.fft.rfft(x[:, c]) * np.fft.rfft(ir), n)
    return out / (np.std(out) + 1e-12)


# --------------------------------------------------------------------------
# Recording path: "envelope transplant"
# --------------------------------------------------------------------------
def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, fs, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        i = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(i >= 1 << 23, i - (1 << 24), i) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 2 ** 31
    else:
        raise ValueError("unsupported WAV sample width")
    x = x.reshape(-1, ch)
    if ch == 1:
        x = np.repeat(x, 2, 1)
    x = x[:, :2]
    if fs != FS:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g, axis=0)
    return x


def prepare_recording(path, n):
    """Return an (n,2) loop-safe, rhythm-free texture from a field recording."""
    x = read_wav(path)
    x = x - x.mean(0)
    need = n + 2 * FS
    if len(x) < need:                      # tile with crossfades if too short
        reps = int(np.ceil(need / len(x))) + 1
        x = np.concatenate([x] * reps, 0)
    x = x[:need]

    # 1) Flatten its own slow wave rhythm (0.5 s RMS envelope)
    p = (x ** 2).mean(1)
    slow = np.sqrt(np.convolve(p, np.ones(FS // 2) / (FS // 2), "same")) + 1e-6
    # 2) Tame crashes: fast (20 ms) envelope may exceed slow by at most +3 dB
    fast = np.sqrt(np.convolve(p, np.ones(FS // 50) / (FS // 50), "same")) + 1e-6
    limit = np.minimum(1.0, (slow * 10 ** (3 / 20)) / fast)
    limit = np.convolve(limit, np.ones(FS // 100) / (FS // 100), "same")   # smooth gain
    flat = x * (limit / slow)[:, None]

    # 3) Loop: crossfade the extra 2 s tail into the head (equal power)
    xf = 2 * FS
    body = flat[:n].copy()
    tail = flat[n:n + xf]
    w = np.linspace(0, np.pi / 2, xf)[:, None]
    body[:xf] = body[:xf] * np.sin(w) + tail * np.cos(w)
    return body / (np.std(body) + 1e-12)


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def render(mode, rate, duration, carrier, seed, period_jitter, source=None, want_stems=False):
    p = PRESETS[mode]
    n = int(round(duration * FS))
    cycles = rate * duration
    if abs(cycles - round(cycles)) > 1e-9:
        raise ValueError(f"duration {duration}s is not an integer number of {rate} Hz "
                         f"waves; use a multiple of {1 / rate:g}s (120 s fits all three).")
    n_cycles = int(round(cycles))
    f = freqs(n)
    notch = notch_gain(f, carrier)
    hp = butter_mag(f, p["hp"], 2, "hp")
    rng = np.random.default_rng(seed)       # texture rng: identical for every rate
    db = lambda v: 10 ** (v / 20.0)

    # ---------------- texture: wash (dark & bright of the SAME material) ----
    if source:
        base = prepare_recording(source, n)
        colour = hp * notch                                   # keep the recording's own colour
    else:
        base = np.stack(stereo_noise(rng, n, np.ones_like(f), p["stereo_corr"]), 1)
        colour = slope_gain(f, p["slope_db_oct"]) * hp * notch
    dark = filt(base, colour * butter_mag(f, p["dark_lp"], 2, "lp"))
    bright = filt(base, colour * butter_mag(f, p["bright_lp"], 2, "lp"))
    s = np.std(bright)
    dark, bright = dark / s, bright / s

    trng = np.random.default_rng(seed + 7)  # extra procedural layers (same across rates)
    rum = np.stack(stereo_noise(trng, n, slope_gain(f, -6.0) * butter_mag(f, 150, 2, "lp")
                                * butter_mag(f, 30, 2, "hp") * notch, 0.8), 1)
    bed = np.stack(stereo_noise(trng, n, slope_gain(f, -5.0) * butter_mag(f, 1200, 2, "lp")
                                * hp * notch, 0.3), 1)
    foam = np.stack(stereo_noise(trng, n, butter_mag(f, 2500, 2, "hp") * butter_mag(f, 9000, 2, "lp")
                                 * slope_gain(f, -3.0), 0.15), 1)
    distant = filt(np.stack(stereo_noise(trng, n, np.ones_like(f), 0.6), 1),
                   slope_gain(f, -5.0) * butter_mag(f, 600, 2, "lp") * hp * notch)
    distant /= np.std(distant)

    # ---------------- timing -------------------------------------------------
    phase = make_phase(np.random.default_rng(seed + 2000 + int(rate * 1000)), n, rate, period_jitter)
    env = np.clip(wave_shape(phase, p["rise_frac"]) *
                  cycle_jitter(np.random.default_rng(seed + 1000), n_cycles, phase, p["amp_jitter"]), 0, 1)
    depth = p["depth_db"].get(rate, 6.0)
    gain = 10 ** (-depth * (1.0 - env) / 20.0)
    brt = (env ** 1.5)[:, None]
    g = gain[:, None]
    lay = p["layers"]

    mix = g * ((1 - brt) * dark + brt * bright) * db(lay["wash"])
    mix += (10 ** (-2.0 * (1.0 - env) / 20.0))[:, None] * rum * db(lay["rumble"])
    mix += bed * db(lay["bed"])
    # distant second line of waves: same rate, half a wave later, dark & quiet
    denv = wave_shape(phase + 0.5, p["rise_frac"])
    mix += (10 ** (-4.0 * (1.0 - denv) / 20.0))[:, None] * distant * db(lay["distant"])
    foam_env = wave_shape(phase - 0.12, p["rise_frac"]) ** 2         # after the crest
    if lay["foam"] is not None:
        mix += foam_env[:, None] * foam * db(lay["foam"])
    if lay["bubbles"] is not None and not source:   # real recordings already have bubbles
        bub = bubble_grains(np.random.default_rng(seed + 3000), n, 0.05 + foam_env)
        mix += bub * db(lay["bubbles"])

    if lay["reverb"] is not None:
        mix += air_reverb(np.random.default_rng(seed + 4000), mix) * db(lay["reverb"])

    if p["pan_width"] > 0:
        pan = p["pan_width"] * np.sin(2 * np.pi * phase)
        th = (pan + 1) * np.pi / 4
        mono, side = 0.5 * (mix[:, 0] + mix[:, 1]), 0.5 * (mix[:, 0] - mix[:, 1])
        mix = np.stack([np.sqrt(2) * np.cos(th) * mono + side,
                        np.sqrt(2) * np.sin(th) * mono - side], 1)

    mix *= 10 ** (TARGET_RMS_DBFS / 20.0) / np.sqrt(np.mean(mix ** 2))
    peak, ceil = np.max(np.abs(mix)), 10 ** (PEAK_CEIL_DBFS / 20.0)
    if peak > ceil:
        mix *= ceil / peak

    stems = None
    if want_stems:
        k = 10 ** (TARGET_RMS_DBFS / 20.0)
        stems = {"dark": dark * k, "bright": bright * k}
    return mix, env, gain, depth, phase, stems


# --------------------------------------------------------------------------
# Output + QA
# --------------------------------------------------------------------------
def write_wav24(path, x):
    x = np.clip(x, -1.0, 1.0 - 2 ** -23)
    i = (x * (2 ** 23)).astype("<i4")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(3)
        w.setframerate(FS)
        w.writeframes(i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())


def qa_report(x, carrier):
    mono = x.mean(1)
    seam = float(np.max(np.abs(x[0] - x[-1])))
    step = float(np.median(np.abs(np.diff(x[:, 0]))))
    res = {"loop_seam_jump": round(seam, 6), "typical_sample_step": round(step, 6)}
    if carrier:
        spec = np.abs(np.fft.rfft(mono)) ** 2
        f = freqs(len(mono))
        band = (f > carrier * 2 ** (-1 / 6)) & (f < carrier * 2 ** (1 / 6))
        near = (f > carrier * 2 ** (-1 / 2)) & (f < carrier * 2 ** (1 / 2)) & ~band
        res["carrier_band_vs_neighbours_db"] = round(float(10 * np.log10(spec[band].mean() / spec[near].mean())), 2)
    hop = FS // 100
    e = np.sqrt(np.convolve(mono ** 2, np.ones(hop) / hop, "same")[::hop])
    E = np.abs(np.fft.rfft(e - e.mean())) ** 2
    fe = np.fft.rfftfreq(len(e), 1 / 100)
    res["envelope_mod_energy_1_12Hz_share"] = round(float(E[(fe >= 1) & (fe <= 12)].sum() / E[fe > 0.02].sum()), 4)
    res["peak_dbfs"] = round(float(20 * np.log10(np.max(np.abs(x)))), 2)
    res["rms_dbfs"] = round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)))), 2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["swell", "soundscape", "both"], default="both")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.25])
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--carrier", type=float, default=144.0, help="BB carrier to notch (Hz); 0 = off")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--period-jitter", type=float, default=0.04,
                    help="natural wave-period drift (fraction); average rate stays exact; 0 = metronomic")
    ap.add_argument("--source", default=None, help="real ocean recording (WAV) for envelope transplant")
    ap.add_argument("--out", default="out")
    ap.add_argument("--stems", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    modes = ["swell", "soundscape"] if a.mode == "both" else [a.mode]
    carrier = a.carrier if a.carrier > 0 else None
    src_tag = "rec" if a.source else "proc"

    for mode in modes:
        for rate in a.rates:
            x, env, gain, depth, phase, stems = render(mode, rate, a.duration, carrier, a.seed,
                                                       a.period_jitter, a.source, a.stems)
            tag = f"ocean_{mode}_{src_tag}_{str(rate).replace('.', 'p')}Hz"
            write_wav24(os.path.join(a.out, tag + ".wav"), x)
            step = FS // 20
            with open(os.path.join(a.out, tag + "_envelope.csv"), "w") as fh:
                fh.write("time_s,wave_phase,envelope_0to1,gain_db\n")
                for i in range(0, len(env), step):
                    fh.write(f"{i / FS:.3f},{phase[i]:.4f},{env[i]:.4f},{20 * np.log10(gain[i]):.3f}\n")
            qa = qa_report(x, carrier)
            meta = {"mode": mode, "texture": src_tag, "source": a.source, "rate_hz": rate,
                    "period_s": 1 / rate, "duration_s": a.duration, "waves": int(round(rate * a.duration)),
                    "period_jitter": a.period_jitter, "depth_db": depth, "carrier_notch_hz": carrier,
                    "seed": a.seed, "fs": FS, "bits": 24, "preset": PRESETS[mode], "qa": qa}
            with open(os.path.join(a.out, tag + ".json"), "w") as fh:
                json.dump(meta, fh, indent=2, default=str)
            if stems:
                for k, v in stems.items():
                    write_wav24(os.path.join(a.out, f"ocean_{mode}_{src_tag}_stem_{k}.wav"), np.clip(v, -1, 1))
            print(f"{tag}: {qa}")


if __name__ == "__main__":
    main()

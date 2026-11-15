#!/usr/bin/env python3
"""
elevenlabs_bed_builder.py — end-to-end builder for flat, swell-free ambience
beds (train, sleeper, cabin, ...) from ElevenLabs Sound Effects, ready to be
driven by swell_engine.py.

Pipeline
  1. GENERATE  N takes per prompt via the ElevenLabs Sound Effects API
               (30 s, loop=True). API key from $ELEVENLABS_API_KEY.
  2. SCREEN    each take, reject or flag:
                 - rhythm in the EEG band (1-12 Hz envelope modulation:
                   rail clacks, speech syllables, flutter)
                 - built-in swell (0.05-0.3 Hz envelope modulation)
                 - tonal hum near the binaural carrier (default 144 Hz)
                 - transients (doors, horns, bangs) and clipping/silence
  3. JOIN      passing takes, shuffled, equal-power crossfades, and a
               circular crossfade at the end -> seamless loop.
  4. CONDITION flatten any residual slow rhythm, cap transients (+3 dB),
               notch the carrier band -6 dB, normalise (-26 dBFS RMS, <= -3 dBFS peak).
  5. EXPORT    48 kHz / 24-bit stereo WAV + JSON report (+ every raw take
               and its screening numbers, so you can audit decisions).

All screening thresholds are config constants (THRESH) — tune them from data.

Requirements: Python 3.9+, numpy, scipy. ffmpeg only if the API returns MP3
(fallback when PCM output isn't available on your plan).

Usage
  export ELEVENLABS_API_KEY=sk_...
  bed-builder --preset train_smooth --takes 10 --out build/train
  bed-builder --prompt "Inside a quiet cable car ..." --takes 8
  bed-builder --from-dir my_downloads/ --out build/manual   # skip API
  bed-builder --preset airplane --dry-run                  # show requests

Env
  ELEVENLABS_API_KEY   required unless --from-dir / --dry-run
  ELEVENLABS_API_BASE  optional, default https://api.elevenlabs.io (for testing/proxies)
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly, lfilter

FS = 48000
FLAT_LINE = ("Constant level throughout, no swelling, no surges, no fades, no build-ups.")

PRESETS = {
    "train_smooth": (
        "Interior of a modern high-speed electric train cruising at steady speed at night, "
        "recorded from a passenger seat. Deep continuous low rumble, soft broadband air hiss, "
        "faint smooth wheel-on-rail hum, gentle cabin resonance. {flat} No rhythmic clicking. "
        "Quiet empty carriage, no voices, no announcements, no doors, no horn, no brakes."),
    "train_sleeper": (
        "Inside a sleeper compartment of an old night train at moderate speed. Warm low rumble, "
        "soft muffled clickety-clack of wheels over rail joints in a slow regular rhythm, distant "
        "and gentle. Cozy, enclosed, calm. {flat} No voices, no announcements, no horn, "
        "no squealing brakes."),
    "airplane": (
        "Airplane cabin interior during smooth cruise at night, steady deep engine drone and "
        "broadband air noise, muffled and even. {flat} No voices, no announcements, no chimes, "
        "no seatbelt sign dings."),
    "car_night": (
        "Inside a quiet car driving on a smooth highway at night, recorded from the back seat. "
        "Soft tire roar on asphalt, low engine hum, gentle wind around the windows. {flat} "
        "No music, no voices, no indicators, no horn, no passing trucks."),
}

# --- screening thresholds (empirical-first; tune with your own data) -------
THRESH = dict(
    eeg_band=(1.0, 12.0),        # Hz, envelope modulation band to keep clean
    # all modulation checks are EXCESS over a phase-randomised (stationary)
    # surrogate of the same take, so random noise fluctuation scores ~0
    eeg_band_excess_db=0.3,      # extra log-envelope std in 1-12 Hz (dB)
    eeg_peak_excess_db=6.0,      # extra sharpness of the strongest 1-12 Hz line (dB)
    swell_band=(0.05, 0.3),      # Hz
    swell_flag_db=0.75,          # excess > this: flag, conditioning will flatten it
    swell_reject_db=3.0,         # excess > this: reject (too much built-in rhythm)
    hum_window_hz=20.0,          # look for tonal lines within carrier +/- this
    hum_reject_db=10.0,          # line prominence over local spectral median (dB)
    transient_crest_db=6.0,      # 20 ms over 0.5 s envelope
    transient_max_per_min=6.0,   # reject if this many MORE events/min than surrogate
    min_rms_dbfs=-60.0,          # reject near-silence
    clip_frac_max=1e-4,          # reject if more samples than this are at full scale
)
XFADE_S = 3.0
TARGET_RMS_DBFS = -26.0
PEAK_CEIL_DBFS = -3.0


# ===========================================================================
# 1. GENERATE
# ===========================================================================
def api_generate(text, duration, influence, loop, model, output_format, api_key, base):
    url = f"{base.rstrip('/')}/v1/sound-generation?output_format={output_format}"
    body = {"text": text, "duration_seconds": duration, "prompt_influence": influence,
            "loop": loop, "model_id": model}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"xi-api-key": api_key,
                                          "Content-Type": "application/json",
                                          "Accept": "audio/*"})
    delay = 4.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 429 or e.code >= 500:
                print(f"    HTTP {e.code}, retrying in {delay:.0f}s ...")
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            print(f"    network error {e.reason}, retrying in {delay:.0f}s ...")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("API failed after retries")


def decode_audio(blob, output_format):
    """Return (n,2) float64 at FS."""
    if output_format.startswith("pcm_"):
        sr = int(output_format.split("_")[1])
        x = np.frombuffer(blob, "<i2").astype(np.float64) / 32768.0   # mono s16le
        x = x[:, None]
    else:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("API returned compressed audio and ffmpeg is not installed")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(blob)
            src = f.name
        try:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-f", "f32le",
                                  "-ac", "2", "-ar", str(FS), "-"],
                                 check=True, capture_output=True).stdout
        finally:
            os.unlink(src)
        return np.frombuffer(raw, "<f4").astype(np.float64).reshape(-1, 2)
    if sr != FS:
        g = gcd(FS, sr)
        x = resample_poly(x, FS // g, sr // g, axis=0)
    return x


def pseudo_stereo(mono, seed=0):
    """Decorrelate a mono take into a natural stereo pair: two different short
    diffuse IRs (~25 ms) blended with the dry signal. No comb/phasey Haas tricks."""
    rng = np.random.default_rng(seed)
    m = int(0.025 * FS)
    t = np.arange(m) / FS
    out = []
    for _ in range(2):
        ir = rng.standard_normal(m) * np.exp(-t / 0.006)
        ir /= np.sqrt(np.sum(ir ** 2))
        wet = lfilter(ir, [1.0], mono)
        out.append(0.7 * mono + 0.714 * wet)
    return np.stack(out, 1)


def generate_takes(args, prompt, raw_dir):
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    base = os.environ.get("ELEVENLABS_API_BASE", "https://api.elevenlabs.io")
    if not api_key and not args.dry_run:
        sys.exit("ELEVENLABS_API_KEY is not set (or use --from-dir / --dry-run)")
    paths = []
    fmt = args.output_format
    for i in range(args.takes):
        # vary prompt influence a little across takes -> more natural variety
        infl = float(np.clip(args.influence + (i % 3 - 1) * 0.05, 0.0, 1.0))
        print(f"  take {i + 1}/{args.takes}  influence={infl:.2f}  format={fmt}")
        if args.dry_run:
            print(f"    POST /v1/sound-generation  text={prompt[:70]!r}...  "
                  f"duration={args.duration} loop=True model={args.model}")
            continue
        try:
            blob = api_generate(prompt, args.duration, infl, True, args.model, fmt, api_key, base)
        except RuntimeError as e:
            if fmt.startswith("pcm_") and ("40" in str(e)[:9] or "42" in str(e)[:9]):
                print(f"    PCM not available ({e}); falling back to mp3_44100_192")
                fmt = "mp3_44100_192"
                blob = api_generate(prompt, args.duration, infl, True, args.model, fmt, api_key, base)
            else:
                raise
        x = decode_audio(blob, fmt)
        if x.shape[1] == 1:
            x = pseudo_stereo(x[:, 0], seed=i)
        p = os.path.join(raw_dir, f"take_{i + 1:02d}.wav")
        write_wav24(p, normalise(x))
        with open(p[:-4] + ".json", "w") as fh:
            json.dump({"prompt": prompt, "influence": infl, "duration": args.duration,
                       "model": args.model, "format": fmt}, fh, indent=2)
        paths.append(p)
    return paths


# ===========================================================================
# 2. SCREEN
# ===========================================================================
def env_db(x, win_s):
    p = (x ** 2).mean(1) if x.ndim == 2 else x ** 2
    k = max(1, int(win_s * FS))
    c = np.concatenate([[0.0], np.cumsum(p)])          # O(N) centred moving average
    lo = np.clip(np.arange(len(p)) - k // 2, 0, len(p))
    hi = np.clip(lo + k, 0, len(p))
    e = np.sqrt(np.maximum((c[hi] - c[lo]) / np.maximum(hi - lo, 1), 0)) + 1e-9
    return 20 * np.log10(e)


def band_std(sig, fs, lo, hi):
    """Std of a (dB) envelope after FFT band-pass."""
    s = sig - sig.mean()
    S = np.fft.rfft(s * np.hanning(len(s)))
    f = np.fft.rfftfreq(len(s), 1 / fs)
    S[(f < lo) | (f > hi)] = 0
    return float(np.std(np.fft.irfft(S, len(s))) * np.sqrt(8 / 3))


def surrogate(m, seed=0):
    """Phase-randomised copy: same power spectrum, perfectly stationary.
    Its envelope statistics are the 'noise floor' a clean bed should match,
    so every check below is RELATIVE -> no per-texture calibration needed."""
    S = np.fft.rfft(m)
    ph = np.exp(2j * np.pi * np.random.default_rng(seed).uniform(0, 1, len(S)))
    ph[0] = 1
    return np.fft.irfft(np.abs(S) * ph, len(m))


def mod_measures(m):
    """Envelope-modulation measures for a mono signal."""
    T = THRESH
    e = env_db(m, 0.025)[:: FS // 100]                 # 25 ms RMS, 100 frames/s
    es = env_db(m, 0.5)[:: FS // 100]
    fast = env_db(m, 0.02)
    slow = env_db(m, 0.5)
    over = (fast - slow) > T["transient_crest_db"]
    on = np.flatnonzero(np.diff(over.astype(int)) == 1)
    if len(on):
        on = on[np.insert(np.diff(on) > 0.25 * FS, 0, True)]
    # sharpest periodic component in the EEG band (rail clacks, flutter)
    seg = 800                                          # 8 s Welch segments on the envelope
    ee = e - e.mean()
    segs = [ee[i:i + seg] * np.hanning(seg) for i in range(0, len(ee) - seg + 1, seg // 2)] or [ee * np.hanning(len(ee))]
    P = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], 0)
    f = np.fft.rfftfreq(len(segs[0]), 1 / 100)
    band = (f >= T["eeg_band"][0]) & (f <= T["eeg_band"][1])
    ref = (f >= 0.5) & (f <= 25)
    peak = 10 * np.log10(P[band].max() / np.median(P[ref]))
    return dict(eeg=band_std(e, 100, *T["eeg_band"]), swell=band_std(es, 100, *T["swell_band"]),
                trans=len(on) / (len(m) / FS / 60), peak=peak,
                peak_hz=float(f[band][np.argmax(P[band])]))


def screen(x, carrier):
    T = THRESH
    dur = len(x) / FS
    res, reasons, flags = {}, [], []
    edge = int(0.5 * FS)                               # ignore fade-in/out edges
    xc = x[edge:-edge] if len(x) > 4 * edge else x
    m = xc.mean(1)

    rms = 20 * np.log10(np.sqrt(np.mean(xc ** 2)) + 1e-12)
    clip = float(np.mean(np.abs(x) > 0.999))
    res.update(rms_dbfs=round(rms, 1), clip_frac=clip)
    if rms < T["min_rms_dbfs"]:
        reasons.append("near-silent")
    if clip > T["clip_frac_max"]:
        reasons.append("clipping")

    a = mod_measures(m)
    b = mod_measures(surrogate(m))
    eeg_x = a["eeg"] - b["eeg"]
    sw_x = a["swell"] - b["swell"]
    tr_x = max(0.0, a["trans"] - b["trans"])
    pk_x = a["peak"] - b["peak"]
    res.update(eeg_band_excess_db=round(eeg_x, 2), eeg_peak_excess_db=round(pk_x, 1),
               eeg_peak_hz=round(a["peak_hz"], 2), swell_excess_db=round(sw_x, 2),
               transients_excess_per_min=round(tr_x, 1))
    if eeg_x > T["eeg_band_excess_db"] or pk_x > T["eeg_peak_excess_db"]:
        reasons.append(f"rhythm in EEG band (excess {eeg_x:.1f} dB, peak +{pk_x:.0f} dB @ {a['peak_hz']:.1f} Hz)")
    if sw_x > T["swell_reject_db"]:
        reasons.append(f"built-in swell (excess {sw_x:.1f} dB)")
    elif sw_x > T["swell_flag_db"]:
        flags.append("swell (will be flattened)")
    if tr_x > T["transient_max_per_min"]:
        reasons.append(f"transients (+{tr_x:.0f}/min)")

    if carrier:                                        # tonal line near the carrier
        nfft = 1 << 14                                 # ~2.9 Hz bins, many averages
        segs = [m[i:i + nfft] * np.hanning(nfft) for i in range(0, len(m) - nfft, nfft // 2)]
        P = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], 0)
        f = np.fft.rfftfreq(nfft, 1 / FS)
        win = (f > carrier - T["hum_window_hz"]) & (f < carrier + T["hum_window_hz"])
        ref = (f > carrier / 1.5) & (f < carrier * 1.5)
        prom = 10 * np.log10(P[win].max() / np.median(P[ref]))
        res["carrier_hum_prominence_db"] = round(float(prom), 1)
        res["carrier_hum_freq_hz"] = round(float(f[win][np.argmax(P[win])]), 1)
        if prom > T["hum_reject_db"]:
            reasons.append(f"tonal line {res['carrier_hum_freq_hz']} Hz near carrier (+{prom:.0f} dB)")

    res["duration_s"] = round(dur, 1)
    res["verdict"] = "reject" if reasons else "pass"
    res["reasons"] = reasons
    res["flags"] = flags
    return res


# ===========================================================================
# 3. JOIN  /  4. CONDITION
# ===========================================================================
def join_loop(clips, xfade_s, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))
    xf = int(xfade_s * FS)
    w = np.linspace(0, np.pi / 2, xf)[:, None]
    fin, fout = np.sin(w), np.cos(w)
    out = clips[order[0]].copy()
    for k in order[1:]:
        c = clips[k]
        out[-xf:] = out[-xf:] * fout + c[:xf] * fin
        out = np.concatenate([out, c[xf:]], 0)
    # circular: blend the tail into the head, then drop the tail -> seamless loop
    head = out[:xf] * fin + out[-xf:] * fout
    return np.concatenate([head, out[xf:-xf]], 0), [int(i) for i in order]


def circ_conv_env(p, k):
    kern = np.zeros(len(p))
    kern[:k] = 1.0 / k
    kern = np.roll(kern, -k // 2)
    return np.fft.irfft(np.fft.rfft(p) * np.fft.rfft(kern), len(p))


def condition(x, crest_db=3.0):
    """Flatten slow rhythm (0.5 s RMS), cap transients; circular -> loop-safe."""
    p = (x ** 2).mean(1)
    slow = np.sqrt(np.maximum(circ_conv_env(p, FS // 2), 0)) + 1e-6
    fast = np.sqrt(np.maximum(circ_conv_env(p, FS // 50), 0)) + 1e-6
    lim = np.minimum(1.0, slow * 10 ** (crest_db / 20) / fast)
    lim = circ_conv_env(lim, FS // 100)
    # keep the overall level, remove only the slow variation around it
    target = np.exp(np.mean(np.log(slow)))
    return x * (lim * target / slow)[:, None]


def notch(x, carrier, depth_db=6.0, bw_oct=1 / 3):
    if not carrier:
        return x
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    z = np.log2(np.maximum(f, 1.0) / carrier) / (bw_oct / 2)
    g = 10 ** ((-depth_db * np.exp(-0.5 * (z * 1.18) ** 2)) / 20)
    return np.stack([np.fft.irfft(np.fft.rfft(x[:, c]) * g, n) for c in range(2)], 1)


def normalise(x):
    x = x - x.mean(0)
    x = x * 10 ** (TARGET_RMS_DBFS / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    pk = np.max(np.abs(x))
    ceil = 10 ** (PEAK_CEIL_DBFS / 20)
    return x * (ceil / pk) if pk > ceil else x


# ===========================================================================
# I/O
# ===========================================================================
def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw == 2:
        x = np.frombuffer(raw, "<i2") / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        i = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(i >= 1 << 23, i - (1 << 24), i) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, "<i4") / 2 ** 31
    else:
        raise ValueError(f"{path}: unsupported sample width")
    x = x.reshape(-1, ch).astype(np.float64)
    if sr != FS:
        g = gcd(FS, sr)
        x = resample_poly(x, FS // g, sr // g, axis=0)
    if x.shape[1] == 1:
        return pseudo_stereo(x[:, 0])
    return x[:, :2]


def load_any(path):
    if path.lower().endswith(".wav"):
        return read_wav(path)
    with open(path, "rb") as fh:
        return decode_audio(fh.read(), "mp3")


def write_wav24(path, x):
    x = np.clip(x, -1.0, 1.0 - 2 ** -23)
    i = (x * 2 ** 23).astype("<i4")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(FS)
        w.writeframes(i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--preset", choices=list(PRESETS), default="train_smooth")
    src.add_argument("--prompt", help="custom prompt (the flat-level line is appended)")
    src.add_argument("--from-dir", help="screen/join existing WAV/MP3 files instead of calling the API")
    ap.add_argument("--takes", type=int, default=10)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--influence", type=float, default=0.65)
    ap.add_argument("--model", default="eleven_text_to_sound_v2")
    ap.add_argument("--output-format", default="pcm_48000")
    ap.add_argument("--carrier", type=float, default=144.0, help="BB carrier (Hz); 0 = off")
    ap.add_argument("--min-pass", type=int, default=4, help="minimum passing takes to build a bed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-condition", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="bed_out")
    a = ap.parse_args()

    carrier = a.carrier if a.carrier > 0 else None
    raw_dir = os.path.join(a.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ---- 1. generate / collect -------------------------------------------
    if a.from_dir:
        paths = sorted(p for p in glob.glob(os.path.join(a.from_dir, "*"))
                       if p.lower().endswith((".wav", ".mp3")))
        prompt, name = None, os.path.basename(os.path.normpath(a.from_dir))
        print(f"[1/5] using {len(paths)} files from {a.from_dir}")
    else:
        prompt = (a.prompt + " " + FLAT_LINE) if a.prompt else PRESETS[a.preset].format(flat=FLAT_LINE)
        name = "custom" if a.prompt else a.preset
        print(f"[1/5] generating {a.takes} takes  ({name})")
        paths = generate_takes(a, prompt, raw_dir)
        if a.dry_run:
            print("dry run: nothing generated")
            return

    # ---- 2. screen ---------------------------------------------------------
    print(f"[2/5] screening {len(paths)} takes")
    clips, report = [], []
    for p in paths:
        x = load_any(p)
        r = screen(x, carrier)
        r["file"] = os.path.basename(p)
        report.append(r)
        tag = "PASS" if r["verdict"] == "pass" else "REJECT"
        why = "; ".join(r["reasons"] + r["flags"])
        print(f"  {tag:6s} {r['file']:<16s} eeg {r['eeg_band_excess_db']:+.2f} dB "
              f"(peak {r['eeg_peak_excess_db']:+.0f} dB)  swell {r['swell_excess_db']:+.2f} dB  "
              f"hum {r.get('carrier_hum_prominence_db', 0):.0f} dB  "
              f"trans {r['transients_excess_per_min']:+.0f}/min  {why}")
        if r["verdict"] == "pass":
            clips.append(x)

    if len(clips) < a.min_pass:
        print(f"only {len(clips)} takes passed (< {a.min_pass}); generate more or relax THRESH")
        with open(os.path.join(a.out, "report.json"), "w") as fh:
            json.dump({"name": name, "prompt": prompt, "takes": report, "built": False}, fh, indent=2)
        sys.exit(2)

    # ---- 3. join -----------------------------------------------------------
    print(f"[3/5] joining {len(clips)} takes with {XFADE_S:g} s crossfades")
    bed, order = join_loop(clips, XFADE_S, a.seed)

    # ---- 4. condition ------------------------------------------------------
    print("[4/5] conditioning" + (" (skipped)" if a.no_condition else ""))
    if not a.no_condition:
        bed = condition(bed)
    bed = normalise(notch(bed, carrier))

    # ---- 5. export + final check ------------------------------------------
    out_wav = os.path.join(a.out, f"bed_{name}.wav")
    write_wav24(out_wav, bed)
    final = screen(bed, carrier)
    final["loop_seam_jump"] = float(np.max(np.abs(bed[0] - bed[-1])))
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump({"name": name, "prompt": prompt, "takes": report, "order": order,
                   "bed_file": os.path.basename(out_wav), "bed_seconds": round(len(bed) / FS, 1),
                   "final_screen": final, "thresholds": THRESH, "carrier": carrier,
                   "built": True}, fh, indent=2, default=str)
    print(f"[5/5] wrote {out_wav}  ({len(bed) / FS / 60:.1f} min loop)  "
          f"final: eeg {final['eeg_band_excess_db']:+.2f} dB, swell {final['swell_excess_db']:+.2f} dB, "
          f"verdict {final['verdict']}")
    print(f"next: swell-engine {out_wav} nap.wav --profile default --simulate")


if __name__ == "__main__":
    main()

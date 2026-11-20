"""Stage 1: the texture bed.

ElevenLabs takes  ->  screen  ->  join into an exact-length seamless loop  ->  condition.

The bed must be FLAT: no rhythm of its own. Stage 2 (swell.py) adds the only rhythm.
"""

import glob
import json
import os
import re

import numpy as np
from scipy import fft as sfft

from ..core import elevenlabs as el
from ..core.analysis import (ENV_RATE, OCTAVE_EDGES_HZ, band_std, circ_conv_env, env_db,  # noqa: F401
                             env_frames, frames_to_samples, octave_balance, power_frames, surrogate)
from ..core.audio_io import FS, decode_audio, load_any, normalise, rms, write_wav24
from ..core.loop import chain, close_loop, loopable_samples  # noqa: F401
from ..core.slots import notch, notch_gain  # noqa: F401

FLAT_LINE = "Constant level throughout, no swelling, no surges, no fades, no build-ups."

# --- screening thresholds (empirical-first; tune with your own data) -------
THRESH = dict(
    eeg_band=(1.0, 12.0),        # Hz, envelope modulation band to keep clean
    # all modulation checks are EXCESS over a phase-randomised (stationary)
    # surrogate of the same take, so random noise fluctuation scores ~0
    eeg_band_excess_db=0.8,      # extra log-envelope std in 1-12 Hz (dB): RANDOM flicker. Was 0.3 (an
                                 # untuned guess) and rejected 5/6 smooth v3 rain takes at 0.31-0.83.
                                 # Regular rhythm is caught by eeg_peak_excess_db, which stays strict.
    eeg_peak_excess_db=6.0,      # extra sharpness of the strongest 1-12 Hz line (dB)
    swell_band=(0.05, 0.3),      # Hz
    swell_flag_db=0.75,          # level or brightness swell > this: condition, then re-check
    swell_reject_db=3.0,         # level swell > this: reported as a built-in swell
    swell_after_max_db=0.75,     # after conditioning, level AND brightness swell must be below this
    bright_xover_hz=1500.0,      # treble/bass split for the brightness-swell measure
    hum_window_hz=20.0,          # look for tonal lines within +/- this of each carrier
    hum_reject_db=10.0,          # line prominence over local spectral median (dB)
    transient_crest_db=6.0,      # 20 ms over 0.5 s envelope
    transient_max_per_min=6.0,   # reject if this many MORE events/min than surrogate
    min_rms_dbfs=-60.0,          # reject near-silence
    clip_frac_max=1e-4,          # reject if more samples than this are at full scale
    tonal_match_db=10.0,         # takes joined into one bed must match within this in every AUDIBLE
                                 # octave (5 dB in all octaves rejected 13 of 18 passing vehicle/wind
                                 # takes, mostly over -45 dB treble nobody hears). The gap left is
                                 # then closed by a static per-take EQ, up to tonal_eq_max_db.
    tonal_floor_db=-30.0,        # an octave is audible if it holds more than this share of a take's power
    tonal_eq_max_db=6.0,         # max static per-octave correction toward the group's median balance
)
XFADE_S = 3.0
REF_HOP_S = 40.0             # SA3 reference windows start every this many seconds
REF_TAKES_PER_WINDOW = 2     # most takes (different seeds) drawn from one window of a reference
# Conditioning divides out each band's level averaged over this window. 0.5 s removes all motion
# slower than ~2 Hz (gusts, washes, sway): the scenes lose their identity and turn into coloured
# noise. 3 s removes only the swell band (< ~0.3 Hz) that the swell stage replaces. Set per run
# (gen --flatten-s); it is part of the screening cache key.
FLATTEN_S = 0.5
PRECAP_DB = None   # if set, every take's transients are capped at +this dB (condition()) BEFORE
                   # screening, so the unchanged screens judge the softened audio that ships


# ===========================================================================
# Generate (with a take cache: existing raw takes are reused, never re-bought)
# ===========================================================================
def api_generate(text, duration, influence, model, output_format, api_key, base):
    return el.sound_generation(text, duration, influence, model, output_format, loop=True,
                               key=api_key, base=base)


def cached_takes(raw_dir, prompt=None, model=None, match=None):
    """Takes in raw_dir; with prompt (and model), only those whose sidecar JSON records that
    exact prompt and model (and every field in match), so editing a prompt or switching models
    never silently reuses takes made the old way."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "take_*.wav")))
    if prompt is None:
        return paths
    keep = []
    for p in paths:
        try:
            with open(p[:-4] + ".json") as fh:
                meta = json.load(fh)
                if (meta.get("prompt") == prompt and (model is None or meta.get("model") == model)
                        and all(meta.get(k) == v for k, v in (match or {}).items())):
                    keep.append(p)
        except (OSError, ValueError):
            pass
    return keep


def generate_takes(prompt, raw_dir, n_takes, duration=30.0, influence=0.65,
                   model="eleven_text_to_sound_v3", output_format="pcm_48000", dry_run=False):
    """Top the cache in raw_dir up to n_takes takes. Returns every cached take path."""
    os.makedirs(raw_dir, exist_ok=True)
    have = cached_takes(raw_dir, prompt, model)
    other = len(cached_takes(raw_dir)) - len(have)
    missing = max(0, n_takes - len(have))
    if have or other:
        print(f"  {len(have)} cached take(s) in {raw_dir}" + (f", need {missing} more" if missing else "")
              + (f" ({other} made from a different prompt or model, ignored)" if other else ""))
    if not missing:
        return have
    api_key = el.api_key(required=not dry_run)
    base = el.api_base()
    nums = [int(m.group(1)) for p in cached_takes(raw_dir) if (m := re.search(r"take_(\d+)\.wav$", p))]
    first = max(nums, default=0) + 1
    fmt = output_format
    for i in range(first, first + missing):
        infl = float(np.clip(influence + (i % 3 - 1) * 0.05, 0.0, 1.0))   # a little variety
        print(f"  take {i:02d}  influence={infl:.2f}  format={fmt}")
        if dry_run:
            print(f"    POST {base}/v1/sound-generation  text={prompt[:70]!r}...  "
                  f"duration={duration} loop=True model={model}")
            continue
        try:
            blob = api_generate(prompt, duration, infl, model, fmt, api_key, base)
        except RuntimeError as e:
            if fmt.startswith("pcm_") and el.pcm_unavailable(e):
                print(f"    PCM not available ({e}); falling back to mp3_44100_192")
                fmt = "mp3_44100_192"
                blob = api_generate(prompt, duration, infl, model, fmt, api_key, base)
            else:
                raise
        p = os.path.join(raw_dir, f"take_{i:02d}.wav")
        write_wav24(p, normalise(decode_audio(blob, fmt, seed=i, duration=duration)))
        with open(p[:-4] + ".json", "w") as fh:
            json.dump({"prompt": prompt, "influence": infl, "duration": duration,
                       "model": model, "format": fmt}, fh, indent=2)
    return cached_takes(raw_dir, prompt, model)


def sa3_generate_takes(spec, raw_dir, n_takes, duration=60.0, dit="medium", steps=8, trim_s=3.0,
                       dry_run=False):
    """Stable Audio 3 (local, free): top raw_dir up to n_takes takes of spec = dict(prompt,
    negative, cfg[, init, init_noise]). Take k uses seed k, so takes are reproducible. Each is
    rendered trim_s longer at both ends and trimmed: SA3 clips tend to fade in and out.

    With init (a list of reference recordings), each take is audio-to-audio from one of them,
    starting init_noise of the way from that recording to noise (0.5 keeps it close, 0.7 keeps its
    timbre and texture but recomposes it). A new take goes to the least-used reference, at its
    least-used window (windows start every REF_HOP_S). A reference gives at most
    REF_TAKES_PER_WINDOW takes per window, here and in what is returned: an 81 s recording has one
    66 s window, and 13 takes from it made a repetitive, unrealistic wind bed (2026-11-19).
    Returns every matching take."""
    from ..core import sa3
    from ..core.audio_io import read_wav
    prompt, neg, cfg = spec["prompt"], spec.get("negative"), float(spec.get("cfg", 1.0))
    neg = neg if cfg != 1.0 else None
    model = sa3.model_tag(dit, steps, cfg)
    inits = spec.get("init") or []
    noise = float(spec.get("init_noise", 0.6))
    match = dict(negative=neg, trim_s=trim_s)
    if inits:
        match.update(init_noise=noise)

    n_in = int((duration + 2 * trim_s) * FS)
    ref_audio, windows = {}, {}
    for ref in inits:
        if not os.path.exists(ref):
            if dry_run:
                windows[ref] = [0]
                continue
            raise SystemExit(f"reference {ref} is missing (refs/ is local only: see scripts/freesound_refs.py)")
        ref_audio[ref] = load_any(ref)
        span = max(0, len(ref_audio[ref]) - n_in)
        windows[ref] = list(range(0, span + 1, int(REF_HOP_S * FS)))
    cap = {ref: REF_TAKES_PER_WINDOW * len(w) for ref, w in windows.items()}

    def matching():
        """Cached takes made this way, up to each reference's cap. A reference take counts while its
        own reference is still listed, so dropping a reference keeps the takes made from the others."""
        out, used = [], {}
        for p in cached_takes(raw_dir, prompt, model, match):
            with open(p[:-4] + ".json") as fh:
                src = (json.load(fh).get("init_from") or {}).get("file")
            if not inits:
                if src is None:
                    out.append(p)
            elif src in inits and used.get(src, 0) < cap[src]:
                used[src] = used.get(src, 0) + 1
                out.append(p)
        return out

    def usage(paths):
        per_ref, per_win = {r: 0 for r in inits}, {}
        for p in paths:
            with open(p[:-4] + ".json") as fh:
                f = json.load(fh).get("init_from") or {}
            if f.get("file") in per_ref:
                per_ref[f["file"]] += 1
                key = (f["file"], int(round(f["offset_s"] * FS)))
                per_win[key] = per_win.get(key, 0) + 1
        return per_ref, per_win
    os.makedirs(raw_dir, exist_ok=True)
    have = matching()
    per_ref, per_win = usage(have)
    if inits:
        room = sum(cap[r] - per_ref[r] for r in inits)
        if n_takes - len(have) > room:
            print(f"  the references hold only {room} more take(s) ({REF_TAKES_PER_WINDOW} per {duration + 2 * trim_s:g} s "
                  f"window every {REF_HOP_S:g} s): add longer or more references")
        n_takes = min(n_takes, len(have) + room)
    missing = max(0, n_takes - len(have))
    if have:
        print(f"  {len(have)} cached take(s) in {raw_dir}" + (f", need {missing} more" if missing else ""))
    if not missing:
        return have
    nums = [int(m.group(1)) for p in cached_takes(raw_dir) if (m := re.search(r"take_(\d+)\.wav$", p))]
    seeds_used = set()
    for p in have:
        with open(p[:-4] + ".json") as fh:
            seeds_used.add(json.load(fh).get("seed"))
    seeds = [k for k in range(1, 10 * n_takes + 100) if k not in seeds_used][:missing]
    first = max(nums, default=0) + 1
    for i, seed in zip(range(first, first + missing), seeds):
        print(f"  take {i:02d}  sa3 {dit} seed={seed} cfg={cfg:g}")
        if dry_run:
            print(f"    ./sa3 --dit {dit} --seconds {duration + 2 * trim_s:g} --seed {seed} --cfg {cfg:g}"
                  + (" --negative-prompt ..." if neg else "") + f"  {prompt[:60]!r}...")
            continue
        p = os.path.join(raw_dir, f"take_{i:02d}.wav")
        tmp = p[:-4] + ".sa3.wav"
        init_wav, init_from = None, None
        if inits:
            ref = min((r for r in inits if per_ref[r] < cap[r]), key=lambda r: (per_ref[r], inits.index(r)))
            off = min(windows[ref], key=lambda o: (per_win.get((ref, o), 0), o))
            per_ref[ref] += 1
            per_win[(ref, off)] = per_win.get((ref, off), 0) + 1
            init_wav, init_from = p[:-4] + ".init.wav", dict(file=ref, offset_s=round(off / FS, 2))
            sa3.write_init(ref_audio[ref][off:off + n_in], init_wav, FS)
            print(f"    from {os.path.basename(ref)} at {off / FS:.0f} s, noise {noise:g}")
        sa3.generate(prompt, tmp, duration + 2 * trim_s, seed, dit, steps, cfg, neg,
                     init_wav=init_wav, init_noise=noise if inits else None)
        if init_wav:
            os.remove(init_wav)
        x = read_wav(tmp)
        k = int(round(trim_s * FS))
        write_wav24(p, normalise(x[k:len(x) - k] if k else x))
        os.remove(tmp)
        with open(p[:-4] + ".json", "w") as fh:
            meta = dict(prompt=prompt, negative=neg, cfg=cfg, seed=seed, duration=duration, trim_s=trim_s,
                        model=model, engine="sa3", dit=dit, steps=steps)
            if inits:
                meta.update(init=list(inits), init_noise=noise, init_from=init_from)
            json.dump(meta, fh, indent=2)
    return matching()


def ref_takes(refs, raw_dir, duration=60.0, dry_run=False):
    """Takes cut straight from the scene's real recordings, no model: each reference is split into
    back-to-back duration-long windows (no overlap, so no material repeats in the bed), and the
    usual screens and bed processing decide which are used. SA3 audio-to-audio recomposed the
    recordings too much to keep them recognisable, the train most of all (listening, 2026-11-20).
    A window already cut (same reference and offset) is reused. Returns every take of these refs."""
    os.makedirs(raw_dir, exist_ok=True)
    have = {}
    for p in cached_takes(raw_dir):
        with open(p[:-4] + ".json") as fh:
            f = json.load(fh).get("init_from") or {}
        have[(f.get("file"), f.get("offset_s"))] = p
    nums = [int(m.group(1)) for p in cached_takes(raw_dir) if (m := re.search(r"take_(\d+)\.wav$", p))]
    nxt = max(nums, default=0) + 1
    n = int(round(duration * FS))
    out, new = [], 0
    for ref in refs:
        if not os.path.exists(ref):
            if dry_run:
                print(f"  would cut {ref} into {duration:g} s windows")
                continue
            raise SystemExit(f"reference {ref} is missing (refs/ is local only: see scripts/freesound_refs.py)")
        x = load_any(ref)
        offs = [round(o / FS, 2) for o in range(0, len(x) - n + 1, n)]
        for off in offs:
            p = have.get((ref, off))
            if p is None and not dry_run:
                p = os.path.join(raw_dir, f"take_{nxt:02d}.wav")
                nxt += 1
                new += 1
                k = int(round(off * FS))
                write_wav24(p, normalise(x[k:k + n]))
                with open(p[:-4] + ".json", "w") as fh:
                    json.dump(dict(prompt=None, model="ref", engine="ref", duration=duration,
                                   init_from=dict(file=ref, offset_s=off)), fh, indent=2)
            if p:
                out.append(p)
        print(f"  {os.path.basename(ref)[:48]}: {len(offs)} window(s) of {duration:g} s")
    if new:
        print(f"  cut {new} new take(s) into {raw_dir}")
    return out


# ===========================================================================
# Screen
# ===========================================================================
def mod_measures(m):
    T = THRESH
    e = env_frames(m, 0.025)                           # 25 ms RMS, 100 frames/s
    es = env_frames(m, 0.5)
    over = (env_db(m, 0.02) - env_db(m, 0.5)) > T["transient_crest_db"]
    on = np.flatnonzero(np.diff(over.astype(int)) == 1)
    if len(on):
        on = on[np.insert(np.diff(on) > 0.25 * ENV_RATE, 0, True)]
    seg = 800                                          # 8 s Welch segments on the envelope
    ee = e - e.mean()
    segs = [ee[i:i + seg] * np.hanning(seg) for i in range(0, len(ee) - seg + 1, seg // 2)] \
        or [ee * np.hanning(len(ee))]
    P = np.mean([np.abs(sfft.rfft(s)) ** 2 for s in segs], 0)
    f = sfft.rfftfreq(len(segs[0]), 1 / 100)
    band = (f >= T["eeg_band"][0]) & (f <= T["eeg_band"][1])
    ref = (f >= 0.5) & (f <= 25)
    return dict(eeg=band_std(e, 100, *T["eeg_band"]), swell=band_std(es, 100, *T["swell_band"]),
                trans=len(on) / (len(m) / FS / 60), peak=10 * np.log10(P[band].max() / np.median(P[ref])),
                peak_hz=float(f[band][np.argmax(P[band])]))


def brightness_swell(m):
    """Swing of the treble-to-bass balance (dB) in the swell band: wave crests that get
    brighter. Level flattening alone can't remove it."""
    f = sfft.rfftfreq(len(m), 1 / FS)
    S = sfft.rfft(m)
    hi_mask = f > THRESH["bright_xover_hz"]
    r = env_frames(sfft.irfft(S * hi_mask, len(m)), 0.5) - env_frames(sfft.irfft(S * ~hi_mask, len(m)), 0.5)
    r = r[50:-50] if len(r) > 200 else r               # drop the 0.5 s window's edge frames
    return band_std(r, 100, *THRESH["swell_band"])


def screen(x):
    T = THRESH
    res, reasons, flags = {}, [], []
    edge = int(0.5 * FS)                               # ignore fade-in/out edges
    xc = x[edge:-edge] if len(x) > 4 * edge else x
    m = xc.mean(1)

    level = 20 * np.log10(np.sqrt(np.mean(xc ** 2)) + 1e-12)
    clip = float(np.mean(np.abs(x) > 0.999))
    res.update(rms_dbfs=round(level, 1), clip_frac=clip)
    if level < T["min_rms_dbfs"]:
        reasons.append("near-silent")
    if clip > T["clip_frac_max"]:
        reasons.append("clipping")

    a = mod_measures(m)
    # average 3 surrogates: one alone scatters the line check by about +/-0.5 dB, enough to flip
    # a borderline take between pass and reject on luck
    bs = [mod_measures(surrogate(m, seed=k)) for k in range(3)]
    b = {k: float(np.mean([q[k] for q in bs])) for k in ("eeg", "swell", "trans", "peak")}
    eeg_x, sw_x, pk_x = a["eeg"] - b["eeg"], a["swell"] - b["swell"], a["peak"] - b["peak"]
    tr_x = max(0.0, a["trans"] - b["trans"])
    br_x = brightness_swell(m) - brightness_swell(surrogate(m))
    res.update(eeg_band_excess_db=round(eeg_x, 2), eeg_peak_excess_db=round(pk_x, 1),
               eeg_peak_hz=round(a["peak_hz"], 2), swell_excess_db=round(sw_x, 2),
               brightness_swell_excess_db=round(br_x, 2), transients_excess_per_min=round(tr_x, 1))
    if eeg_x > T["eeg_band_excess_db"] or pk_x > T["eeg_peak_excess_db"]:
        reasons.append(f"rhythm in EEG band (excess {eeg_x:.1f} dB, peak +{pk_x:.0f} dB @ {a['peak_hz']:.1f} Hz)")
    if sw_x > T["swell_reject_db"]:
        reasons.append(f"built-in swell (excess {sw_x:.1f} dB)")
        res["swell_reason"] = reasons[-1]
    elif sw_x > T["swell_flag_db"]:
        flags.append("swell (will be flattened)")
    if tr_x > T["transient_max_per_min"]:
        reasons.append(f"transients (+{tr_x:.0f}/min)")

    res["duration_s"] = round(len(x) / FS, 1)
    res["verdict"] = "reject" if reasons else "pass"
    res["reasons"], res["flags"] = reasons, flags
    return res


def hum_measures(x, carriers):
    """Tonal-line prominence (dB over the local spectral median) within +/-hum_window_hz of each
    carrier. One averaged spectrum serves every carrier. Returns {"<c>": dict(prominence_db, freq_hz)}."""
    T = THRESH
    edge = int(0.5 * FS)
    m = (x[edge:-edge] if len(x) > 4 * edge else x).mean(1)
    nfft = 1 << 14
    segs = [m[i:i + nfft] * np.hanning(nfft) for i in range(0, len(m) - nfft, nfft // 2)]
    P = np.mean([np.abs(sfft.rfft(s)) ** 2 for s in segs], 0)
    f = sfft.rfftfreq(nfft, 1 / FS)
    out = {}
    for c in carriers:
        win = (f > c - T["hum_window_hz"]) & (f < c + T["hum_window_hz"])
        ref = (f > c / 1.5) & (f < c * 1.5)
        prom = 10 * np.log10(P[win].max() / (np.median(P[ref]) + 1e-30) + 1e-30)
        out[f"{c:g}"] = dict(prominence_db=round(float(prom), 1), freq_hz=round(float(f[win][np.argmax(P[win])]), 1))
    return out


def with_hum(r, hum):
    """Add the per-carrier hum verdicts to a carrier-free screening result."""
    r = json.loads(json.dumps(r))
    r["carrier_hum"] = hum
    if hum:
        worst = max(hum.values(), key=lambda h: h["prominence_db"])
        r["carrier_hum_prominence_db"], r["carrier_hum_freq_hz"] = worst["prominence_db"], worst["freq_hz"]
    for c, h in hum.items():
        if h["prominence_db"] > THRESH["hum_reject_db"]:
            r["reasons"].append(f"tonal line {h['freq_hz']} Hz near the {c} Hz carrier (+{h['prominence_db']:.0f} dB)")
    r["verdict"] = "reject" if r["reasons"] else "pass"
    return r


# ===========================================================================
# Join / condition
# ===========================================================================


CONDITION_BANDS_HZ = (150.0, 400.0, 1000.0, 2500.0, 6000.0)   # crossovers -> 6 bands


def _flatten(x):
    """Divide out the slow (FLATTEN_S RMS) level, keeping the band's average level."""
    slow = np.sqrt(np.maximum(circ_conv_env(power_frames(x), int(round(ENV_RATE * FLATTEN_S))), 0)) + 1e-9
    return x * frames_to_samples(np.exp(np.mean(np.log(slow))) / slow, len(x))[:, None]


def condition(x, crest_db=6.0, bands=CONDITION_BANDS_HZ):
    """Remove the take's own slow rhythm, then cap transients at +crest_db. Circular -> loop-safe.

    The slow level is flattened in each frequency band separately, so a wave that gets brighter
    at its crest is flattened in timbre as well as loudness. The band masks are zero-phase and
    sum to exactly 1, so the spectrum is untouched; only slow level changes per band are removed."""
    n = len(x)
    f = sfft.rfftfreq(n, 1 / FS)
    lp = [np.zeros_like(f)] + [1.0 / (1.0 + (f / fc) ** 4) for fc in bands] + [np.ones_like(f)]
    X = sfft.rfft(x, axis=0)
    y = np.zeros_like(x)
    for lo, hi in zip(lp[:-1], lp[1:]):
        y += _flatten(sfft.irfft(X * (hi - lo)[:, None], n, axis=0))
    p = power_frames(y)
    slow = np.sqrt(np.maximum(circ_conv_env(p, ENV_RATE // 2), 0)) + 1e-6
    fast = np.sqrt(np.maximum(circ_conv_env(p, ENV_RATE // 50), 0)) + 1e-6
    lim = circ_conv_env(np.minimum(1.0, slow * 10 ** (crest_db / 20) / fast), ENV_RATE // 100)
    return y * frames_to_samples(lim, n)[:, None]


def screen_take(x, do_condition=True):
    """screen() plus the verdict on the take's own slow rhythm.

    A loudness or brightness swell is not a reason to reject by itself when conditioning will
    run: the take is conditioned and screened again, and kept only if both swells are then below
    swell_after_max_db. Rhythm in the EEG band, transients, hum, clipping and silence are judged
    on the raw take and always reject.
    Carrier hum is judged separately (hum_measures), so this result holds for any carrier."""
    T = THRESH
    r = screen(x)
    swell_reason = r.pop("swell_reason", None)
    other = [q for q in r["reasons"] if q != swell_reason]
    r["flags"] = [q for q in r["flags"] if not q.startswith("swell")]
    swelly = (swell_reason is not None or r["swell_excess_db"] > T["swell_flag_db"]
              or r["brightness_swell_excess_db"] > T["swell_flag_db"])
    if not swelly or other:
        r["reasons"] = other + ([swell_reason] if swell_reason else [])
    elif not do_condition:
        worst = max(r["swell_excess_db"], r["brightness_swell_excess_db"])
        r["reasons"] = [f"built-in swell {worst:+.2f} dB and conditioning is off"] \
            if worst > T["swell_after_max_db"] else []
    else:
        c = screen(condition(x))
        after = dict(swell_excess_db=c["swell_excess_db"], brightness_swell_excess_db=c["brightness_swell_excess_db"])
        r["after_conditioning"] = after
        desc = (f"own swell {r['swell_excess_db']:+.2f} dB level / {r['brightness_swell_excess_db']:+.2f} dB "
                f"brightness -> {after['swell_excess_db']:+.2f} / {after['brightness_swell_excess_db']:+.2f} dB "
                f"after conditioning")
        if max(after.values()) <= T["swell_after_max_db"]:
            r["reasons"] = []
            r["flags"].append(desc)
        else:
            r["reasons"] = [desc + f" (limit {T['swell_after_max_db']:g})"]
    if do_condition and not r["reasons"]:
        # judged as it ships too: conditioning can add transients (river, 2026-11-20)
        tc = screen(condition(x))["transients_excess_per_min"]
        r["transients_after_conditioning_per_min"] = round(tc, 1)
        if tc > T["transient_max_per_min"]:
            r["reasons"] = [f"transients after conditioning (+{tc:.0f}/min)"]
    r["verdict"] = "reject" if r["reasons"] else "pass"
    return r


def tonal_distance(a, b, floor):
    """Largest octave-balance gap (dB) between two takes, over octaves either take can be heard in."""
    audible = (a >= floor) | (b >= floor)
    return float(np.max(np.abs(a - b)[audible])) if audible.any() else 0.0


def tonal_group(balances, tol, floor=-30.0):
    """Indices of the largest set of takes whose audible octave balances all match within tol dB
    (exhaustive over subsets for small n, greedy beyond), so crossfades never jump in timbre."""
    n = len(balances)
    fits = lambda i, j: tonal_distance(balances[i], balances[j], floor) <= tol
    if n <= 14:
        from itertools import combinations
        for k in range(n, 0, -1):
            for c in combinations(range(n), k):
                if all(fits(i, j) for i, j in combinations(c, 2)):
                    return list(c)
    best = []
    for seed in range(n):
        g = [seed]
        for j in sorted(range(n), key=lambda j: tonal_distance(balances[j], balances[seed], floor)):
            if j != seed and all(fits(j, i) for i in g):
                g.append(j)
        best = max(best, g, key=len)
    return sorted(best)


def match_tone(clips, balances, max_db):
    """Static, zero-phase EQ pulling each take's octave balance toward the group median (at most
    max_db per octave, interpolated smoothly over log frequency), then each take to the median
    RMS, so the crossfades between takes don't change timbre or level."""
    e = np.asarray(OCTAVE_EDGES_HZ, float)
    lc = np.log2(np.sqrt(e[:-1] * e[1:]))
    target = np.median(balances, axis=0)
    level = float(np.median([rms(c) for c in clips]))
    out, applied = [], []
    for c, b in zip(clips, balances):
        g = np.clip(target - b, -max_db, max_db)
        f = sfft.rfftfreq(len(c), 1 / FS)
        curve = 10 ** (np.interp(np.log2(np.maximum(f, 1.0)), lc, g) / 20)
        y = sfft.irfft(sfft.rfft(c, axis=0) * curve[:, None], len(c), axis=0)
        out.append(y * (level / (rms(y) + 1e-12)))
        applied.append([round(float(v), 1) for v in g])
    return out, applied


SCREEN_VERSION = 5   # bump whenever a measurement changes, so cached verdicts get recomputed


def _screen_key(do_condition):
    import hashlib
    t = {k: v for k, v in THRESH.items()                 # grouping and hum are not screening
         if not k.startswith(("tonal_", "hum_"))}
    d = dict(v=SCREEN_VERSION, t=t, d=do_condition, b=CONDITION_BANDS_HZ)
    if FLATTEN_S != 0.5:                                  # older keys stay valid
        d["f"] = FLATTEN_S
    if PRECAP_DB:
        d["p"] = PRECAP_DB
    blob = json.dumps(d,
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _hum_key(c):
    return f"{c:g}@{THRESH['hum_window_hz']:g}"


def screen_take_cached(path, x, carriers, do_condition=True):
    """screen_take() plus hum at each carrier, remembered in the take's sidecar JSON (raw takes
    only: files without a sidecar, e.g. --from-dir, are always screened). The screening result
    is carrier-free; hum is cached per carrier, so adding a carrier costs one spectrum, not a
    re-screen. x may be the audio or a loader callable, called only on a cache miss.
    Returns (result, was_cached)."""
    carriers = [c for c in (carriers or []) if c]
    side = os.path.splitext(path)[0] + ".json"
    key = _screen_key(do_condition)
    meta, base = None, None
    if os.path.exists(side):
        try:
            with open(side) as fh:
                meta = json.load(fh)
        except ValueError:
            meta = None
        hit = (meta or {}).get("screen", {})
        if hit.get("key") == key:
            base = hit["result"]
    hums = dict((meta or {}).get("hum", {}))
    need = [c for c in carriers if _hum_key(c) not in hums]
    cached = base is not None and not need
    if not cached:
        if callable(x):
            x = x()
        if base is None:
            base = screen_take(x, do_condition)
        if need:
            got = hum_measures(x, need)
            for c in need:
                hums[_hum_key(c)] = got[f"{c:g}"]
        if meta is not None:
            meta["screen"] = {"key": key, "result": base}
            meta["hum"] = hums
            with open(side, "w") as fh:
                json.dump(meta, fh, indent=2, default=str)
    return with_hum(base, {f"{c:g}": hums[_hum_key(c)] for c in carriers}), cached


def screen_and_match(paths, carriers, do_condition=True):
    """Screen takes, keep the passing ones that match in tone, and EQ them toward each other.
    Returns (clips, names, report)."""
    xf = int(XFADE_S * FS)
    clips, names, takes = [], [], []
    n_cached = 0
    for i, p in enumerate(paths):
        x = load_any(p, seed=i)
        if PRECAP_DB:
            x = condition(x, crest_db=PRECAP_DB)
        r, hit = screen_take_cached(p, x, carriers, do_condition)
        n_cached += hit
        r["file"] = os.path.basename(p)
        takes.append(r)
        if r["verdict"] == "pass" and len(x) < 3 * xf:
            r["verdict"], r["reasons"] = "reject", [f"shorter than {3 * XFADE_S:g} s"]
        tag = "PASS" if r["verdict"] == "pass" else "REJECT"
        why = "; ".join(r["reasons"] + r["flags"])
        print(f"  {tag:6s} {r['file']:<14s} eeg {r['eeg_band_excess_db']:+.2f} dB "
              f"(peak {r['eeg_peak_excess_db']:+.0f} dB)  swell {r['swell_excess_db']:+.2f} / "
              f"{r['brightness_swell_excess_db']:+.2f} dB  "
              f"hum {r.get('carrier_hum_prominence_db', 0):.0f} dB  "
              f"trans {r['transients_excess_per_min']:+.0f}/min  {why}")
        if r["verdict"] == "pass":
            clips.append(x)
            names.append(r["file"])
    if n_cached:
        print(f"  ({n_cached} of {len(paths)} screening results reused from the take sidecars)")
    report = {"takes": takes, "built": False}
    if len(clips) > 1:
        bal = [octave_balance(c) for c in clips]
        keep = tonal_group(bal, THRESH["tonal_match_db"], THRESH["tonal_floor_db"])
        for k in range(len(clips)):
            if k not in keep:
                r = next(t for t in takes if t["file"] == names[k])
                worst = max(tonal_distance(bal[k], bal[j], THRESH["tonal_floor_db"]) for j in keep)
                r["verdict"], r["reasons"] = "reject", [f"tonal outlier: {worst:.0f} dB off the other passing takes"]
                print(f"  REJECT {names[k]:<14s} tonal outlier ({worst:.0f} dB off the matching set)")
        clips, names = [clips[k] for k in keep], [names[k] for k in keep]
        if len(clips) > 1:
            clips, eq = match_tone(clips, [bal[k] for k in keep], THRESH["tonal_eq_max_db"])
            report["tone_match_eq_db"] = dict(zip(names, eq))
            print(f"  tone-matched {len(clips)} takes (max correction "
                  f"{max(max(abs(v) for v in g) for g in eq):.1f} dB per octave)")
    return clips, names, report


def _finish(clips, order, n, carriers, do_condition):
    xf = int(XFADE_S * FS)
    bed = close_loop(chain(clips, order, xf), n, xf)
    if do_condition:
        bed = condition(bed)
    return normalise(notch(bed, carriers))


def build_bed(paths, carriers, loop_unit_s, loop_seconds=None, seed=1, min_pass=3,
              do_condition=True):
    """Screen takes, join the passing ones into a loop whose length is a whole multiple of
    loop_unit_s (so every swell rate fits a whole number of waves), condition, notch, normalise.
    Returns (bed or None, report dict)."""
    xf = int(XFADE_S * FS)
    clips, names, report = screen_and_match(paths, carriers, do_condition)
    if len(clips) < min_pass:
        report["error"] = f"only {len(clips)} takes passed (< {min_pass}); add takes or relax THRESH"
        return None, report

    unit = int(round(loop_unit_s * FS))
    avail = loopable_samples(clips, xf)
    if loop_seconds:
        n = int(round(loop_seconds * FS))
        if n % unit:
            raise SystemExit(f"--loop-seconds must be a multiple of {loop_unit_s:g} s for these rates")
        if n > avail:
            report["error"] = (f"{loop_seconds:g} s loop needs more material "
                               f"(passing takes allow {avail / FS:.0f} s)")
            return None, report
    else:
        n = (avail // unit) * unit
        if n == 0:
            report["error"] = f"passing takes give {avail / FS:.0f} s; need at least {loop_unit_s:g} s"
            return None, report

    order = [int(k) for k in np.random.default_rng(seed).permutation(len(clips))]
    bed = _finish(clips, order, n, carriers, do_condition)
    report.update(built=True, order=[names[k] for k in order], loop_s=n / FS)
    bad = finished_problems(bed)
    if bad:                                              # reported, not fatal: --split drops such loops
        report["finished_problems"] = bad
        print(f"  !! the finished loop fails the screens: {'; '.join(bad)}")
    return bed, report


def finished_problems(bed, window_s=60.0):
    """Screen a finished loop (after joining and conditioning) in window_s windows, circularly.
    The takes are screened raw, and conditioning can still add transients: it gave some river
    takes that measured 0/min raw +6 to +26/min (2026-11-20). Returns the reasons found."""
    n = int(window_s * FS)
    if len(bed) < n:
        return list(screen(bed)["reasons"])
    out = []
    for i in range(0, len(bed), n):
        w = np.concatenate([bed[i:], bed[:max(0, i + n - len(bed))]])[:n] if i + n > len(bed) else bed[i:i + n]
        out += [f"{q} at {i / FS:.0f} s" for q in screen(w)["reasons"]]
    return out


def _source(path):
    """(recording, offset) a take was cut or generated from, for ordering; (path, 0) if unknown."""
    try:
        with open(os.path.splitext(path)[0] + ".json") as fh:
            f = json.load(fh).get("init_from") or {}
        return f.get("file") or path, float(f.get("offset_s") or 0)
    except (OSError, ValueError):
        return path, 0.0


def build_split(paths, carriers, loop_unit_s, split_s, do_condition=True):
    """Several split_s-long loops from one scene's passing takes, each from its own takes (none
    shared), so a long bed becomes a few short, different loops. Takes are grouped in recording
    order: the windows of a --engine ref recording are back to back, so a loop's joins fall
    between neighbouring stretches of one recording where it can. Takes left over that are too
    short for one more loop are unused.
    Returns ([(bed, [take names])], report)."""
    xf = int(XFADE_S * FS)
    n = int(round(split_s * FS))
    if n % int(round(loop_unit_s * FS)):
        raise SystemExit(f"--split must be a multiple of {loop_unit_s:g} s for these rates")
    clips, names, report = screen_and_match(paths, carriers, do_condition)
    where = {os.path.basename(p): _source(p) for p in paths}
    idx = sorted(range(len(clips)), key=lambda k: where.get(names[k], (names[k], 0.0)))
    groups, cur = [], []
    for k in idx:
        cur.append(k)
        if loopable_samples([clips[j] for j in cur], xf) >= n:
            groups.append(cur)
            cur = []
    beds, dropped = [], {}
    for g in groups:
        b, o = _finish(clips, g, n, carriers, do_condition), [names[k] for k in g]
        bad = finished_problems(b)
        if bad:
            dropped[" + ".join(o)] = bad
            print(f"  DROP   loop from {' + '.join(o)}: {'; '.join(bad)}")
        else:
            beds.append((b, o))
    report.update(built=bool(beds), loop_s=n / FS, variants=[o for _, o in beds],
                  unused=[names[k] for k in cur])
    if dropped:
        report["dropped_loops"] = dropped
    if not beds:
        report["error"] = (f"passing takes give {loopable_samples(clips, xf) / FS:.0f} s; "
                           f"need {split_s:g} s for one loop")
    return beds, report

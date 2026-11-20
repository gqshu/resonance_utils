#!/usr/bin/env python3
"""
ss gen inducer (alias: ss_gen): nap-inducer soundscapes for the power-nap app.

For each scenario:
  1. BED    ElevenLabs takes (cached in <out>/<scenario>/raw/, never re-bought) -> screen
            -> join into a seamless loop -> flatten, cap transients, carrier notch, normalise
  2. SWELL  bake one swell per rate (default 0.1, 0.2, 0.25 Hz) into the same bed.
            The loop length is a whole number of waves at every rate (multiple of 20 s
            for the defaults), so every output loops sample-exactly and all rates share the
            same texture: the app can switch rates at a trough.
  3. TONE   (--tone) a sedative ambient drone that breathes with the same swell, kept clear
            of the binaural carrier and free of 1-12 Hz beating.

The binaural beat / AM is NOT added here: the app's music driver adds it on its own bus.

Outputs, per scenario, in <out>/inducer/<scenario>/:
  <scenario>_flat.wav                    conditioned flat bed (no swell, no tone)
  <scenario>_<rate>Hz[_tone].wav         one loop per rate, 48 kHz / 24-bit
  <scenario>_<rate>Hz[_tone].json        parameters + QA
  <scenario>_<rate>Hz[_tone]_envelope.csv  swell envelope at 20 Hz, for EEG alignment
  report.json                            screening results for every take
  with --stems: ..._bed.wav and ..._tonelayer.wav at the same scale as the mix

Examples
  ss gen inducer --list
  ss gen inducer train_smooth rain_soft --takes 8 --tone
  ss gen inducer all --dry-run                          # show the API calls, spend nothing
  ss gen inducer --from-dir my_takes/ --name cabin --profile train_smooth --tone
  ss gen inducer all --probe 2 --dry-run                # what a prompt probe would buy, and its cost
  ss gen inducer rain_soft --probe 3                    # 3 takes per prompt variant -> pass-rate table
  ss gen inducer rain_soft --variant c --tone           # build from a variant that won the probe
"""

import argparse
import json
import os
import sys

import numpy as np

from . import bed as bedmod
from . import swell, tone
from ..core.audio_io import FS, PEAK_CEIL_DBFS, TARGET_RMS_DBFS, rms, write_wav24
from ..core.audio_io import load_any
from ..core.layout import type_dir as _type_dir
from ..core.slots import DEFAULT_CARRIERS_HZ, clean_carriers
from .scenarios import (DEFAULT_PROFILE, SCENARIOS, profile_for, prompt_for, sa3_precap_db, sa3_spec, sa3_tone_db,
                        sa3_variant_names, variant_names)

TYPE = "inducer"
CREDITS_PER_SECOND = 40      # ElevenLabs sound effects with an explicit duration (docs, 2026-11)


def type_dir(root):
    return _type_dir(root, TYPE)


def rate_tag(r):
    return f"{r:g}".replace(".", "p") + "Hz"


def scale_to_target(mix):
    """Gain that puts mix at the target RMS without exceeding the peak ceiling."""
    k = 10 ** (TARGET_RMS_DBFS / 20) / (rms(mix) + 1e-12)
    pk = np.max(np.abs(mix)) * k
    ceil = 10 ** (PEAK_CEIL_DBFS / 20)
    return k * (ceil / pk if pk > ceil else 1.0)


def qa(x):
    """Light QA for a finished output: only the modulation measures (the takes were already
    fully screened, and the bed is notched), against the same 3-surrogate baseline."""
    m = x.mean(1)
    a = bedmod.mod_measures(m)
    bs = [bedmod.mod_measures(bedmod.surrogate(m, seed=k)) for k in range(3)]
    b = {k: float(np.mean([q[k] for q in bs])) for k in ("eeg", "swell", "trans", "peak")}
    s = dict(eeg_band_excess_db=round(a["eeg"] - b["eeg"], 2), eeg_peak_excess_db=round(a["peak"] - b["peak"], 1),
             eeg_peak_hz=round(a["peak_hz"], 2), swell_excess_db=round(a["swell"] - b["swell"], 2),
             transients_excess_per_min=round(max(0.0, a["trans"] - b["trans"]), 1))
    out = dict(rms_dbfs=round(20 * np.log10(rms(x) + 1e-12), 1), **s)
    out["peak_dbfs"] = round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 2)
    out["loop_seam_jump"] = round(float(np.max(np.abs(x[0] - x[-1]))), 6)
    out["eeg_band_clean"] = (s["eeg_band_excess_db"] <= bedmod.THRESH["eeg_band_excess_db"]
                             and s["eeg_peak_excess_db"] <= bedmod.THRESH["eeg_peak_excess_db"])
    return out


def write_envelope_csv(path, sw):
    step = FS // 20
    with open(path, "w") as fh:
        fh.write("time_s,wave_phase,envelope_0to1,gain_db,pan\n")
        for i in range(0, len(sw["env"]), step):
            fh.write(f"{i / FS:.3f},{sw['phase'][i] % 1:.4f},{sw['env'][i]:.4f},"
                     f"{sw['gain_db'][i]:.3f},{sw['pan'][i]:.4f}\n")


LOCAL = ("sa3", "ref")    # engines built from the real recordings in scenarios.SA3_SCENES


def out_name(name, a):
    """SA3 builds live next to the ElevenLabs ones, as <scenario>_sa3, so the two can be compared
    (and deployed side by side)."""
    return f"{name}_{a.engine}" if a.engine in LOCAL else name


def variants_for(name, a):
    return sa3_variant_names(name) if a.engine in LOCAL else variant_names(name)


def describe(name, v, a):
    if a.engine not in LOCAL:
        return prompt_for(name, v)
    sp = sa3_spec(name, v)
    if a.engine == "ref":
        return "real recordings: " + ", ".join(os.path.basename(r) for r in sp["init"])
    return sp["prompt"] + (f"  [cfg {sp['cfg']:g}, negative: {sp['negative']}]" if sp.get("negative") else "")


def take_seconds(a):
    return a.take_seconds or (60.0 if a.engine in LOCAL else 30.0)


def make_takes(name, v, raw_dir, n, a):
    if a.engine == "ref":                            # every window of the references; n is unused
        return bedmod.ref_takes(sa3_spec(name, v)["init"], raw_dir, duration=take_seconds(a), dry_run=a.dry_run)
    if a.engine == "sa3":
        return bedmod.sa3_generate_takes(sa3_spec(name, v), raw_dir, n, duration=take_seconds(a), dit=a.sa3_dit,
                                         steps=a.sa3_steps, trim_s=a.sa3_trim, dry_run=a.dry_run)
    return bedmod.generate_takes(prompt_for(name, v), raw_dir, n, duration=take_seconds(a), influence=a.influence,
                                 model=a.model, output_format=a.output_format, dry_run=a.dry_run)


def probe_scenario(name, a, carriers):
    """Generate a.probe takes per prompt variant, screen them, print a pass-rate table.
    Takes stay in the cache, so a probe's passing takes count towards a later build."""
    label = out_name(name, a)
    raw_dir = os.path.join(a.out, label, "raw")
    model = {"sa3": f"sa3 {a.sa3_dit}", "ref": "real recordings"}.get(a.engine, a.model)
    print(f"\n=== probe {label}: {a.probe} take(s) per variant, model {model} ===")
    rows, buy = [], 0
    for v in variants_for(name, a):
        prompt = describe(name, v, a)
        if a.engine == "elevenlabs":
            buy += max(0, a.probe - len(bedmod.cached_takes(raw_dir, prompt, a.model)))
        print(f"[{v}] {prompt[:100]}...")
        paths = make_takes(name, v, raw_dir, a.probe, a)
        if a.dry_run:
            continue
        res = []
        for i, p in enumerate(paths):
            r, _ = bedmod.screen_take_cached(p, lambda: load_any(p, seed=i), carriers, not a.no_condition)
            r["file"] = os.path.basename(p)
            res.append(r)
            print(f"    {'PASS' if r['verdict'] == 'pass' else 'REJECT':6s} {r['file']}  "
                  f"eeg {r['eeg_band_excess_db']:+.2f}  trans {r['transients_excess_per_min']:+.0f}/min  "
                  f"swell {r['swell_excess_db']:+.2f}/{r['brightness_swell_excess_db']:+.2f}  "
                  + "; ".join(r["reasons"]))
        rows.append(dict(variant=v, prompt=prompt, takes=res,
                         passed=sum(r["verdict"] == "pass" for r in res)))
    if a.dry_run:
        if a.engine == "elevenlabs":
            print(f"  would buy {buy} take(s) of {take_seconds(a):g} s = about "
                  f"{buy * take_seconds(a) * CREDITS_PER_SECOND:,.0f} credits")
        return True
    med = lambda rs, k: float(np.median([r[k] for r in rs])) if rs else float("nan")
    print(f"\n  {'variant':8s} {'pass':>6s}  {'eeg':>6s} {'trans/min':>9s} {'swell':>6s} {'bright':>6s}")
    for row in rows:
        t = row["takes"]
        print(f"  {row['variant']:8s} {row['passed']:>2d}/{len(t):<3d}  {med(t, 'eeg_band_excess_db'):+6.2f} "
              f"{med(t, 'transients_excess_per_min'):+9.0f} {med(t, 'swell_excess_db'):+6.2f} "
              f"{med(t, 'brightness_swell_excess_db'):+6.2f}")
    print(f"  (medians; limits: eeg {bedmod.THRESH['eeg_band_excess_db']}, trans "
          f"{bedmod.THRESH['transient_max_per_min']:g}/min, swell/bright "
          f"{bedmod.THRESH['swell_after_max_db']} after conditioning)")
    os.makedirs(os.path.join(a.out, label), exist_ok=True)
    with open(os.path.join(a.out, label, "probe_report.json"), "w") as fh:
        json.dump(dict(scenario=name, model=model, thresholds=bedmod.THRESH, variants=rows),
                  fh, indent=2, default=str)
    return True


def write_bed_outputs(name, bed, order, report, a, carriers, profile, tone_db, out_dir):
    """Write one bed's flat loop and its swell (+ tone) renders. Returns the output file names."""
    n = len(bed)
    write_wav24(os.path.join(out_dir, f"{name}_flat.wav"), bed)
    print(f"  bed: {n / FS:g} s loop from {len(order)} takes -> {name}_flat.wav")
    outputs = []

    stems = None
    if a.tone:
        print("[3/3] tone layer (rendered once, shaped by each rate's swell below)")
        fund, upper, tinfo = tone.render_tone_stems(n, a.tone_root, a.tone_chord, carriers, a.seed)
        stems = (fund, upper)
        report["tone"] = tinfo
        print(f"  partials {tinfo['partials_hz']}" +
              (f"; dropped {[d['f'] for d in tinfo['dropped']]}" if tinfo["dropped"] else ""))

    print(f"[2/3] swells at {', '.join(f'{r:g}' for r in a.rates)} Hz")
    for r in a.rates:
        y, sw = swell.apply_swell(bed, r, profile, a.period_jitter, a.depth_jitter, a.seed)
        tag = f"{name}_{rate_tag(r)}" + ("_tone" if stems else "")
        t = None
        if stems:
            t = tone.mix_tone(stems[0], stems[1], sw["env"], sw["params"]["depth_db"] * a.tone_swell)
            t *= rms(y) * 10 ** (tone_db / 20) / (rms(t) + 1e-12)
            mix = y + t
        else:
            mix = y
        k = scale_to_target(mix)
        mix = mix * k
        write_wav24(os.path.join(out_dir, tag + ".wav"), mix)
        write_envelope_csv(os.path.join(out_dir, tag + "_envelope.csv"), sw)
        if a.stems and t is not None:
            write_wav24(os.path.join(out_dir, tag + "_bed.wav"), y * k)
            write_wav24(os.path.join(out_dir, tag + "_tonelayer.wav"), t * k)
        q = qa(mix)
        meta = dict(scenario=name, file=tag + ".wav", loop_s=n / FS, fs=FS, bits=24, carriers_hz=carriers,
                    swell=sw["params"], profile=profile,
                    tone=(dict(level_db_vs_bed=tone_db, swell_db=round(sw["params"]["depth_db"] * a.tone_swell, 2),
                               **report["tone"]) if stems else None),
                    qa=q)
        with open(os.path.join(out_dir, tag + ".json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        outputs.append(tag + ".wav")
        flag = "" if q["eeg_band_clean"] else "  !! EEG-band modulation above threshold"
        print(f"  {tag}.wav  {sw['params']['waves']} waves, depth {sw['params']['depth_db']:g} dB, "
              f"eeg excess {q['eeg_band_excess_db']:+.2f} dB, seam {q['loop_seam_jump']:.4f}{flag}")

    return outputs


def run_scenario(name, a, carriers, prompts=(), profile=None, paths=None, variants=None, tone_db=None):
    """variants: (scenario, [variant, ...]) to generate takes from; prompts is what the report
    records. Without variants, paths are the takes (--from-dir)."""
    tone_db = a.tone_db if tone_db is None else tone_db
    out_dir = os.path.join(a.out, name)
    raw_dir = os.path.join(out_dir, "raw")
    unit = swell.loop_unit_s(a.rates)

    print(f"\n=== {name} ===")
    print("[1/3] bed")
    if paths is None:
        paths = []
        scen, vs = variants
        for v in vs:                                   # --takes applies to each variant
            paths += make_takes(scen, v, raw_dir, a.takes, a)
        if a.dry_run:
            return True
    os.makedirs(out_dir, exist_ok=True)
    if a.split:
        beds, report = bedmod.build_split(paths, carriers, unit, a.split, do_condition=not a.no_condition)
        beds = [(f"{name}_v{k + 1}", b, o) for k, (b, o) in enumerate(beds)]
    else:
        bed, report = bedmod.build_bed(paths, carriers, unit, a.loop_seconds, a.seed, a.min_pass,
                                       do_condition=not a.no_condition)
        beds = [(name, bed, report.get("order"))] if bed is not None else []
    report.update(name=name, prompts=list(prompts),
                  model={"sa3": f"sa3 {a.sa3_dit}", "ref": "real recordings"}.get(a.engine, a.model), profile=profile, carriers=carriers, rates=a.rates,
                  thresholds=bedmod.THRESH)
    if a.engine == "sa3":
        report.update(engine="sa3", sa3=dict(dit=a.sa3_dit, steps=a.sa3_steps, trim_s=a.sa3_trim))
    if a.engine in LOCAL and bedmod.PRECAP_DB:
        report["precap_db"] = bedmod.PRECAP_DB
    if a.engine == "ref":
        report["engine"] = "ref"
    if not beds:
        print(f"  {report['error']}")
        with open(os.path.join(out_dir, "report.json"), "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        return False
    outputs = []
    for loop_name, bed, order in beds:
        outputs += write_bed_outputs(loop_name, bed, order, report, a, carriers, profile, tone_db, out_dir)
    report["outputs"] = outputs
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return True


def add_args(ap):
    """Register the inducer's `gen` arguments on ap (shared by `ss gen inducer` and `ss_gen`)."""
    ap.add_argument("scenarios", nargs="*", help="scenario names, or 'all' (see --list)")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--from-dir", help="use existing WAV/MP3 takes instead of calling the API")
    ap.add_argument("--name", help="output name for --from-dir (default: the folder name)")
    ap.add_argument("--profile", default=None, choices=list(SCENARIOS),
                    help="swell profile for --from-dir (default: a generic one)")
    ap.add_argument("--out", default="build", help="build root; output goes to <out>/inducer/<scenario>/")
    g = ap.add_argument_group("bed (ElevenLabs)")
    g.add_argument("--takes", type=int, default=8, help="takes per scenario (cached takes count)")
    g.add_argument("--engine", choices=("elevenlabs", "sa3", "ref"), default="elevenlabs",
                   help="where takes come from: the ElevenLabs API, Stable Audio 3 run locally from the "
                        "scene's real recordings (SA3_DIR), or those recordings themselves, cut into "
                        "windows (builds go to <scenario>_sa3 / <scenario>_ref)")
    g.add_argument("--take-seconds", type=float, default=None, help="default 30 (ElevenLabs), 60 (sa3)")
    g.add_argument("--influence", type=float, default=0.65)
    g.add_argument("--model", default="eleven_text_to_sound_v3",
                   help="eleven_text_to_sound_v3 (newest) or _v2 (deprecated, removed 2027-03-01)")
    g.add_argument("--output-format", default="pcm_48000")
    g.add_argument("--min-pass", type=int, default=3, help="minimum passing takes to build a bed")
    g.add_argument("--loop-seconds", type=float, default=None,
                   help="loop length (default: longest whole multiple of the wave unit)")
    g.add_argument("--split", type=float, metavar="SECONDS", default=None,
                   help="build several loops this long from one scene, each from its own takes "
                        "(<scenario>_v1, _v2, ...) instead of one long loop; e.g. 100")
    g.add_argument("--no-condition", action="store_true")
    g.add_argument("--flatten-s", type=float, default=None,
                   help="conditioning level window (s): 0.5 flattens all motion below ~2 Hz, 3 only the "
                        "swell band, keeping the scene's own gusts and washes. Default 0.5 (ElevenLabs), 3 (sa3)")
    g.add_argument("--dry-run", action="store_true", help="show the API calls, spend nothing")
    g.add_argument("--variant", default=None,
                   help="prompt variant(s) to build from, e.g. 'c' or 'a,b,c' (see --list); "
                        "--takes applies to each")
    g.add_argument("--probe", type=int, metavar="N",
                   help="make N takes per prompt variant, report pass rates, build nothing")
    g = ap.add_argument_group("bed (Stable Audio 3, --engine sa3)")
    g.add_argument("--sa3-dit", choices=("medium", "sm-sfx", "sm-music"), default="medium")
    g.add_argument("--sa3-steps", type=int, default=8)
    g.add_argument("--sa3-trim", type=float, default=3.0,
                   help="seconds rendered extra and cut at each end of a take (SA3 clips fade in/out)")
    g = ap.add_argument_group("swell")
    g.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.25])
    g.add_argument("--period-jitter", type=float, default=0.03, help="wave period drift (fraction)")
    g.add_argument("--depth-jitter", type=float, default=0.10, help="per-wave depth variation (fraction)")
    g = ap.add_argument_group("tone layer")
    g.add_argument("--tone", action="store_true", help="add the ambient tone layer")
    g.add_argument("--tone-db", type=float, default=-12.0, help="tone RMS relative to the bed (dB)")
    g.add_argument("--tone-root", type=float, default=tone.DEFAULT_ROOT_HZ, help="root (Hz)")
    g.add_argument("--tone-chord", default=tone.DEFAULT_CHORD, help="ratios over the root")
    g.add_argument("--tone-swell", type=float, default=0.6, help="tone swell depth as a fraction of the bed's")
    g.add_argument("--stems", action="store_true", help="also write bed and tone stems")
    ap.add_argument("--carrier", type=float, nargs="+", default=list(DEFAULT_CARRIERS_HZ),
                    help="every binaural carrier the app may use (Hz): the bed is notched and screened "
                         "at each, and the tone layer avoids them all; 0 = none")
    ap.add_argument("--seed", type=int, default=1)


def list_scenarios(engine="elevenlabs"):
    class A:
        pass
    a = A()
    a.engine = engine
    for k in SCENARIOS:
        for i, v in enumerate(variants_for(k, a)):
            print(f"{k if i == 0 else '':15s} [{v}] {describe(k, v, a)[:95]}...")


def run(a, ap):
    """Run a parsed `gen` command. a.out is the build root; this type writes to <out>/inducer/."""
    a.out = type_dir(a.out)
    if a.list:
        list_scenarios(a.engine)
        return 0
    carriers = clean_carriers(a.carrier)
    bedmod.FLATTEN_S = a.flatten_s or (3.0 if a.engine in LOCAL else 0.5)
    ok = True
    if a.from_dir:
        paths = sorted(os.path.join(a.from_dir, p) for p in os.listdir(a.from_dir)
                       if p.lower().endswith((".wav", ".mp3")))
        if not paths:
            raise SystemExit(f"no WAV/MP3 files in {a.from_dir}")
        name = a.name or os.path.basename(os.path.normpath(a.from_dir))
        profile = profile_for(a.profile) if a.profile else dict(DEFAULT_PROFILE)
        ok = run_scenario(name, a, carriers, profile=profile, paths=paths)
    else:
        names = list(SCENARIOS) if a.scenarios == ["all"] else a.scenarios
        if not names:
            ap.error("name at least one scenario (or 'all'), or use --from-dir; see --list")
        unknown = [s for s in names if s not in SCENARIOS]
        if unknown:
            ap.error(f"unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}")
        for s in names:
            if a.probe:
                ok &= probe_scenario(s, a, carriers)
                continue
            vs = [v.strip() for v in (a.variant or ("r" if a.engine in LOCAL else "a")).split(",") if v.strip()]
            bad = [v for v in vs if v not in variants_for(s, a)]
            if bad:
                ap.error(f"{s} has no variant(s) {bad}; it has {variants_for(s, a)}")
            bedmod.PRECAP_DB = sa3_precap_db(s) if a.engine in LOCAL else None
            ok &= run_scenario(out_name(s, a), a, carriers, prompts=[describe(s, v, a) for v in vs],
                               profile=profile_for(s), variants=(s, vs),
                               tone_db=sa3_tone_db(s, a.tone_db) if a.engine in LOCAL else None)
    return 0 if ok else 2


def main(argv=None):
    """Legacy `ss_gen` entry point: same as `ss gen inducer`."""
    ap = argparse.ArgumentParser(prog="ss_gen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    return run(ap.parse_args(argv), ap)


if __name__ == "__main__":
    sys.exit(main())

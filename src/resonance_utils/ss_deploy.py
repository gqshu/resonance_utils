"""ss_deploy: collect every ready soundscape from an ss_gen build into one directory, with a
single manifest.json for the resonance app.

    ss_deploy                          # build/ -> deploy/, every ready scenario
    ss_deploy car_night rain_soft      # only these
    ss_deploy --stems                  # also ship bed and tone stems (app-side tone level)
    ss_deploy --clean --out ../app/assets/soundscapes

A scenario is ready when its report.json says built, every listed output and its JSON exist,
every output's QA says the EEG band is clean, and the outputs agree on loop length and carriers.
Anything else is skipped with the reason. Nothing in build/ is modified.

Layout:
    <out>/manifest.json
    <out>/<scenario>/<scenario>_flat.wav                 conditioned bed, no swell, no tone
    <out>/<scenario>/<scenario>_<rate>Hz[_tone].wav      one seamless loop per swell rate
    <out>/<scenario>/<scenario>_<rate>Hz[_tone]_envelope.csv
    <out>/<scenario>/..._bed.wav, ..._tonelayer.wav      with --stems
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import sys

from . import __version__

SCHEMA_VERSION = 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wave_marks(csv_path):
    """Trough and crest times (s) of the swell, from the envelope CSV. Troughs are where the
    wave phase wraps (envelope 0): the app should switch rates or scenarios there."""
    t, ph, env = [], [], []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["time_s"]))
            ph.append(float(row["wave_phase"]))
            env.append(float(row["envelope_0to1"]))
    troughs = [t[0]] + [t[i] for i in range(1, len(t)) if ph[i] < ph[i - 1]]
    bounds = [i for i in range(len(t)) if i == 0 or ph[i] < ph[i - 1]] + [len(t)]
    crests = [t[max(range(a, b), key=lambda i: env[i])] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return [round(v, 3) for v in troughs], [round(v, 3) for v in crests]


def check_ready(sdir, name):
    """Returns (report, outputs) for a ready scenario, or raises ValueError(reason)."""
    rp = os.path.join(sdir, "report.json")
    if not os.path.exists(rp):
        raise ValueError("no report.json (never built)")
    with open(rp) as fh:
        rep = json.load(fh)
    if not rep.get("built"):
        raise ValueError(rep.get("error") or "last build failed")
    if not rep.get("outputs"):
        raise ValueError("report lists no outputs")
    outs = []
    for wav in rep["outputs"]:
        base = os.path.splitext(wav)[0]
        paths = {k: os.path.join(sdir, base + suf) for k, suf in
                 (("wav", ".wav"), ("json", ".json"), ("envelope", "_envelope.csv"))}
        missing = [os.path.basename(p) for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise ValueError("missing " + ", ".join(missing))
        with open(paths["json"]) as fh:
            meta = json.load(fh)
        if str(meta.get("qa", {}).get("eeg_band_clean")) != "True":
            raise ValueError(f"{wav}: QA says the EEG band is not clean")
        outs.append((paths, meta))
    for _, m in outs:
        if "carriers_hz" not in m:
            raise ValueError("built before multi-carrier support; rebuild with ss_gen")
        m["carriers_hz"] = tuple(m["carriers_hz"])
    for key in ("loop_s", "carriers_hz", "fs"):
        if len({m[key] for _, m in outs}) > 1:
            raise ValueError(f"outputs disagree on {key} (stale files from an older build?)")
    if abs(outs[0][1]["loop_s"] - rep.get("loop_s", outs[0][1]["loop_s"])) > 1e-6:
        raise ValueError("outputs are from an older build than report.json")
    flat = os.path.join(sdir, f"{name}_flat.wav")
    return rep, outs, (flat if os.path.exists(flat) else None)


def file_entry(src, dst_dir, out_root):
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dict(path=os.path.relpath(dst, out_root).replace(os.sep, "/"),
                bytes=os.path.getsize(dst), sha256=sha256(dst))


def deploy_scenario(name, rep, outs, flat, out_root, stems):
    dst_dir = os.path.join(out_root, name)
    os.makedirs(dst_dir, exist_ok=True)
    first = outs[0][1]
    loops = []
    for paths, meta in outs:
        troughs, crests = wave_marks(paths["envelope"])
        loop = dict(rate_hz=meta["swell"]["rate_hz"], tone=meta.get("tone") is not None,
                    audio=file_entry(paths["wav"], dst_dir, out_root),
                    envelope_csv=file_entry(paths["envelope"], dst_dir, out_root),
                    swell=meta["swell"], troughs_s=troughs, crests_s=crests, qa=meta.get("qa", {}))
        if stems:
            base = paths["wav"][:-4]
            st = {}
            for key, suf in (("bed", "_bed.wav"), ("tone", "_tonelayer.wav")):
                if os.path.exists(base + suf):
                    st[key] = file_entry(base + suf, dst_dir, out_root)
            if st:
                loop["stems"] = st
        loops.append(loop)
    loops.sort(key=lambda d: d["rate_hz"])
    tone = next((m.get("tone") for _, m in outs if m.get("tone")), None)
    if tone:
        tone = {k: v for k, v in tone.items() if k not in ("swell_db",)}
    return dict(
        id=name,
        title=name.replace("_", " ").title(),
        loop_s=first["loop_s"],
        rates_hz=[d["rate_hz"] for d in loops],
        flat=file_entry(flat, dst_dir, out_root) if flat else None,
        loops=loops,
        tone=tone,
        profile=rep.get("profile"),
        source=dict(generator="elevenlabs", model=rep.get("model"), prompts=rep.get("prompts", []),
                    takes=rep.get("order", [])),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ss_deploy", description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios", nargs="*", help="scenario names (default: every ready one)")
    ap.add_argument("--build", default="build", help="ss_gen output directory (default build/)")
    ap.add_argument("--out", default="deploy", help="deploy directory (default deploy/)")
    ap.add_argument("--stems", action="store_true", help="also ship bed and tone-layer stems")
    ap.add_argument("--clean", action="store_true",
                    help="empty --out first (only if it holds a previous ss_deploy manifest)")
    a = ap.parse_args(argv)

    names = a.scenarios or sorted(d for d in os.listdir(a.build)
                                  if os.path.isfile(os.path.join(a.build, d, "report.json")))
    if not names:
        sys.exit(f"no built scenarios in {a.build}/")

    ready, skipped = [], {}
    for name in names:
        try:
            ready.append((name,) + check_ready(os.path.join(a.build, name), name))
        except (ValueError, OSError) as e:
            skipped[name] = str(e)
    for name, why in skipped.items():
        print(f"  SKIP   {name:<16s} {why}")
    if not ready:
        sys.exit("nothing ready to deploy")
    carriers = {outs[0][1]["carriers_hz"] for _, _, outs, _ in ready}
    if len(carriers) > 1:
        sys.exit(f"scenarios were built for different carrier sets {sorted(carriers)}; "
                 "rebuild them with the same --carrier list")
    fs = {outs[0][1]["fs"] for _, _, outs, _ in ready}

    man_path = os.path.join(a.out, "manifest.json")
    if a.clean and os.path.isdir(a.out):
        if os.listdir(a.out) and not os.path.exists(man_path):
            sys.exit(f"--clean refused: {a.out}/ has files but no manifest.json, so it isn't an ss_deploy directory")
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)

    scen = []
    for name, rep, outs, flat in ready:
        s = deploy_scenario(name, rep, outs, flat, a.out, a.stems)
        mb = sum(e["bytes"] for l in s["loops"] for e in [l["audio"], *l.get("stems", {}).values()])
        mb += s["flat"]["bytes"] if s["flat"] else 0
        print(f"  DEPLOY {name:<16s} {s['loop_s']:g} s loop, rates {s['rates_hz']}, {mb / 1e6:.0f} MB")
        scen.append(s)

    manifest = dict(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        generator=f"resonance_utils {__version__}",
        audio=dict(container="wav", sample_rate_hz=fs.pop(), bit_depth=24, channels=2,
                   loudness_rms_dbfs=-26.0, peak_ceiling_dbfs=-3.0, seamless_loop=True),
        carriers_hz=list(carriers.pop()),
        notes=[
            "Every loop is seamless: play it with gapless looping, never re-encode to MP3.",
            "All rates of one scenario share the same bed: crossfade between them at a trough "
            "(troughs_s) for an inaudible switch.",
            "Every soundscape works with any carrier in carriers_hz: the bed is notched -6 dB in a "
            "1/3 octave around each, takes with tonal hum near any of them were rejected, and the tone "
            "layer keeps every partial at least 20 Hz from all of them. Run the binaural carrier at "
            "one of these frequencies; any other may beat against the tone layer.",
            "No beat or AM is baked in: the app's music driver adds it.",
            "envelope_csv columns: time_s, wave_phase, envelope_0to1, gain_db, pan, at 20 Hz.",
        ],
        scenarios=scen,
    )
    if os.path.exists(man_path) and not a.clean:
        with open(man_path) as fh:
            old = {s["id"]: s for s in json.load(fh).get("scenarios", [])}
        kept = [s for sid, s in old.items() if sid not in {s["id"] for s in scen}
                and all(os.path.exists(os.path.join(a.out, l["audio"]["path"])) for l in s["loops"])]
        manifest["scenarios"] = sorted(scen + kept, key=lambda s: s["id"])
        if kept:
            print(f"  kept from the previous manifest: {', '.join(s['id'] for s in kept)}")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs_ in os.walk(a.out) for f in fs_)
    print(f"{len(manifest['scenarios'])} scenario(s) in {a.out}/ ({total / 1e6:.0f} MB) -> {man_path}")


if __name__ == "__main__":
    main()

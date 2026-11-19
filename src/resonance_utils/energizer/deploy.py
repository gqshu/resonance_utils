"""ss deploy energizer: collect ready energizer beds into deploy/energizer/ with one manifest.json
for the app's music engine.

    ss deploy energizer                         every ready bed that passes the QA gate
    ss deploy energizer keys_d118 piano_g122    only these (batch names or bed ids)
    ss deploy energizer --allow-qa-fail ...     also beds whose lever_compatible tag is false
    ss deploy energizer --stems                 also ship the six stems next to the final mix

A bed is ready when its manifest says built (and it is not a reference-only folder), every stem
and sidecar exists, and every stem's sha256 matches the manifest. The QA gate is the plan's
`lever_compatible` tag (§9); with --allow-qa-fail a failing bed is deployed and keeps its tags
and failed checks in the manifest, so the app can tell. Nothing in build/ is modified.

Each bed ships as one finished soundscape, <name>.wav (the bed's final mix, the one QA measured):
the app adds AM and monaural beats on its own bus, as with the inducer pack.

Layout:
    <out>/energizer/manifest.json
    <out>/energizer/<name>/<name>.wav                       the final mix
    <out>/energizer/<name>/{tempo_map,arc,chords}.json
    <out>/energizer/<name>/stems/*.wav                      with --stems
    <out>/index.json
"""

import glob
import json
import os

from .. import __version__
from ..core.deploy import file_entry, merge_previous, now_utc, prepare_out, sha256, tree_bytes, write_index
from ..core.layout import type_dir

TYPE = "energizer"
SCHEMA_VERSION = 1
SIDECARS = ("tempo_map.json", "arc.json", "chords.json")
NOTES = [
    "Each bed is one finished soundscape (`audio`), 48 kHz / 24-bit stereo, 4 min, not a loop: the "
    "tempo ramp and the arc (entry -> ramp -> plateau -> hand-off) are baked in. Play it once, start to end.",
    "No AM or beat is baked in: add them on the app's own bus. Monaural beats: use mb_slot.carrier_hz with "
    "beats_hz; the mix is notched >= 18 dB in mb_keep_clear_hz, so the beat tones sit clear.",
    "AM applied to this whole mix also modulates the music and percussion (the plan's design applies AM to "
    "the pad stem only; ship with --stems for that). qa.native_mod_db.mix is the mix's own modulation.",
    "tempo_map.json gives bpm at 100 Hz and every beat / 16th / 32nd time; tempo-lock AM to it. "
    "arc.json gives the phase times; sections (hybrid beds) follow the AI music's structure.",
    "ai_generated beds contain Eleven Music output: show the AI-content label. Confirm in writing that "
    "in-app streaming of modified tracks is covered by the ElevenLabs plan before shipping.",
    "qa.provisional is true: every threshold is an engineering hypothesis until the M6 runs.",
]


def batch_names(build):
    """{bed_id: name} from every batch summary in the build folder."""
    names = {}
    for p in glob.glob(os.path.join(build, "batch_*.json")):
        try:
            for r in json.load(open(p)):
                if r.get("bed_id"):
                    names[r["bed_id"]] = r["name"]
        except (OSError, ValueError):
            pass
    return names


def check_ready(bdir):
    mp = os.path.join(bdir, "manifest.json")
    if not os.path.exists(mp):
        raise ValueError("no manifest.json (never built)")
    m = json.load(open(mp))
    if m.get("kind") == "reference":
        raise ValueError("reference mix only (no hybrid render)")
    if not m.get("built"):
        raise ValueError("last build failed")
    for k, s in m["stems"].items():
        p = os.path.join(bdir, s["path"])
        if not os.path.exists(p):
            raise ValueError(f"missing {s['path']}")
        if sha256(p) != s["sha256"]:
            raise ValueError(f"{s['path']} changed since the build (sha256 mismatch)")
    missing = [f for f in SIDECARS + ("qa_report.json", "preview_mix.wav") if not os.path.exists(os.path.join(bdir, f))]
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    return m, json.load(open(os.path.join(bdir, "qa_report.json")))


def deploy_bed(bid, name, bdir, m, q, out, with_stems):
    dst = os.path.join(out, name)
    audio = file_entry(os.path.join(bdir, "preview_mix.wav"), dst, out, name=f"{name}.wav")
    stems = {k: dict(file_entry(os.path.join(bdir, s["path"]), os.path.join(dst, "stems"), out),
                     gain_db=s["gain_db"]) for k, s in m["stems"].items()} if with_stems else None
    side = {os.path.splitext(f)[0]: file_entry(os.path.join(bdir, f), dst, out) for f in SIDECARS}
    sections = None
    ap = os.path.join(bdir, "ai_analysis.json")
    if os.path.exists(ap):
        sections = [{k: s[k] for k in ("name", "role", "start_s", "end_s")}
                    for s in json.load(open(ap)).get("sections") or []]
    arc = json.load(open(os.path.join(bdir, "arc.json")))
    p = m["params"]
    return dict(
        id=name, bed_id=bid, title=name.replace("_", " "),
        length_s=m["length_s"], tempo=m["tempo"], key=f"{p['key_root']} {p['mode']}",
        phases_s=arc["phases"], sections=sections,
        style=dict(music=p["music_style"] if p["music_source"] != "none" else None,
                   music_variant=p.get("music_variant"), pulse=p["pulse_style"], pad=p["pad_timbre"],
                   texture=p["texture_source"]),
        audio=audio, sidecars=side, stems=stems,
        qa=dict(tags=q["tags"], checks=q["checks"], provisional=q.get("provisional", True),
                failed=[k for k, v in q["checks"].items() if not v],
                true_peak_dbfs=q["true_peak_dbfs"], phase_rms_dbfs=q["phase_rms_dbfs"],
                native_mod_db_mix=q["native_mod_db"]["mix"], mb_slot_residual_db_mix=q["mb_slot_residual_db"]["mix"]),
        ai_generated=m.get("ai_generated", False), licence=m.get("licence"),
        source=dict(params=p, generator=m.get("generator")),
    )


def add_args(ap):
    ap.add_argument("beds", nargs="*", help="batch names or bed ids (default: every ready bed)")
    ap.add_argument("--build", default="build", help="build root (reads build/energizer/)")
    ap.add_argument("--out", default="deploy", help="deploy root (writes deploy/energizer/)")
    ap.add_argument("--allow-qa-fail", action="store_true",
                    help="also deploy beds whose lever_compatible tag is false (their failed checks are recorded)")
    ap.add_argument("--stems", action="store_true", help="also ship the six stems next to the final mix")
    ap.add_argument("--clean", action="store_true",
                    help="empty deploy/energizer/ first (only if it holds a previous manifest)")


def run(a, ap=None):
    build, out = type_dir(a.build, TYPE), type_dir(a.out, TYPE)
    if not os.path.isdir(build):
        print(f"  no energizer beds in {build}/")
        return 1
    names = batch_names(build)
    by_name = {v: k for k, v in names.items()}
    all_ids = sorted(d for d in os.listdir(build) if os.path.isfile(os.path.join(build, d, "manifest.json")))
    want = [by_name.get(b, b) for b in a.beds] if a.beds else all_ids
    ready, skipped = [], {}
    for bid in want:
        label = names.get(bid, bid)
        try:
            m, q = check_ready(os.path.join(build, bid))
        except (ValueError, OSError) as e:
            skipped[label] = str(e)
            continue
        if not q["tags"].get("lever_compatible") and not a.allow_qa_fail:
            failed = [k for k, v in q["checks"].items() if not v]
            skipped[label] = f"QA gate: lever_compatible is false (failed: {', '.join(failed)}); --allow-qa-fail to ship"
            continue
        ready.append((bid, label, m, q))
    for k, why in skipped.items():
        print(f"  SKIP   {k:<14s} {why}")
    if not ready:
        print("  nothing ready to deploy")
        return 1
    if len({json.dumps(m["lever_slots"], sort_keys=True) for _, _, m, _ in ready}) > 1:
        raise SystemExit("beds were built against different lever slots; rebuild them with the same config")
    man_path = prepare_out(out, a.clean)
    beds = []
    for bid, name, m, q in ready:
        b = deploy_bed(bid, name, os.path.join(build, bid), m, q, out, a.stems)
        mb = (b["audio"]["bytes"] + sum(e["bytes"] for e in (b["stems"] or {}).values())) / 1e6
        flag = "" if q["tags"].get("lever_compatible") else f"  (QA failed: {', '.join(b['qa']['failed'])})"
        print(f"  DEPLOY {name:<14s} {b['key']:<9s} {b['tempo']['entry_bpm']:.0f}->{b['tempo']['plateau_bpm']:g} BPM, "
              f"{'mix + %d stems' % len(b['stems']) if b['stems'] else 'final mix'}, {mb:.0f} MB{flag}")
        beds.append(b)
    first = ready[0][2]
    manifest = dict(
        schema_version=SCHEMA_VERSION, generated_at=now_utc(), generator=f"resonance_utils {__version__}",
        type=TYPE,
        audio=dict(container="wav", sample_rate_hz=first["fs"], bit_depth=first["bits"], channels=first["channels"],
                   equal_length_stems=True, seamless_loop=False),
        lever_slots=first["lever_slots"], notes=NOTES, beds=beds,
    )
    present = lambda s: os.path.exists(os.path.join(out, s["audio"]["path"]))
    manifest["beds"], kept = merge_previous(man_path, beds, "beds", present, a.clean)
    if kept:
        print(f"  kept from the previous manifest: {', '.join(kept)}")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    write_index(a.out, manifest["generator"])
    print(f"{len(manifest['beds'])} bed(s) in {out}/ ({tree_bytes(out) / 1e9:.2f} GB) -> {man_path}")
    return 0

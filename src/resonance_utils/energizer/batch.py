"""`ss gen energizer --batch tracks.yaml`: render a list of beds end to end.

tracks.yaml is a list of dicts: any EnergizerParams field, plus an optional `name`. Unset fields
take the dataclass defaults, except music_source, which defaults to elevenlabs here. A track whose
AI music misses the key or tempo check is retried with the next seed, up to max_retries times
(each retry is a new composition). Other failures are reported and the batch moves on. A summary
is written to <out>/batch_<file stem>.json.
"""

import dataclasses
import json
import os
import time

from .config import EnergizerParams
from .hybrid import AIRejected


def load_tracks(path):
    import yaml
    with open(path) as fh:
        rows = yaml.safe_load(fh)
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a YAML list of tracks")
    fields = {f.name for f in dataclasses.fields(EnergizerParams)}
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        name = str(r.pop("name", f"track{i + 1}"))
        bad = set(r) - fields
        if bad:
            raise SystemExit(f"{path}: track {name}: unknown field(s) {sorted(bad)}")
        r.setdefault("music_source", "elevenlabs")
        out.append((name, EnergizerParams(**r)))
    return out


def run_batch(path, out_root, max_retries=2, dry_run=False, log=print):
    from .render import render, render_reference
    tracks = load_tracks(path)
    summary = []
    for name, params in tracks:
        log(f"\n=== {name}: {params.music_style}, {params.key_root} {params.mode}, {params.plateau_bpm:g} BPM, "
            f"variant {params.music_variant}, pulse {params.pulse_style}, pad {params.pad_timbre}, seed {params.seed} ===")
        if dry_run:
            render_reference(params, out_root, dry_run=True, log=log)
            continue
        row = dict(name=name, params=dataclasses.asdict(params), attempts=[])
        p = params
        for attempt in range(max_retries + 1):
            t = time.time()
            try:
                out, man = render(p, out_root, log=log)
                row.update(status="ok", bed_id=man["bed_id"], seed=p.seed, out=out, tags=man["qa_tags"],
                           seconds=round(time.time() - t))
                row["attempts"].append(dict(seed=p.seed, result="ok"))
                break
            except AIRejected as e:
                log(f"  rejected (seed {p.seed}): {e}")
                row["attempts"].append(dict(seed=p.seed, result=str(e)))
                p = dataclasses.replace(p, seed=p.seed + 1)
            except (SystemExit, Exception) as e:              # report, keep going with the next track
                log(f"  FAILED: {type(e).__name__}: {e}")
                row["attempts"].append(dict(seed=p.seed, result=f"{type(e).__name__}: {e}"))
                row["status"] = "failed"
                break
        else:
            row["status"] = "rejected"
        summary.append(row)
    if dry_run:
        return 0
    os.makedirs(out_root, exist_ok=True)
    sp = os.path.join(out_root, f"batch_{os.path.splitext(os.path.basename(path))[0]}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log("\n" + "\n".join(f"  {r['name']:<10s} {r['status']:<9s} {r.get('bed_id', '-'):<13s} seed {r.get('seed', '-')}"
                          f"  tags {r.get('tags', {}).get('lever_compatible', '-')}  -> {r.get('out', '')}"
                          for r in summary) + f"\nsummary -> {sp}")
    return 0 if all(r["status"] == "ok" for r in summary) else 2

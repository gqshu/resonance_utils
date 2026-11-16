"""Deploy helpers shared by every soundscape type: hashed file entries, a guarded --clean,
merging a partial deploy into the previous manifest, and the deploy/index.json of packs."""

import datetime
import hashlib
import json
import os
import shutil

INDEX_SCHEMA_VERSION = 1


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(src, dst_dir, out_root, name=None):
    """Copy src into dst_dir and describe it relative to out_root: {path, bytes, sha256}."""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name or os.path.basename(src))
    shutil.copy2(src, dst)
    return dict(path=os.path.relpath(dst, out_root).replace(os.sep, "/"),
                bytes=os.path.getsize(dst), sha256=sha256(dst))


def prepare_out(out, clean, manifest_name="manifest.json"):
    """Create out; with clean, empty it first, but only if it already holds our manifest."""
    man_path = os.path.join(out, manifest_name)
    if clean and os.path.isdir(out):
        if os.listdir(out) and not os.path.exists(man_path):
            raise SystemExit(f"--clean refused: {out}/ has files but no {manifest_name}, "
                             "so it isn't an ss deploy directory")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    return man_path


def merge_previous(man_path, items, key, still_present, clean):
    """Items from the previous manifest at man_path[key] that this deploy didn't replace and whose
    files still exist (still_present(item) -> bool). Returns (merged sorted by id, kept ids)."""
    if clean or not os.path.exists(man_path):
        return sorted(items, key=lambda s: s["id"]), []
    with open(man_path) as fh:
        old = {s["id"]: s for s in json.load(fh).get(key, [])}
    new_ids = {s["id"] for s in items}
    kept = [s for sid, s in old.items() if sid not in new_ids and still_present(s)]
    return sorted(items + kept, key=lambda s: s["id"]), [s["id"] for s in kept]


def tree_bytes(root):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(root) for f in fs)


def write_index(deploy_root, generator):
    """deploy/index.json: every pack (type) that has a manifest under deploy_root."""
    packs = []
    for t in sorted(os.listdir(deploy_root)) if os.path.isdir(deploy_root) else []:
        mp = os.path.join(deploy_root, t, "manifest.json")
        if not os.path.isfile(mp):
            continue
        with open(mp) as fh:
            m = json.load(fh)
        items = m.get("scenarios", m.get("beds", []))
        packs.append(dict(type=t, manifest=f"{t}/manifest.json", schema_version=m.get("schema_version"),
                          generated_at=m.get("generated_at"), count=len(items)))
    idx = dict(schema_version=INDEX_SCHEMA_VERSION, generated_at=now_utc(), generator=generator, packs=packs)
    with open(os.path.join(deploy_root, "index.json"), "w") as fh:
        json.dump(idx, fh, indent=2)
    return idx

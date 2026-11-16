#!/usr/bin/env python3
"""ss: one command for every soundscape type.

    ss list                              the soundscape types
    ss list inducer                      what a type can generate (scenarios, presets)
    ss gen inducer rain_soft --tone      build into build/inducer/<id>/   (ss gen <type> -h for options)
    ss gen energizer ...                 build into build/energizer/<id>/
    ss deploy inducer --stems            ready builds -> deploy/inducer/manifest.json
    ss deploy all                        every type; also writes deploy/index.json
    ss migrate [--dry-run]               move a pre-0.3 build/ and deploy/ into the per-type layout

ss_gen and ss_deploy remain as aliases for `ss gen inducer` and `ss deploy inducer`.
"""

import argparse
import json
import os
import shutil
import sys

from .core import types


def _type_or_exit(name):
    try:
        return types.get(name)
    except ImportError as e:
        sys.exit(f"soundscape type {name!r} needs extra dependencies ({e}); "
                 f"install them with: pip install -e '.[{name}]'")


def cmd_list(argv):
    if argv:
        _type_or_exit(argv[0]).list()
        return 0
    for n in types.names():
        try:
            print(f"{n:11s} {types.get(n).summary}")
        except ImportError as e:
            print(f"{n:11s} (unavailable: {e})")
    return 0


def _typed(cmd, argv, add, run):
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: ss {cmd} <type> [options]   types: {', '.join(types.names())}")
        return 0 if argv else 2
    t = _type_or_exit(argv[0])
    ap = argparse.ArgumentParser(prog=f"ss {cmd} {t.name}",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    getattr(t, add)(ap)
    return getattr(t, run)(ap.parse_args(argv[1:]), ap)


def cmd_deploy(argv):
    if argv and argv[0] == "all":
        rc = 0
        for n in types.names():
            try:
                t = types.get(n)
            except ImportError as e:
                print(f"--- {n}: skipped ({e})")
                continue
            print(f"--- {n}")
            ap = argparse.ArgumentParser(prog=f"ss deploy {n}")
            t.add_deploy_args(ap)
            a, _ = ap.parse_known_args(argv[1:])
            r = t.deploy(a, ap)
            rc = rc or (r if r not in (None, 1) else 0)      # 1 = nothing to deploy, not an error
        return rc
    return _typed("deploy", argv, "add_deploy_args", "deploy")


# --- migrate: pre-0.3 layout (build/<scenario>/, deploy/manifest.json) -> per-type folders ----
def _is_inducer_build(d):
    return os.path.isfile(os.path.join(d, "report.json")) or os.path.isdir(os.path.join(d, "raw"))


def cmd_migrate(argv):
    ap = argparse.ArgumentParser(prog="ss migrate", description=(
        "Move inducer builds from build/<scenario>/ to build/inducer/<scenario>/ (raw takes "
        "included, so nothing is re-bought), and a schema-1 deploy/manifest.json with its "
        "scenario folders to deploy/inducer/. Folders that don't look like builds are left alone."))
    ap.add_argument("--build", default="build")
    ap.add_argument("--deploy", default="deploy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    moves, left = [], []
    tnames = set(types.names())
    if os.path.isdir(a.build):
        for d in sorted(os.listdir(a.build)):
            src = os.path.join(a.build, d)
            if not os.path.isdir(src) or d in tnames:
                continue
            (moves if _is_inducer_build(src) else left).append((src, os.path.join(a.build, "inducer", d)))
    man = os.path.join(a.deploy, "manifest.json")
    if os.path.isfile(man):
        with open(man) as fh:
            ids = [s["id"] for s in json.load(fh).get("scenarios", [])]
        moves.append((man, os.path.join(a.deploy, "inducer", "manifest.json")))
        for i in ids:
            if os.path.isdir(os.path.join(a.deploy, i)):
                moves.append((os.path.join(a.deploy, i), os.path.join(a.deploy, "inducer", i)))
    if not moves:
        print("nothing to migrate")
        return 0
    clash = [dst for _, dst in moves if os.path.exists(dst)]
    if clash:
        sys.exit(f"refusing to migrate: already exists: {', '.join(clash)}")
    for src, dst in moves:
        print(f"  {'would move' if a.dry_run else 'move'} {src} -> {dst}")
        if not a.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    for src, _ in left:
        print(f"  left alone (not a build): {src}")
    if not a.dry_run and os.path.isdir(os.path.join(a.deploy, "inducer")):
        from . import __version__
        from .core.deploy import write_index
        write_index(a.deploy, f"resonance_utils {__version__}")
    return 0


COMMANDS = {
    "list": cmd_list,
    "gen": lambda argv: _typed("gen", argv, "add_gen_args", "gen"),
    "deploy": cmd_deploy,
    "migrate": cmd_migrate,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    return COMMANDS[argv[0]](argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())

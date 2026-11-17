import dataclasses
import json
import os

from ..core.layout import type_dir
from ..core.types import SoundscapeType
from .config import EnergizerParams

TYPE = "energizer"


class EnergizerType(SoundscapeType):
    name = TYPE
    summary = "wake-to-work energizer: tempo-ramped hybrid beds with lever-ready stems (in development)"

    def add_gen_args(self, ap):
        ap.add_argument("--out", default="build", help="build root; beds go to <out>/energizer/<bed_id>/")
        ap.add_argument("--dry-run", action="store_true",
                        help="print the params, bed_id and any API calls with their cost; render nothing")
        ap.add_argument("--batch", metavar="FILE",
                        help="YAML list of tracks (each a dict of params, plus an optional name); renders "
                             "each end to end, retrying a key/tempo miss with the next seed")
        ap.add_argument("--max-retries", type=int, default=2,
                        help="with --batch: new seeds to try per track after a key/tempo miss (each costs a "
                             "new composition, ~3,100 credits)")
        ap.add_argument("--reference-only", action="store_true",
                        help="with --music-source elevenlabs: compose the AI tracks and write the "
                             "un-separated reference mix (M2 c), not the hybrid")
        for f in dataclasses.fields(EnergizerParams):
            flag = "--" + f.name.replace("_", "-")
            if f.type in (bool, "bool"):
                ap.add_argument(flag, type=lambda v: v.lower() in ("1", "true", "yes"), default=f.default)
            else:
                tname = f.type if isinstance(f.type, str) else f.type.__name__
                kind = {"int": int, "float": float}.get(tname, str)
                ap.add_argument(flag, type=kind, default=f.default)

    def gen(self, args, ap):
        if args.batch:
            from .batch import run_batch
            return run_batch(args.batch, type_dir(args.out, TYPE), args.max_retries, dry_run=args.dry_run)
        from .config import bed_id, lever_slots
        fields = dataclasses.fields(EnergizerParams)
        params = EnergizerParams(**{f.name: getattr(args, f.name) for f in fields})
        slots = lever_slots()
        bid = bed_id(params, slots)
        out = os.path.join(type_dir(args.out, TYPE), bid)
        if params.music_source == "elevenlabs":
            from .render import render_reference
            if args.reference_only or args.dry_run:
                render_reference(params, type_dir(args.out, TYPE), dry_run=args.dry_run)
                return 0
            from .render import render
            render(params, type_dir(args.out, TYPE))
            return 0
        print(json.dumps(dict(bed_id=bid, out=out, params=dataclasses.asdict(params), lever_slots=slots),
                         indent=2))
        if args.dry_run:
            return 0
        from .render import render
        render(params, type_dir(args.out, TYPE))
        return 0

    def add_deploy_args(self, ap):
        ap.add_argument("beds", nargs="*")
        ap.add_argument("--build", default="build")
        ap.add_argument("--out", default="deploy")
        ap.add_argument("--clean", action="store_true")

    def deploy(self, args, ap):
        build = type_dir(args.build, TYPE)
        if not os.path.isdir(build) or not os.listdir(build):
            print(f"  no energizer beds in {build}/")
            return 1
        print("energizer deploy is not implemented yet (needs the QA gate, plan §9 / M3)")
        return 3

    def list(self):
        from .config import load_yaml
        print("arc profiles: " + ", ".join(load_yaml("arc_profiles.yaml")))
        print("music styles: " + ", ".join(load_yaml("music_styles.yaml")))
        print("params (ss gen energizer --<param> value):")
        for f in dataclasses.fields(EnergizerParams):
            print(f"  --{f.name.replace('_', '-'):18s} {f.default!r}")

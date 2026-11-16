"""Golden regression test for the inducer pipeline (ss_gen --from-dir -> ss_deploy).

Builds a short soundscape from synthetic takes (no API) and compares the SHA-256 of every
output against tests/golden/inducer.json, so refactors must keep outputs bit-identical.

Re-record after an intentional output change:
    RECORD_GOLDEN=1 pytest tests/test_inducer_golden.py
"""

import glob
import hashlib
import json
import os

import numpy as np
import pytest
from scipy.signal import lfilter

from resonance_utils.core.audio_io import FS, normalise, write_wav24
from resonance_utils.cli import main as ss_main

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "inducer.json")


def _takes(d, n=4, seconds=30.0):
    """Smooth, stationary brown-ish noise: passes screening, varies per take."""
    os.makedirs(d)
    for i in range(n):
        rng = np.random.default_rng(100 + i)
        w = rng.standard_normal((int(seconds * FS), 2))
        x = lfilter([1.0], [1.0, -0.98], w, axis=0)
        write_wav24(os.path.join(d, f"take_{i:02d}.wav"), normalise(x))


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _digest(root, skip_dirs=("raw",)):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isdir(p) or any(f"{os.sep}{s}{os.sep}" in p for s in skip_dirs):
            continue
        name = os.path.basename(p)
        if name == "index.json":                        # new in 0.3, timestamped; checked below
            continue
        if name == "manifest.json":
            with open(p) as fh:
                m = json.load(fh)
            m.pop("generated_at", None)
            m.pop("generator", None)
            out[name] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
        elif name.endswith((".json",)):
            with open(p) as fh:
                out[name] = hashlib.sha256(json.dumps(json.load(fh), sort_keys=True).encode()).hexdigest()
        else:
            out[name] = _sha(p)
    return out


def test_inducer_golden(tmp_path):
    takes = str(tmp_path / "takes" / "cabin")
    _takes(takes)
    build, deploy = str(tmp_path / "build"), str(tmp_path / "deploy")
    assert ss_main(["gen", "inducer", "--from-dir", takes, "--out", build, "--loop-seconds", "20",
                    "--tone", "--stems", "--seed", "1", "--min-pass", "2"]) == 0
    assert ss_main(["deploy", "inducer", "--build", build, "--out", deploy, "--stems"]) == 0
    got = dict(build=_digest(build), deploy=_digest(deploy))
    if os.environ.get("RECORD_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=2, sort_keys=True)
        pytest.skip("golden recorded")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
    with open(os.path.join(deploy, "index.json")) as fh:
        idx = json.load(fh)
    packs = [(p["type"], p["manifest"], p["count"]) for p in idx["packs"]]
    assert packs == [("inducer", "inducer/manifest.json", 1)]

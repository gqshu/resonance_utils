"""Energizer M1 acceptance (plan §12): sidecars, slot and pairwise-beat rules, grid, determinism.
Short beds (the arc scales proportionally) keep the suite fast."""

import glob
import hashlib
import json
import os

import pytest

from resonance_utils.energizer.config import EnergizerParams, lever_slots, partial_clash
from resonance_utils.energizer.render import render
from resonance_utils.energizer.tempo import TempoMap

SIDE = ["manifest.json", "tempo_map.json", "arc.json", "chords.json", "pad_partials.json",
        "qa_report.json", "preview_mix.wav", "stems/pad.wav", "stems/bass.wav", "stems/pulse.wav",
        "stems/texture.wav"]


def _hashes(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)):
        with open(p, "rb") as fh:
            out[os.path.relpath(p, d)] = hashlib.sha256(fh.read()).hexdigest()
    return out


@pytest.fixture(scope="module")
def beds(tmp_path_factory):
    p = EnergizerParams(seed=3, length_s=24, mode="mixolydian", key_root="E")
    a, _ = render(p, str(tmp_path_factory.mktemp("a")), log=lambda *x: None)
    b, _ = render(p, str(tmp_path_factory.mktemp("b")), log=lambda *x: None)
    return a, b


def test_sidecars(beds):
    for f in SIDE:
        assert os.path.isfile(os.path.join(beds[0], f)), f


def test_bit_identical(beds):
    assert _hashes(beds[0]) == _hashes(beds[1])


def test_pad_partials_obey_slots(beds):
    slots = lever_slots()
    rows = json.load(open(os.path.join(beds[0], "pad_partials.json")))["chords"]
    for r in rows:
        fs = [p["f"] for p in r["partials"]]
        assert all(partial_clash(f, slots) is None for f in fs)
        for i, f in enumerate(fs):
            for g in fs[i + 1:]:
                assert not any(lo <= abs(f - g) <= hi for lo, hi in slots["native_mod_bands_hz"])


def test_qa(beds):
    q = json.load(open(os.path.join(beds[0], "qa_report.json")))
    assert max(q["grid_error_ms"].values()) <= 1.0
    assert q["mb_slot_residual_db"]["mix"] <= -18
    assert q["true_peak_dbfs"] <= -1.0


def test_tempo_map_exact():
    tm = TempoMap(240, [(0, 100), (30, 100), (150, 118), (240, 118)])
    # entry: 50 beats; ramp: mean bpm 109 over 120 s = 218 beats; plateau 90 s at 118 = 177 beats
    assert abs(tm.total_beats - (50 + 218 + 177)) < 1e-6
    assert abs(tm.time_of_beat(50) - 30) < 1e-6 and abs(tm.time_of_beat(268) - 150) < 1e-6


def test_deploy_gate_and_manifest(beds, tmp_path):
    import shutil
    from resonance_utils.cli import main as ss
    build = tmp_path / "build" / "energizer"
    shutil.copytree(beds[0], build / os.path.basename(beds[0]))
    bid = os.path.basename(beds[0])
    q = json.load(open(build / bid / "qa_report.json"))
    passing = q["tags"]["lever_compatible"]
    out = tmp_path / "deploy"
    rc = ss(["deploy", "energizer", "--build", str(tmp_path / "build"), "--out", str(out)])
    assert rc == (0 if passing else 1)
    assert ss(["deploy", "energizer", "--build", str(tmp_path / "build"), "--out", str(out), "--allow-qa-fail"]) == 0
    m = json.load(open(out / "energizer" / "manifest.json"))
    b = m["beds"][0]
    assert b["bed_id"] == bid and b["stems"] is None
    assert b["audio"]["path"] == f"{b['id']}/{b['id']}.wav" and os.path.isfile(out / "energizer" / b["audio"]["path"])
    assert ss(["deploy", "energizer", "--build", str(tmp_path / "build"), "--out", str(tmp_path / "ds"),
               "--allow-qa-fail", "--stems"]) == 0
    bs = json.load(open(tmp_path / "ds" / "energizer" / "manifest.json"))["beds"][0]
    assert set(bs["stems"]) == {"pad", "bass", "pulse", "texture"}
    assert json.load(open(out / "index.json"))["packs"][0]["type"] == "energizer"
    # a stem changed after the build is refused
    with open(build / bid / "stems" / "pad.wav", "ab") as fh:
        fh.write(b"x")
    assert ss(["deploy", "energizer", "--build", str(tmp_path / "build"), "--out", str(tmp_path / "d2"),
               "--allow-qa-fail"]) == 1

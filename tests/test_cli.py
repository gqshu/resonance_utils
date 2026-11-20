import json
import os

from resonance_utils.cli import main as ss


def test_list_types(capsys):
    assert ss(["list"]) == 0
    out = capsys.readouterr().out
    assert "inducer" in out and "energizer" in out


def test_list_inducer(capsys):
    assert ss(["list", "inducer"]) == 0
    assert "rain_soft" in capsys.readouterr().out


def test_energizer_dry_run(tmp_path, capsys):
    assert ss(["gen", "energizer", "--dry-run", "--out", str(tmp_path), "--plateau-bpm", "114"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["params"]["plateau_bpm"] == 114.0
    assert d["lever_slots"]["mb_tones_hz"] == [220.0, 240.0, 260.0]
    assert d["lever_slots"]["mb_keep_clear_hz"] == [[200.0, 280.0]]
    assert len(d["bed_id"]) == 12


def test_inducer_dry_run_alias(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    from resonance_utils.inducer.gen import main as ss_gen
    assert ss_gen(["rain_soft", "--dry-run", "--takes", "1", "--out", str(tmp_path)]) == 0
    assert "POST" in capsys.readouterr().out
    assert os.path.isdir(tmp_path / "inducer" / "rain_soft" / "raw")


def test_migrate(tmp_path):
    b, d = tmp_path / "build", tmp_path / "deploy"
    (b / "rain_soft" / "raw").mkdir(parents=True)
    (b / "rain_soft" / "report.json").write_text("{}")
    (b / "scratch").mkdir()                           # not a build: left alone
    (d / "rain_soft").mkdir(parents=True)
    (d / "rain_soft" / "x.wav").write_bytes(b"x")
    (d / "manifest.json").write_text(json.dumps(dict(schema_version=1, scenarios=[dict(id="rain_soft")])))
    assert ss(["migrate", "--build", str(b), "--deploy", str(d)]) == 0
    assert (b / "inducer" / "rain_soft" / "raw").is_dir() and (b / "scratch").is_dir()
    assert (d / "inducer" / "manifest.json").is_file() and (d / "inducer" / "rain_soft" / "x.wav").is_file()
    idx = json.loads((d / "index.json").read_text())
    assert idx["packs"][0]["type"] == "inducer"
    assert ss(["migrate", "--build", str(b), "--deploy", str(d)]) == 0   # idempotent: nothing left


def test_energizer_partial_clash():
    from resonance_utils.energizer.config import lever_slots, partial_clash
    s = lever_slots()
    assert partial_clash(250, s).startswith("in the monaural slot")
    assert "40 Hz with the 220" in partial_clash(180, s)          # bass note beating at 40 Hz
    assert "40 Hz with the 260" in partial_clash(300, s)
    assert partial_clash(146.8, s) is None and partial_clash(330, s) is None


def test_inducer_sa3_dry_run(tmp_path, capsys):
    """--engine sa3 defaults to the reference variant and writes to <scenario>_sa3; a dry run
    never starts the model, so it needs neither the stable-audio-3 checkout nor refs/."""
    from resonance_utils.inducer.gen import main as ss_gen
    assert ss_gen(["rain_soft", "--engine", "sa3", "--dry-run", "--takes", "1", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "./sa3 --dit medium --seconds 66" in out
    assert os.path.isdir(tmp_path / "inducer" / "rain_soft_sa3" / "raw")


def test_identity_separates_spectra():
    import numpy as np
    from resonance_utils.core import identity
    rng = np.random.default_rng(0)
    white = rng.standard_normal((48000 * 10, 2))
    dark = np.cumsum(white, 0)
    dark -= dark.mean(0)
    a, b = identity.fingerprint(white), identity.fingerprint(dark / np.abs(dark).max())
    assert identity.distance(a, a)[0] == 0
    assert identity.distance(a, b)[0] > 10


def test_build_split_makes_disjoint_loops(tmp_path):
    """--split: four 60 s takes -> two 100 s loops, no take shared, every loop screened clean."""
    import numpy as np
    from scipy.signal import lfilter

    from resonance_utils.core.audio_io import FS, normalise, write_wav24
    from resonance_utils.inducer import bed
    paths = []
    for i in range(4):
        w = np.random.default_rng(200 + i).standard_normal((60 * FS, 2))
        p = str(tmp_path / f"take_{i:02d}.wav")
        write_wav24(p, normalise(lfilter([1.0], [1.0, -0.98], w, axis=0)))
        paths.append(p)
    beds, report = bed.build_split(paths, [110.0], 20.0, 100.0)
    assert [len(b) for b, _ in beds] == [100 * FS, 100 * FS]
    used = [t for _, o in beds for t in o]
    assert len(used) == len(set(used)) == 4
    assert not report.get("dropped_loops")

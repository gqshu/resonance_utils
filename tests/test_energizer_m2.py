"""Energizer M2 pieces on synthetic audio, no API: separation parsing (mocked HTTP), tempo grid,
downbeat and chords, the calibrated time-map stretch, and the bar map / splice."""

import io
import zipfile

import numpy as np
import pytest

from resonance_utils.core.audio_io import FS
from resonance_utils.energizer import analyze, conform, el_music
from resonance_utils.energizer.arc import load as load_arc
from resonance_utils.energizer.tempo import from_arc

BPM, T0, DOWNBEAT = 118.0, 0.3, 1


def synth_track(seconds=40.0):
    """Drums on every beat, bass alternating D/G every 2 bars from beat DOWNBEAT, keys on the triad."""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    per = 60 / BPM
    drums, bass, keys = np.zeros(n), np.zeros(n), np.zeros(n)
    beats = T0 + np.arange(int((seconds - T0) / per)) * per
    click = np.random.default_rng(0).standard_normal(2400) * np.exp(-np.arange(2400) / 300)
    for b in beats:
        s = int(b * FS)
        m = min(2400, n - s)
        drums[s:s + m] += click[:m]
    for j in range(0, len(beats) // 4):
        s0 = beats[DOWNBEAT] + 4 * j * per
        seg = (t >= s0) & (t < s0 + 4 * per)
        root, tri = (73.42, (293.66, 369.99, 440.0)) if (j // 2) % 2 == 0 else (98.0, (392.0, 493.88, 587.33))
        bass[seg] += 0.5 * np.sin(2 * np.pi * root * t[seg])
        for f in tri:
            keys[seg] += 0.2 * np.sin(2 * np.pi * f * t[seg])
    st = lambda x: np.stack([x, x], 1)
    return dict(drums=st(drums), bass=st(bass), other=st(keys), vocals=st(np.zeros(n)))


def test_separation_zip_parsing(monkeypatch, tmp_path):
    stems = synth_track(4.0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, x in stems.items():
            z.writestr(f"{k}.pcm", (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    calls = []
    monkeypatch.setattr(el_music.el, "post", lambda *a, **k: calls.append(k) or buf.getvalue())
    monkeypatch.setenv("ELEVENLABS_CACHE_DIR", str(tmp_path / "cache"))
    wav = tmp_path / "music.wav"
    from resonance_utils.core.audio_io import write_wav24
    write_wav24(str(wav), stems["other"])
    out, meta = el_music.separate(str(wav), str(tmp_path / "stems"), log=lambda *a: None)
    assert set(out) == set(stems) and "multipart/form-data" in calls[0]["content_type"]
    assert el_music.music_stem_names(out) == ["other"]


@pytest.fixture(scope="module")
def analysed():
    st = synth_track()
    return st, analyze.analyse_track(st, BPM, "D", "major")


def test_grid_and_downbeat(analysed):
    _, (music, grid, key, bars) = analysed
    assert abs(grid["bpm"] - BPM) < 0.05 and grid["tracked_on"] == "drums"
    assert abs(grid["t0"] - T0) < 0.01
    assert grid["downbeat"] == DOWNBEAT
    assert key["pitch_set_ok"]


def test_bar_chords(analysed):
    _, (_, _, _, bars) = analysed
    want = ["D", "D", "G", "G"] * 3
    assert [b["label"] for b in bars[:12]] == want


def test_stretch_lands_on_map():
    from resonance_utils.energizer.stretch import stretch_to_map
    n = 20 * FS
    x = np.zeros((n, 2))
    for k in range(0, n, FS // 2):
        x[k:k + 200] = (np.hanning(400)[200:] * 0.8)[:, None]
    ai = np.arange(41) * FS // 2
    ao = np.concatenate([[0], np.cumsum(np.linspace(0.59, 0.5, 40))]) * FS
    y, info = stretch_to_map(x, ai, ao, max_stretch=1.18)
    assert info["residual_ms"]["median"] < 2.0
    with pytest.raises(ValueError):
        stretch_to_map(x, ai, ai * 1.3, max_stretch=1.18)


def test_bar_map_and_splice():
    arc = load_arc("wake_to_work_4m")
    tm = from_arc(arc, BPM, 0.85)
    per = 60 / BPM
    grid = dict(period_s=per, t0=0.5, downbeat=0, n_beats=int(218 / per))
    plan = el_music.main_plan(type("P", (), dict(music_style="ambient_electronic_keys", plateau_bpm=BPM,
                                                 key_root="D", mode="major"))(), arc)
    pairs, ai_bar_t = conform.bar_map(grid, conform.ai_sections(plan, el_music.PREROLL_S), tm, arc,
                                      el_music.PREROLL_S)
    bed = [b for _, b in pairs]
    ai = [a for a, _ in pairs]
    assert bed == list(range(bed[0], bed[0] + len(bed)))           # every bed bar once, in order
    assert all(b >= a for a, b in zip(ai, ai[1:]))                 # AI bars never go backwards
    assert len(set(ai)) < max(ai) + 1                              # the ramp needed fewer bars: some dropped
    x = np.random.default_rng(1).standard_normal((int(220 * FS), 2)) * 0.1
    y, starts = conform.splice(x, pairs, ai_bar_t, per)
    assert len(starts) == len(pairs) and abs(len(y) - len(pairs) * 4 * per * FS) < len(pairs) + 2


def test_variant_plans_fill_the_ramp_exactly():
    from resonance_utils.energizer.config import EnergizerParams
    arc = load_arc("wake_to_work_4m")
    tm = from_arc(arc, BPM, 0.85)
    a, _ = el_music.build_main(EnergizerParams(music_variant="a"), arc, tm)
    assert [c["text"] for c in a["chunks"]] == ["[Build Sparse]", "[Build Fuller]", "[Plateau]", "[Handoff]"]
    for v in "bcd":
        plan, phases = el_music.build_main(EnergizerParams(music_variant=v), arc, tm)
        assert all(3000 <= c["duration_ms"] <= 120000 for c in plan["chunks"])
        bar_ms = 4 * 60 / BPM * 1000
        assert all(abs(c["duration_ms"] / bar_ms - round(c["duration_ms"] / bar_ms)) < 1e-3 for c in plan["chunks"])
        grid = dict(period_s=60 / BPM, t0=0.0, downbeat=0, n_beats=int(el_music.plan_seconds(plan) * BPM / 60))
        pairs, _ = conform.bar_map(grid, conform.ai_sections(plan, el_music.PREROLL_S, phases), tm, arc,
                                   el_music.PREROLL_S)
        ai = [p[0] for p in pairs]
        assert len(set(ai)) >= len(ai) - 2, v                       # (almost) no AI bars dropped or repeated
        assert "risers" not in plan["chunks"][0]["negative_styles"]


def test_pulse_and_bass_follow_sections():
    from resonance_utils.energizer.synths import bass, pulse
    arc = load_arc("wake_to_work_4m")
    tm = from_arc(arc, BPM, 0.85)
    b_pl = round(float(tm.beat_at(arc.phases["plateau"][0])) / 4) * 4
    secs = [dict(name="[Full]", role="main", index=3, start_beat=b_pl, end_beat=b_pl + 32),
            dict(name="[Breakdown]", role="breakdown", index=4, start_beat=b_pl + 32, end_beat=b_pl + 48),
            dict(name="[Return]", role="main", index=5, start_beat=b_pl + 48, end_beat=b_pl + 96)]
    ev = pulse.events("soft_four", tm, arc.entries["pulse"], arc.phases["ramp"], arc.phases["plateau"],
                      arc.phases["handoff"], 1, secs)
    kicks = {round(b, 3) for b, k, _ in ev if k == "kick"}
    assert not any(b_pl + 32 <= b < b_pl + 48 for b in kicks)          # breakdown: no kick
    assert b_pl + 31 not in kicks and b_pl + 28 in kicks               # fill bar: beat 4 left out
    assert all(abs(b * 4 - round(b * 4)) < 1e-9 for b, _, _ in ev)     # nothing finer than 16ths
    assert any(b_pl + 50 == b for b in kicks) and b_pl + 49 not in kicks   # 2nd main section: half-time kick
    chords = [dict(start_beat=float(b), end_beat=float(b + 8), start_s=float(tm.time_of_beat(b)),
                   end_s=float(tm.time_of_beat(b + 8)), root_pc=2) for b in range(int(b_pl), int(b_pl) + 96, 8)]
    nts = bass.notes(chords, tm, 60, arc.phases["plateau"], arc.phases["handoff"], secs)
    in_bd = [n for n in nts if b_pl + 32 <= n["beat"] < b_pl + 48]
    assert len(in_bd) == 2 and all(n["vel"] == 0.7 for n in in_bd)     # breakdown: held soft notes


def test_batch_file_validation(tmp_path):
    from resonance_utils.energizer.batch import load_tracks
    p = tmp_path / "t.yaml"
    p.write_text("- name: x\n  key_root: F\n  plateau_bpm: 114\n- seed: 3\n")
    rows = load_tracks(str(p))
    assert [n for n, _ in rows] == ["x", "track2"] and rows[0][1].music_source == "elevenlabs"
    p.write_text("- name: x\n  tempo: 114\n")
    with pytest.raises(SystemExit):
        load_tracks(str(p))

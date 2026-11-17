"""Full pipeline -> one bed folder (plan §8). M1: all-procedural (music_source = none).

    plan -> tempo map -> chords -> pad / bass / pulse / texture on the map -> arc -> slot notch
    -> per-stem levels -> preview mix -> sidecars -> QA

Output, in build/energizer/<bed_id>/:
    manifest.json   params, seeds, generator version, lever slots, stem files + gains, QA tags
    stems/<stem>.wav  48 kHz / 24-bit, equal length, sample-aligned
    tempo_map.json  arc.json  chords.json  pad_partials.json  preview_mix.wav  qa_report.json
"""

import dataclasses
import json
import os
import time

import numpy as np

from .. import __version__
from ..core.audio_io import FS, write_wav24
from ..core.deploy import now_utc, sha256
from ..core.slots import band_stop
from . import arc as arcmod
from . import fx, qa, tempo as tempomod, theory
from .config import GENERATOR_VERSION, bed_id, lever_slots, load_yaml
from .synths import air, bass, pad, pulse

# RMS (dBFS) of each stem where its arc gain is at its top (the plateau for most; the entry for
# the motif); the arc is relative to these. The texture sits well back ("noisy" in M1 listening).
STEM_TARGET_DBFS = dict(pad=-21.0, bass=-24.0, pulse=-25.0, texture=-38.0)
HYBRID_TARGET_DBFS = dict(music=-20.0, motif=-21.0, pad=-27.0, bass=-25.0, pulse=-26.0, texture=-40.0)
TRUE_PEAK_MAX_DBFS = -1.0


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def render(params, out_root, log=print):
    t_start = time.time()
    hybrid = params.music_source == "elevenlabs"
    if params.music_source not in ("none", "elevenlabs"):
        raise SystemExit(f"unknown music_source {params.music_source}")
    if params.texture_source not in ("procedural_air", "none"):
        raise SystemExit(f"texture_source={params.texture_source} is milestone M2; use procedural_air or none")
    th = load_yaml("thresholds.yaml")
    slots = lever_slots()
    if hybrid:
        from . import el_music
        bid = bed_id(params, slots, dict(music=el_music.MODEL_ID))
    else:
        bid = bed_id(params, slots)
    out = os.path.join(out_root, bid)
    os.makedirs(os.path.join(out, "stems"), exist_ok=True)

    arc = arcmod.load(params.arc_profile, params.length_s)
    tm = tempomod.from_arc(arc, params.plateau_bpm, params.entry_tempo_ratio)
    n = int(round(arc.length_s * FS))
    n_cr = int(arc.length_s * arcmod.CR) + 2
    bright = arc.brightness_curve(n_cr)
    ai = None
    if hybrid:
        from .hybrid import ai_layer
        ai = ai_layer(params, arc, tm, out, slots, th, n, log=log)
        chords = ai["chords"]
    else:
        chords = theory.chord_timeline(tm, params.key_root, params.mode, params.seed)
    targets = HYBRID_TARGET_DBFS if hybrid else STEM_TARGET_DBFS
    log(f"  bed {bid}: {arc.length_s:g} s, {tm.points[0][1]:.1f} -> {params.plateau_bpm:g} BPM, "
        f"{params.key_root} {params.mode}, {len(chords)} chords ({' '.join(c['label'] for c in chords[:4])} ...)")

    # bass first: its partials are audited against the pad's
    sections = ai["sections"] if ai else None
    bnotes = bass.notes(chords, tm, arc.entries["bass"] * 1.0, arc.phases["plateau"], arc.phases["handoff"],
                        sections)
    bpart = {i: bass.partials(bass.root_hz(c["root_pc"]), slots) for i, c in enumerate(chords)}
    rows = pad.plan(chords, slots, params.pad_timbre, params.pad_tuning, arc.phases["plateau"], bpart, sections)

    stems = {}
    t0 = time.time()
    p_main, p_hand = pad.render(chords, rows, n, n_cr, params.pad_timbre, bright, params.seed, split=True)
    stems["pad"] = fx.reverb(p_main, params.seed) + fx.REVERB_DRY * p_hand
    log(f"  pad: {sum(len(r['partials']) for r in rows)} partials over {len(rows)} chords, "
        f"{sum(len(r['dropped']) for r in rows)} dropped ({time.time() - t0:.1f} s)")
    stems["bass"] = bass.render(bnotes, n, slots)
    onsets = {"bass": [dict(beat=nt["beat"], sample=int(round(nt["t0"] * FS))) for nt in bnotes]}
    if params.pulse_style != "none":
        ev = pulse.events(params.pulse_style, tm, arc.entries["pulse"], arc.phases["ramp"],
                          arc.phases["plateau"], arc.phases["handoff"], params.seed, sections)
        stems["pulse"], onsets["pulse"] = pulse.render(ev, tm, n, params.seed,
                                                      lambda t: float(np.interp(t * arcmod.CR, np.arange(n_cr), bright)))
    if params.texture_source == "procedural_air":
        stems["texture"] = fx.brightness_blend(air.render(n, n_cr, params.seed), bright)
    if ai:
        stems.update(ai["stems"])

    # arc gains, slot notch, levels (each measured where its arc gain is at the top)
    gains_db = {}
    for k in list(stems):
        gc = arc.gain_curve(k, n_cr)
        x = fx.apply_gain(stems[k], gc)
        x = band_stop(x, slots["mb_keep_clear_hz"], depth_db=th["MB_NOTCH_MIN_DB"] + 12)
        top = np.flatnonzero(gc >= 0.9 * gc.max())
        win = slice(int(top[0] / arcmod.CR * FS), int(top[-1] / arcmod.CR * FS))
        g = 10 ** (targets[k] / 20) / _rms(x[win])
        stems[k] = x * g
        gains_db[k] = round(20 * np.log10(g), 2)
    mix = sum(stems.values())
    tp = qa.true_peak_dbfs(mix)
    if tp > TRUE_PEAK_MAX_DBFS:
        k = 10 ** ((TRUE_PEAK_MAX_DBFS - 0.1 - tp) / 20)
        stems = {s: x * k for s, x in stems.items()}
        mix = mix * k
        gains_db = {s: round(v + 20 * np.log10(k), 2) for s, v in gains_db.items()}

    files = {}
    for k, x in stems.items():
        path = os.path.join(out, "stems", f"{k}.wav")
        write_wav24(path, x)
        files[k] = path
    write_wav24(os.path.join(out, "preview_mix.wav"), mix)
    q = qa.run(stems, mix, tm, arc, slots, onsets, th)
    if ai:
        from .conform import beat_alignment_ms
        attack = beat_alignment_ms(stems["music"], tm, *arc.phases["plateau"])
        # gate: the conform's timing (envelope lag of the stretched AI drum guide against its
        # time-mapped source, per 8 s region; the music shares its stretcher) plus the AI grid fit.
        # The attack-based offset on the music is reported only: it reads ~10 ms even unstretched.
        lag = ai["stretch"]["residual_ms"]
        al = dict(conform_lag_ms=lag, ai_grid_fit_ms=ai["grid"]["fit_residual_ms"], music_attack_offset_ms=attack)
        al_ok = lag["median"] <= 15 and lag["max"] <= 30 and ai["grid"]["fit_residual_ms"] <= 15
        share = 20 * np.log10(_rms(qa.band_limit(stems["music"], slots["am_band"]["lo_hz"], slots["am_band"]["hi_hz"]))
                              / _rms(qa.band_limit(stems["pad"], slots["am_band"]["lo_hz"], slots["am_band"]["hi_hz"])))
        q["hybrid"] = dict(beat_alignment_ms=al, music_am_band_share_db=round(float(share), 1),
                           ai_tempo_bpm=ai["grid"]["bpm"], ai_key=ai["key"]["ks_major_tonic"],
                           pitch_set_ok=ai["key"]["pitch_set_ok"], stretch=ai["stretch"], motif=ai["motif"],
                           dropped_ai_bars=ai["dropped_bars"],
                           low_confidence_bars=sum(1 for c in chords if c["quality"] == "fallback"))
        q["checks"]["beat_alignment"] = bool(al_ok)
        q["tags"]["lever_compatible"] = bool(q["tags"]["lever_compatible"] and al_ok)

    def dump(name, obj):
        with open(os.path.join(out, name), "w") as fh:
            json.dump(obj, fh, indent=2, default=float)

    dump("tempo_map.json", tm.export())
    dump("arc.json", arc.export())
    dump("chords.json", dict(source="AI track (analyze.bar_chords)" if ai else "procedural (theory.chord_timeline)",
                             key=params.key_root, mode=params.mode,
                             chords=[{k: v for k, v in c.items()} for c in chords]))
    dump("pad_partials.json", dict(timbre=params.pad_timbre, tuning=params.pad_tuning, chords=rows))
    dump("qa_report.json", q)
    manifest = dict(
        bed_id=bid, type="energizer", generated_at=now_utc(),
        generator=f"resonance_utils {__version__} / energizer {GENERATOR_VERSION}",
        params=dataclasses.asdict(params), lever_slots=slots,
        length_s=arc.length_s, fs=FS, bits=24, channels=2,
        tempo=dict(entry_bpm=round(tm.points[0][1], 3), plateau_bpm=params.plateau_bpm),
        stems={k: dict(path=f"stems/{k}.wav", gain_db=gains_db[k], target_rms_dbfs=targets[k],
                       sha256=sha256(p)) for k, p in files.items()},
        preview_mix=dict(path="preview_mix.wav"),
        sidecars=["tempo_map.json", "arc.json", "chords.json", "pad_partials.json", "qa_report.json"],
        ai_generated=bool(ai),
        licence=("music + motif: Eleven Music (paid plan; confirm in-app streaming of modified tracks before "
                 "shipping); everything else procedural, own work") if ai else "procedural, own work",
        api_requests=([dict(endpoint="/v1/music/stem-separation", request=ai["separation"]["request"])]) if ai else [],
        reference_mix="reference_mix.wav" if ai else None,
        qa_tags=q["tags"], built=True,
    )
    dump("manifest.json", manifest)
    log(f"  stems {', '.join(stems)}; true peak {q['true_peak_dbfs']} dBFS; slot residual "
        f"{q['mb_slot_residual_db']['mix']} dB; pad native mod {q['native_mod_db']['pad_am_band']} dB; "
        f"grid {q['grid_error_ms']} ms")
    log(f"  tags {q['tags']}  ({time.time() - t_start:.0f} s) -> {out}/")
    return out, manifest


def tm_ref(params, arc):
    return tempomod.from_arc(arc, params.plateau_bpm, params.entry_tempo_ratio)


def render_reference(params, out_root, dry_run=False, log=print):
    """M2 (c): the un-separated Eleven Music tracks with only the arc gain applied. Not
    lever-ready (no separation, no notch, no time-map: the track stays at plateau tempo); for
    judging the AI music before the hybrid is built from the same cached calls.

    Timeline: motif from 0:00, fading out by 0:30; main track from 0:30 - pre-roll to the end."""
    from . import el_music
    slots = lever_slots()
    bid = bed_id(params, slots, dict(music=el_music.MODEL_ID))
    arc = arcmod.load(params.arc_profile, params.length_s)
    main, motif = el_music.main_plan(params, arc, tm_ref(params, arc)), el_music.motif_plan(params)
    secs = el_music.plan_seconds(main) + (el_music.plan_seconds(motif) if params.motif else 0)
    log(f"  reference {bid}: {params.music_style}, {params.key_root} {params.mode}, {params.plateau_bpm:g} BPM; "
        f"{secs:.0f} s of music, ~{secs / 60 * el_music.CREDITS_PER_MIN:,.0f} credits if not cached")
    if dry_run:
        el_music.compose(main, params.seed, None, dry_run=True, log=log)
        if params.motif:
            el_music.compose(motif, params.seed, None, dry_run=True, log=log)
        return None, None
    out = os.path.join(out_root, bid)
    os.makedirs(out, exist_ok=True)
    from ..core.audio_io import read_wav
    calls = {"music": el_music.compose(main, params.seed, os.path.join(out, "music_raw.wav"), log=log)}
    if params.motif:
        calls["motif"] = el_music.compose(motif, params.seed, os.path.join(out, "motif_raw.wav"), log=log)
    n = int(round(arc.length_s * FS))
    n_cr = int(arc.length_s * arcmod.CR) + 2
    starts = dict(music=arc.phases["ramp"][0] - el_music.PREROLL_S, motif=0.0)
    mix = np.zeros((n, 2))
    for k in calls:
        x = read_wav(os.path.join(out, f"{k}_raw.wav"))
        s0 = int(round(starts[k] * FS))
        m = min(len(x), n - s0)
        y = np.zeros((n, 2))
        y[s0:s0 + m] = x[:m]
        mix += fx.apply_gain(y, arc.gain_curve(k, n_cr))
    pk = np.max(np.abs(mix))
    mix *= min(10 ** (-20 / 20) / _rms(mix[int(arc.phases["plateau"][0] * FS):int(arc.phases["plateau"][1] * FS)]),
               10 ** (-1.5 / 20) / (pk + 1e-12))
    write_wav24(os.path.join(out, "reference_mix.wav"), mix)
    manifest = dict(bed_id=bid, type="energizer", kind="reference", generated_at=now_utc(),
                    generator=f"resonance_utils {__version__} / energizer {GENERATOR_VERSION}",
                    params=dataclasses.asdict(params), length_s=arc.length_s, starts_s=starts,
                    api_requests=calls, ai_generated=True, lever_ready=False,
                    note="listening reference only: un-separated AI tracks, arc gain only, no time-map")
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    log(f"  -> {out}/reference_mix.wav (+ music_raw.wav, motif_raw.wav)")
    return out, manifest

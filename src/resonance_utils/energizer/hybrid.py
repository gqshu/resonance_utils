"""The ElevenLabs layer of a hybrid bed (plan §5, §7): compose -> separate -> analyse -> conform.

Returns the `music` and `motif` stems in bed time and the bed's chord timeline (from the AI
track's chords), which then drives the procedural pad and bass. Every API call is cached, so a
re-render of the same params costs nothing.
"""

import json
import os

import numpy as np

from ..core.audio_io import FS, read_wav, write_wav24
from . import analyze, conform, el_music
from .stretch import stretch_to_map
from .theory import NOTE_PC, PC_NAME

MOTIF_FIRST_BEAT = 4                      # bed beat of the motif's first beat (bar 2: bar 1 is in the fade-in)


def fallback_chord(key_root):
    pc = NOTE_PC[key_root]
    return dict(label=PC_NAME[pc] + "5add9", root_pc=pc, third=None, quality="fallback",
                pcs=sorted({pc, (pc + 7) % 12, (pc + 2) % 12}), confidence=None)


def bed_chords(pairs, ai_bars, tempo, key_root, conf_min):
    """Merge per-bar chords (bed bars) into segments; bars without AI music, or whose chord
    confidence is below conf_min, play the key's root + fifth + ninth (fits any diatonic chord)."""
    by_ai = {r["bar"]: r for r in ai_bars}
    bed = {}
    for ai_bar, bed_bar in pairs:
        r = by_ai.get(min(ai_bar, len(ai_bars) - 1))
        if r is None or (r["confidence"] or 0) < conf_min:
            bed[bed_bar] = dict(fallback_chord(key_root), confidence=r["confidence"] if r else None)
        else:
            bed[bed_bar] = {k: r[k] for k in ("label", "root_pc", "third", "quality", "pcs", "confidence")}
    n_bars = int(np.ceil(tempo.total_beats / 4))
    rows = []
    for m in range(n_bars):
        c = bed.get(m, fallback_chord(key_root))
        b0, b1 = 4.0 * m, min(4.0 * (m + 1), tempo.total_beats)
        if rows and rows[-1]["label"] == c["label"]:
            rows[-1]["end_beat"] = b1
            rows[-1]["end_s"] = round(float(tempo.time_of_beat(b1)), 6)
            continue
        rows.append(dict(c, bar=m + 1, start_beat=b0, end_beat=b1,
                         start_s=round(float(tempo.time_of_beat(b0)), 6),
                         end_s=round(float(tempo.time_of_beat(b1)), 6)))
    return rows


def conform_motif(path, tempo, params, slots, th, n, fallback_lag_ms=None):
    x = read_wav(path)
    entry = params.plateau_bpm * params.entry_tempo_ratio
    g = analyze.beat_grid(analyze.onset_envelope(x), entry, len(x) / FS, audio=x)
    bed_per = 60.0 / float(tempo.bpm_at(0.0))
    first = g["t0"]
    k = np.arange(int((len(x) / FS - first) / g["period_s"]))
    ai_in = (first + k * g["period_s"]) * FS
    bed_out = np.array([tempo.time_of_beat(MOTIF_FIRST_BEAT + kk) for kk in k]) * FS
    keep = bed_out < n
    y, info = stretch_to_map(x, ai_in[keep], bed_out[keep], fallback_lag_ms=fallback_lag_ms)
    full = np.zeros((n, 2))
    s0 = int(round(bed_out[0]))
    m = min(len(y), n - s0)
    full[s0:s0 + m] = y[:m]
    info.update(bpm=g["bpm"], tempo_ok=g["tempo_ok"], first_beat_s=g["t0"], bed_entry_bpm=round(60 / bed_per, 3))
    return conform.band_handling(full, slots, th), info


def ai_layer(params, arc, tempo, out, slots, th, n, log=print):
    from .render import render_reference
    render_reference(params, os.path.dirname(out), log=log)          # compose (cached) + reference mix
    main, phases = el_music.build_main(params, arc, tempo)
    stems_p, sep = el_music.separate(os.path.join(out, "music_raw.wav"), os.path.join(out, "ai_stems"), log=log)
    stems = {k: read_wav(p) for k, p in stems_p.items()}
    music, grid, key, bars = analyze.analyse_track(stems, params.plateau_bpm, params.key_root, params.mode)
    log(f"  AI track: {grid['bpm']:.2f} BPM (tracked on {grid['tracked_on']}, residual {grid['fit_residual_ms']} ms), "
        f"key {key['ks_major_tonic']} (pitch set {'ok' if key['pitch_set_ok'] else 'MISMATCH'}), "
        f"{len(bars)} bars: {' '.join(r['label'] for r in bars[:8])} ...")
    if not key["pitch_set_ok"]:
        raise SystemExit(f"AI track is not in {params.key_root} {params.mode} (its pitch set is shifted by "
                         f"{key['pitch_set_shift']} semitones): the pad and bass would follow the wrong key. "
                         "Regenerate with another --seed or --music-variant")
    if not grid["tempo_ok"]:
        raise SystemExit(f"AI track tempo {grid['bpm']} is outside +/-3% of {params.plateau_bpm}: "
                         "regenerate with another --seed")
    pairs, ai_bar_t = conform.bar_map(grid, conform.ai_sections(main, el_music.PREROLL_S, phases), tempo, arc,
                                      el_music.PREROLL_S)
    edit, starts = conform.splice(music, pairs, ai_bar_t, grid["period_s"])
    drums = next((v for k, v in stems.items() if "drum" in k), None)
    guide = conform.splice(drums, pairs, ai_bar_t, grid["period_s"])[0] if drums is not None else None
    music_bed, sinfo = conform.time_map(edit, starts, pairs, grid["period_s"], tempo, n, th["MAX_STRETCH"],
                                        guide=guide)
    music_bed = conform.band_handling(music_bed, slots, th)
    n_ai = len(bars)
    used = sorted({a for a, _ in pairs})
    log(f"  conform: {len(pairs)} bed bars from {len(used)} of {n_ai} AI bars "
        f"(dropped {n_ai - len(used)}), stretch {sinfo['stretch']}, map residual {sinfo['residual_ms']} ms")
    out_stems = dict(music=music_bed)
    minfo = None
    if params.motif and os.path.exists(os.path.join(out, "motif_raw.wav")):
        lat = float(np.mean(sinfo["latency_ms"]))           # same engine, ratio ~1: reuse the music's latency
        out_stems["motif"], minfo = conform_motif(os.path.join(out, "motif_raw.wav"), tempo, params, slots, th, n,
                                                  fallback_lag_ms=lat)
    chords = bed_chords(pairs, bars, tempo, params.key_root, th["CHORD_CONF_MIN"])
    with open(os.path.join(out, "ai_analysis.json"), "w") as fh:
        json.dump(dict(grid=grid, key=key, ai_bars=bars, bar_map=pairs, separation=sep,
                       music_stems=el_music.music_stem_names(stems), stretch=sinfo, motif=minfo),
                  fh, indent=2, default=float)
    write_wav24(os.path.join(out, "ai_stems", "music_conformed.wav"), music_bed)
    return dict(stems=out_stems, chords=chords, grid=grid, key=key, stretch=sinfo, motif=minfo,
                separation=sep, dropped_bars=n_ai - len(used))

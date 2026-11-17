"""Analysis of the AI track (plan §7.1-7.2), numpy/scipy only (no librosa: its llvmlite has no
Intel-Mac wheel for Python 3.13, and no madmom models: their licence is non-commercial).

    onset_envelope  log-magnitude spectral flux at 100 frames/s
    beat_grid       constant-tempo grid: autocorrelation tempo near the requested BPM, comb
                    phase, then a weighted least-squares fit to the onset peaks; downbeat phase
    key_check       pitch-set match of the requested mode (and the Krumhansl tonic)
    bar_chords      bar-synchronous chroma -> diatonic triad templates (+ bass-stem root
                    evidence) -> Viterbi smoothing; one chord per bar with a confidence
"""

import numpy as np
from scipy import fft as sfft

from ..core.audio_io import FS
from .theory import MODES, PC_NAME, triad

ENV_FPS = 100
TEMPO_TOL = 0.03
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])


def _mono(x):
    return x.mean(1) if x.ndim == 2 else x


def onset_envelope(x, n_fft=2048, lo_hz=30.0, hi_hz=8000.0):
    m = np.concatenate([np.zeros(n_fft // 2), _mono(x)])   # frame i is centred on i / ENV_FPS
    hop = FS // ENV_FPS
    n_frames = 1 + (len(m) - n_fft) // hop
    win = np.hanning(n_fft)
    f = sfft.rfftfreq(n_fft, 1 / FS)
    band = (f >= lo_hz) & (f <= hi_hz)
    out = np.zeros(n_frames)
    prev = None
    for c0 in range(0, n_frames, 2000):
        idx = np.arange(c0, min(n_frames, c0 + 2000))
        frames = np.stack([m[i * hop:i * hop + n_fft] for i in idx]) * win
        mag = np.log1p(100 * np.abs(sfft.rfft(frames, axis=1))[:, band])
        full = mag if prev is None else np.vstack([prev[None], mag])
        d = np.maximum(np.diff(full, axis=0), 0).sum(1)     # flux into each frame after the first
        out[idx[-len(d):]] = d
        prev = mag[-1]
    out -= np.convolve(out, np.ones(50) / 50, mode="same")          # remove the slow trend
    return np.maximum(out, 0)


def _interp(env, t):
    return np.interp(t * ENV_FPS, np.arange(len(env)), env, left=0, right=0)


def attack_times(x, approx_s, win_s=0.03):
    """Precise onset time near each approximate time: the steepest rise of the 1 ms log-energy
    envelope within +/- win_s. Returns (times_s, strengths). Unbiased, unlike frame-based flux."""
    from ..core.analysis import power_frames
    le = 10 * np.log10(power_frames(_mono(x)) + 1e-12)
    le = np.convolve(le, np.ones(3) / 3, mode="same")
    d = np.diff(le)                                     # d[j]: rise into 1 ms block j + 1
    w = int(win_s * 1000)
    t_out, s_out = [], []
    for t in approx_s:
        c = int(round(t * 1000))
        a, b = max(0, c - w), min(len(d), c + w + 1)
        if b <= a:
            t_out.append(t)
            s_out.append(0.0)
            continue
        j = a + int(np.argmax(d[a:b]))
        t_out.append((j + 1) / 1000.0)
        s_out.append(float(d[j]))
    return np.asarray(t_out), np.asarray(s_out)


def beat_grid(env, bpm_expected, length_s, low_env=None, audio=None):
    """Constant-tempo grid. Returns dict(bpm, t0 (first beat, s), downbeat (beat index offset of
    the first bar line), n_beats, fit_residual_ms, tempo_ok)."""
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lo, hi = 60 / (bpm_expected * (1 + TEMPO_TOL)), 60 / (bpm_expected * (1 - TEMPO_TOL))
    lags = np.arange(int(lo * ENV_FPS) - 1, int(hi * ENV_FPS) + 2)
    k = lags[np.argmax(ac[lags])]
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    period = (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)) / ENV_FPS
    # comb phase
    phases = np.arange(0, period, 0.005)
    n = int(length_s / period)
    kk = np.arange(n)
    score = [(_interp(env, ph + kk * period)).sum() for ph in phases]
    ph = phases[int(np.argmax(score))]
    # least squares on local onset peaks
    pred = ph + kk * period
    pk_t, pk_w = [], []
    for t in pred:
        a, b = int((t - 0.07) * ENV_FPS), int((t + 0.07) * ENV_FPS) + 1
        if a < 0 or b > len(env):
            pk_t.append(t)
            pk_w.append(0.0)
            continue
        j = a + int(np.argmax(env[a:b]))
        pk_t.append(j / ENV_FPS)
        pk_w.append(env[j])
    w = np.asarray(pk_w)
    strong = w > np.percentile(w, 40)
    A = np.stack([np.ones(n), kk], 1)[strong] * np.sqrt(w[strong])[:, None]
    sol = np.linalg.lstsq(A, np.asarray(pk_t)[strong] * np.sqrt(w[strong]), rcond=None)[0]
    t0, per = float(sol[0]), float(sol[1])
    while t0 - per >= 0:
        t0 -= per
    while t0 < 0:
        t0 += per
    resid = np.abs(np.asarray(pk_t)[strong] - (sol[0] + kk[strong] * per))
    if audio is not None:                              # refine on sample-accurate attacks
        kk2 = np.arange(int((length_s - t0) / per))
        bt = t0 + kk2 * per
        at, sg = attack_times(audio, bt, 0.04)
        m = _mono(audio)
        lvl = np.array([np.sqrt(np.mean(m[max(0, int((t - 0.1) * FS)):int((t + 0.1) * FS)] ** 2) + 1e-20) for t in bt])
        active = 20 * np.log10(lvl) > -60                # sections where this source plays at all
        if active.sum() >= 16:
            good = active & (sg > np.percentile(sg[active], 50))
            for _ in range(3):                           # robust: refit without outliers
                A = np.stack([np.ones(good.sum()), kk2[good]], 1) * np.sqrt(sg[good])[:, None]
                sol = np.linalg.lstsq(A, at[good] * np.sqrt(sg[good]), rcond=None)[0]
                r = at - (sol[0] + kk2 * sol[1])
                mad = np.median(np.abs(r[good] - np.median(r[good])))
                good = good & (np.abs(r - np.median(r[good])) <= 3 * mad + 0.002)
            resid = np.abs(at[good] - (sol[0] + kk2[good] * sol[1]))   # before t0 wraps by a period
            t0, per = float(sol[0]), float(sol[1])
            while t0 < 0:
                t0 += per
    nb = int((length_s - t0) / per) + 1
    beats = t0 + np.arange(nb) * per
    if callable(low_env):                              # e.g. bass-chroma novelty per beat
        nov = low_env(beats, per)
        db_scores = [float(np.mean(nov[p::4])) for p in range(4)]
    else:
        src = low_env if low_env is not None else env
        db_scores = [float(_interp(src, beats[p::4]).mean()) for p in range(4)]
    bpm = 60 / per
    return dict(bpm=round(bpm, 4), period_s=per, t0=round(t0, 5), n_beats=nb, downbeat=int(np.argmax(db_scores)),
                downbeat_scores=[round(s, 3) for s in db_scores],
                fit_residual_ms=round(float(np.median(resid)) * 1000, 1),
                tempo_ok=abs(bpm / bpm_expected - 1) <= TEMPO_TOL)


def chroma(x, lo_hz=60.0, hi_hz=2000.0):
    """12-bin chroma of one segment (magnitude, cos^2-weighted around each semitone), L2-normed."""
    m = _mono(x)
    if len(m) < 2048:
        return np.zeros(12)
    S = np.abs(sfft.rfft(m * np.hanning(len(m))))
    f = sfft.rfftfreq(len(m), 1 / FS)
    sel = (f >= lo_hz) & (f <= hi_hz)
    midi = 69 + 12 * np.log2(f[sel] / 440.0)
    near = np.round(midi)
    w = np.cos(np.pi * (midi - near)) ** 2 * S[sel]
    c = np.bincount((near.astype(int) % 12), weights=w, minlength=12)
    return c / (np.linalg.norm(c) + 1e-12)


def key_check(x, key_root, mode):
    """Does the track's pitch set match the requested mode? (All 7 modes of one scale share a pitch
    set, so the check is on the set; the tonic is reported from Krumhansl-Schmuckler.)"""
    from .theory import NOTE_PC
    c = chroma(x)
    root = NOTE_PC[key_root]
    scale = np.zeros(12)
    scale[[(root + s) % 12 for s in MODES[mode]]] = 1
    scores = [float(np.dot(np.roll(scale, r), c)) for r in range(12)]
    best = int(np.argmax(scores))
    margin = 0.98                                        # ambiguous sets (e.g. no G or G# at all) pass
    ks = [float(np.corrcoef(np.roll(KS_MAJOR, r), c)[0, 1]) for r in range(12)]
    return dict(pitch_set_ok=bool(scores[0] >= margin * scores[best]), pitch_set_shift=best, pitch_set_scores=[round(s, 3) for s in scores],
                ks_major_tonic=PC_NAME[int(np.argmax(ks))], chroma=[round(float(v), 3) for v in c])


def diatonic_chords(key_root, mode):
    from .theory import NOTE_PC
    kp = NOTE_PC[key_root]
    out = []
    for d in range(7):
        r, third, fifth = triad(mode, d)
        if fifth != 7:
            continue
        pc = (kp + r) % 12
        q = "major" if third == 4 else "minor"
        tpl = np.zeros(12)
        tpl[pc], tpl[(pc + third) % 12], tpl[(pc + 7) % 12] = 1.0, 0.8, 0.9
        out.append(dict(degree=d + 1, root_pc=pc, third=third, quality=q,
                        label=PC_NAME[pc] + ("" if q == "major" else "m"),
                        pcs=sorted({pc, (pc + third) % 12, (pc + 7) % 12}), tpl=tpl / np.linalg.norm(tpl)))
    return out


def bass_novelty(bass, band=(30.0, 300.0)):
    """Per-beat change of the bass stem's chroma: chord changes happen on bar lines."""
    def f(beats, per):
        cs = [chroma(bass[int(t * FS):int((t + per) * FS)], *band) for t in beats]
        nov = [0.0] + [float(1 - np.dot(a, b)) for a, b in zip(cs[:-1], cs[1:])]
        return np.asarray(nov)
    return f


def analyse_track(stems, bpm_expected, key_root, mode):
    """Grid, key and bar chords from the separated AI stems ({name: (n, 2)}). Beats are tracked
    on the drum stem when it has content (exact here: ~3 ms residual), else on the mix; the
    downbeat comes from where the bass changes; chord roots lean on the bass stem."""
    from .el_music import music_stem_names
    music = sum(stems[k] for k in music_stem_names(stems))
    drums = next((v for k, v in stems.items() if "drum" in k), None)
    bass = next((v for k, v in stems.items() if "bass" in k), None)
    level = lambda x: 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12) if x is not None else -200
    src = drums if level(drums) > -40 else music + (bass if bass is not None else 0)
    length = len(music) / FS
    grid = beat_grid(onset_envelope(src), bpm_expected, length,
                     low_env=bass_novelty(bass) if level(bass) > -45 else None, audio=src)
    grid["tracked_on"] = "drums" if src is drums else "mix"
    key = key_check(music, key_root, mode)
    bars = bar_chords(music, bass if level(bass) > -45 else None, grid, key_root, mode)
    return music, grid, key, bars


def bar_chords(music, bass, grid, key_root, mode, switch_penalty=0.08, bass_weight=1.0, bass_band=(30.0, 300.0)):
    """One chord per AI bar (bar j starts at beat downbeat + 4j). Returns rows with bar, start_s,
    end_s, label, root_pc, third, quality, pcs, confidence (0-1)."""
    per, t0 = grid["period_s"], grid["t0"]
    first = grid["downbeat"]
    cands = diatonic_chords(key_root, mode)
    bars = []
    b = first
    while b + 4 <= grid["n_beats"]:
        s, e = t0 + b * per, t0 + (b + 4) * per
        seg = slice(int(s * FS), int(e * FS))
        cm, cb = chroma(music[seg]), chroma(bass[seg], *bass_band) if bass is not None else np.zeros(12)
        sc = np.array([float(np.dot(c["tpl"], cm)) + bass_weight * float(cb[c["root_pc"]]) for c in cands])
        bars.append(dict(bar=len(bars), start_beat=b, start_s=s, end_s=e, scores=sc,
                         energy=float(np.sqrt(np.mean(_mono(music[seg]) ** 2)))))
        b += 4
    if not bars:
        return []
    S = np.array([r["scores"] for r in bars])
    n, k = S.shape
    dp, bp = S[0].copy(), np.zeros((n, k), int)
    for i in range(1, n):
        trans = dp[:, None] - switch_penalty * (1 - np.eye(k))
        bp[i] = np.argmax(trans, 0)
        dp = trans[bp[i], np.arange(k)] + S[i]
    path = [int(np.argmax(dp))]
    for i in range(n - 1, 0, -1):
        path.append(int(bp[i][path[-1]]))
    path = path[::-1]
    quiet = np.percentile([r["energy"] for r in bars], 10) * 0.3
    out = []
    for r, j in zip(bars, path):
        c = cands[j]
        srt = np.sort(r["scores"])[::-1]
        conf = float(np.clip((srt[0] - srt[1]) / (abs(srt[0]) + 1e-9) * 4, 0, 1))
        if r["energy"] < quiet:
            conf = 0.0
        out.append(dict(bar=r["bar"], start_beat=r["start_beat"], start_s=round(r["start_s"], 4),
                        end_s=round(r["end_s"], 4), label=c["label"], degree=c["degree"], root_pc=c["root_pc"],
                        third=c["third"], quality=c["quality"], pcs=c["pcs"], confidence=round(conf, 3)))
    return out


def bar_features(x, grid):
    """Per bar: 12-bin chroma (unit) + 8 octave-band levels (dB re the bar's total, /20)."""
    from ..core.analysis import octave_balance
    per, t0, db = grid["period_s"], grid["t0"], grid["downbeat"]
    feats, b = [], db
    while b + 4 <= grid["n_beats"]:
        seg = x[int((t0 + b * per) * FS):int((t0 + (b + 4) * per) * FS)]
        xs = seg if seg.ndim == 2 else np.stack([seg, seg], 1)
        feats.append(np.concatenate([chroma(seg), octave_balance(xs) / 20.0]))
        b += 4
    return np.asarray(feats)


def repetition_report(x, bpm_expected, key_root, mode, window=8):
    """How repetitive is a track? On the unseparated mix: chord movement (the mix's 30-300 Hz band
    stands in for the bass), novelty events (bars that differ from the previous `window` bars in
    harmony + timbre), and how far loudness and brightness move from bar to bar."""
    lowband = (30.0, 140.0)                            # bass register, below the keys
    g = beat_grid(onset_envelope(x), bpm_expected, len(x) / FS, audio=x, low_env=bass_novelty(x, lowband))
    F = bar_features(x, g)
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    U = Fz / (np.linalg.norm(Fz, axis=1, keepdims=True) + 1e-12)
    n = len(U)
    nov = np.array([1 - float(U[i] @ (U[i - window:i].mean(0) / (np.linalg.norm(U[i - window:i].mean(0)) + 1e-12)))
                    for i in range(window, n)])
    thr = np.median(nov) + 2 * np.median(np.abs(nov - np.median(nov)))
    events = int(np.sum((nov[1:-1] > thr) & (nov[1:-1] >= nov[:-2]) & (nov[1:-1] >= nov[2:])))
    bars = bar_chords(x, x, g, key_root, mode, bass_band=lowband)
    labels = [r["label"] for r in bars]
    changes = sum(a != b for a, b in zip(labels, labels[1:]))
    minutes = len(x) / FS / 60
    seq = "".join(chr(65 + sorted(set(labels)).index(lab)) for lab in labels)
    period = next((p for p in range(1, 33)                    # repeats with <= 10 % of bars deviating
                   if np.mean([seq[i] != seq[i + p] for i in range(len(seq) - p)]) <= 0.1), None)
    per, t0, db = g["period_s"], g["t0"], g["downbeat"]
    lv, br = [], []
    from scipy import fft as _f
    for j in range(n):
        seg = (x.mean(1) if x.ndim == 2 else x)[int((t0 + (db + 4 * j) * per) * FS):int((t0 + (db + 4 * j + 4) * per) * FS)]
        P = np.abs(_f.rfft(seg)) ** 2
        f = _f.rfftfreq(len(seg), 1 / FS)
        lv.append(10 * np.log10(P.sum() / len(seg) ** 2 + 1e-20))
        br.append(np.log2(max((f * P).sum() / (P.sum() + 1e-20), 1.0)))
    rng = lambda v: round(float(np.percentile(v, 90) - np.percentile(v, 10)), 2)
    return dict(
        bpm=g["bpm"], bars=n, distinct_chords=len(set(labels)), chord_changes_per_min=round(changes / minutes, 1),
        chord_pattern_period_bars=period, chords=" ".join(labels[:24]) + (" ..." if len(labels) > 24 else ""),
        novelty_events_per_min=round(events / minutes, 1),
        loudness_range_db=rng(lv), brightness_range_oct=rng(br),
        timbre_spread_db=round(float(np.mean(F[:, 12:].std(0)) * 20), 2),
    )

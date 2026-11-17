"""Time-map stretching with Rubber Band (pylibrb, R3 'finer' engine). GPL: this is an internal
offline tool; the output audio is not covered, and Rubber Band must never ship in the app.

The binding can't pass a key-frame map, so the stretcher runs in real-time mode with a ratio set
per block: each anchor segment gets the ratio that lands it on its target output sample, recomputed
from the samples actually produced, so error never accumulates.

Real-time mode adds a latency that the library under-reports and that varies with the ratio
(about 80-90 ms here). So the map is calibrated: a pilot click track with one click per anchor
is stretched first with the same map, each click's output position gives that anchor's latency,
and the audio pass aims every anchor that much early. Rubber Band's timing does not depend on
content, so the audio then lands on the map (checked: pilot error after correction is reported)."""

import numpy as np

from ..core.audio_io import FS

BLOCK = 1024
DEFAULT_LATENCY_MS = 86.0   # measured 82-90 ms on every calibrated run so far: used when nothing correlates
CLICK = np.hanning(96)[48:]


def _run(src, ai, ao, ratios):
    import pylibrb as rb
    O = rb.Option
    ch = src.shape[0]
    st = rb.RubberBandStretcher(FS, ch, O.PROCESS_REALTIME | O.ENGINE_FINER | O.CHANNELS_TOGETHER,
                                initial_time_ratio=float(ratios[0]))
    st.set_max_process_size(BLOCK)
    st.process(np.zeros((ch, st.get_preferred_start_pad()), np.float32))
    delay = st.get_start_delay()
    out, produced, pos, seg, end = [], 0, int(ai[0]), 0, int(ai[-1])
    while pos < end:
        while seg < len(ratios) - 1 and pos >= ai[seg + 1]:
            seg += 1
        have = produced - delay + max(st.available(), 0)
        r = (ao[seg + 1] - ao[0] - have) / max(ai[seg + 1] - pos, 1)
        st.time_ratio = float(np.clip(r, ratios[seg] * 0.9, ratios[seg] * 1.1))
        n = int(max(1, min(BLOCK, ai[seg + 1] - pos, end - pos)))
        st.process(src[:, pos:pos + n], final=pos + n >= end)
        pos += n
        while st.available() > 0:
            y = st.retrieve(st.available())
            out.append(y)
            produced += y.shape[1]
    while st.available() > 0:
        out.append(st.retrieve(st.available()))
    return np.concatenate(out, 1).T.astype(np.float64)[delay:]


def _click_positions(y, targets, win=8000):
    env = np.abs(y)
    pos = []
    for t in targets:
        a, b = max(0, int(t) - win), min(len(env), int(t) + win)
        w = env[a:b]
        pos.append(a + int(np.argmax(w > 0.5 * w.max())) if len(w) and w.max() > 0 else int(t))
    return np.array(pos, float)


def _onset_env(x):
    """1 ms positive log-energy rise: a sharp, content-agnostic timing signature."""
    from ..core.analysis import power_frames
    le = 10 * np.log10(power_frames(x.mean(1) if x.ndim == 2 else x) + 1e-12)
    return np.maximum(np.diff(le, prepend=le[0]), 0)


def measure_lag(x, y, ai, ao, win_s=8.0, step_s=2.0, max_lag_ms=150):
    """Lag (output samples, + = late) of y against x warped through the map (ai -> ao), per
    region: cross-correlation of 1 ms onset envelopes in win_s windows every step_s. Returns
    (centres_out_samples, lags_samples, peak_corr)."""
    ex, ey = _onset_env(x), _onset_env(y)
    rel = ao - ao[0]
    t_out = np.arange(len(ey)) / 1000.0 * FS                 # output sample of each 1 ms frame
    t_in = np.interp(t_out, rel, ai)                         # where the map says it came from
    exp = np.interp(t_in / FS * 1000.0, np.arange(len(ex)), ex)
    L = max_lag_ms
    w, st = int(win_s * 1000), int(step_s * 1000)
    cs, lags, peaks = [], [], []
    for c0 in range(L, len(ey) - w - L, st):
        a = exp[c0:c0 + w] - exp[c0:c0 + w].mean()
        if np.std(a) == 0:
            continue
        sc = [float(np.dot(a, ey[c0 + k:c0 + k + w] - ey[c0 + k:c0 + k + w].mean())) for k in range(-L, L + 1)]
        k = int(np.argmax(sc))
        norm = np.linalg.norm(a) * np.linalg.norm(ey[c0 + k - L:c0 + k - L + w]) + 1e-12
        cs.append((c0 + w / 2) / 1000.0 * FS)
        lags.append((k - L) / 1000.0 * FS)
        peaks.append(sc[k] / norm)
    return np.asarray(cs), np.asarray(lags), np.asarray(peaks)


def stretch_to_map(x, anchors_in, anchors_out, max_stretch=None, calib_channels=None, fallback_lag_ms=None):
    """x: (n, ch). anchors_in / anchors_out: increasing sample positions, at least ~0.2 s apart.
    Input before anchors_in[0] and after anchors_in[-1] is dropped. Returns (y, info): y covers
    anchors_out[0]..anchors_out[-1] (sample 0 = anchors_out[0]); info has the residual error.

    Calibration, both passes on the audio itself (Rubber Band's real-time latency depends on
    content: a click pilot read ~1 ms while music was 23 ms early): pass 1 runs the map as given,
    the per-region lag of the output against the warped input is measured by envelope
    cross-correlation, and pass 2 runs the map shifted by that lag curve."""
    ai, ao = np.asarray(anchors_in, float), np.asarray(anchors_out, float)
    ratios = np.diff(ao) / np.diff(ai)
    if max_stretch and (ratios.max() > max_stretch or ratios.min() < 1 / max_stretch):
        raise ValueError(f"stretch {ratios.min():.3f}-{ratios.max():.3f} exceeds MAX_STRETCH {max_stretch}")
    src = np.ascontiguousarray(x.T, dtype=np.float32)
    rel = ao - ao[0]
    want = int(round(rel[-1]))

    def run(ao_target):
        r = np.diff(ao_target) / np.diff(ai)
        y = _run(src, ai, ao_target, r)
        return np.concatenate([y, np.zeros((max(0, want - len(y)), y.shape[1]))])[:want]

    cc = calib_channels if calib_channels is not None else slice(None)
    y1 = run(ao)
    c, lag, pk = measure_lag(x[:, cc], y1[:, cc], ai, ao)
    ok = pk > 0.2
    if ok.sum() >= 2:
        lag_at = np.interp(rel, c[ok], lag[ok])
    else:                                  # nothing to correlate on (e.g. a soft solo line)
        fb = (fallback_lag_ms if fallback_lag_ms is not None else DEFAULT_LATENCY_MS) / 1000 * FS
        lag_at = np.full(len(rel), fb)
    ao_c = ao - lag_at
    ao_c[0] = ao[0]
    y = run(ao_c)
    c2, lag2, pk2 = measure_lag(x[:, cc], y[:, cc], ai, ao)
    ok2 = pk2 > 0.2
    res = np.abs(lag2[ok2]) / FS * 1000 if ok2.any() else np.array([np.nan])
    info = dict(calibrated_on="audio" if ok.sum() >= 2 else "fallback latency",
                latency_ms=[round(float(lag_at.min()) / FS * 1000, 1), round(float(lag_at.max()) / FS * 1000, 1)],
                residual_ms=dict(median=round(float(np.median(res)), 2), max=round(float(np.max(res)), 2),
                                 regions=int(ok2.sum())),
                stretch=[round(float(ratios.min()), 4), round(float(ratios.max()), 4)])
    return y, info

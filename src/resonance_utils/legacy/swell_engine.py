#!/usr/bin/env python3
"""
swell_engine.py — real-time "swell" modulator for ANY nature soundscape,
driven by EEG state, for the power-nap induction phase.

Idea: texture and rhythm are separated. Your existing beds (river, rain,
wind, ocean, ...) supply the texture; this engine adds the rhythm live:

  * amplitude swell     0.1 -> 0.2 -> 0.25 Hz, asymmetric (rise < fall)
  * brightness swell    high band dips in the trough, opens at the crest
  * left-right sway     equal-power balance, synced to the swell
  * level fade          bed drops toward ~35 dBA after onset
  * carrier notch       static -6 dB, 1/3-octave around the BB carrier

All parameter changes glide (one-pole, per sample); the rate glides in the
log domain through a phase accumulator, so the rhythm speeds up without any
jump or click, at any moment, on any block size.

NOTE ON EEG: 0.1-0.25 Hz modulation sits far below your 2 Hz band-power floor,
so (unlike 1-12 Hz AM) it does not plant a steady-state peak in the theta/alpha
bins the controller reads. Never pan or modulate the binaural-beat tones
themselves — apply this engine to the BED bus only.

Pieces
  SwellEngine          block DSP, stateful, real-time safe (numpy + scipy)
  NapInductionController  EEG features -> SETTLE / DRIFT / ONSET / HOLD
  condition_source()   optional offline pre-pass: removes a recording's own
                       slow rhythm (gusts, waves) and tames transients
  CLI                  offline render with a scripted or simulated EEG
                       trajectory + CSV log of the exact stimulus envelope

Real-time use (e.g. sounddevice, 48 kHz, 512-frame callback):
    eng = SwellEngine(fs=48000, carrier=144, profile="river")
    ctl = NapInductionController()
    def on_eeg_epoch(feat):                         # every ~1-2 s
        eng.set_state(ctl.update(**feat))
    def callback(indata, outdata, frames, t, status):
        outdata[:] = eng.process(next_bed_block(frames))
Port to iOS: the per-block math maps 1:1 onto an AVAudioSourceNode /
AUAudioUnit render block (phase accumulator, one-pole smoothers, two biquads).

CLI examples
  python3 swell_engine.py river.wav out.wav --profile river \
          --script "0:SETTLE,150:DRIFT,420:ONSET,600:HOLD" --minutes 12
  python3 swell_engine.py rain.wav out.wav --profile rain --simulate --condition
"""

import argparse
import csv
import wave

import numpy as np
from scipy.signal import lfilter, lfilter_zi

# --------------------------------------------------------------------------
# Config constants — empirical-first: tune via factorial runs, not by belief
# --------------------------------------------------------------------------
STATE_TARGETS = {
    #            rate Hz  depth dB  tilt dB  pan width  level dB  rise frac
    "SETTLE": dict(rate=0.10, depth=8.0, tilt=6.0, pan=0.00, level=0.0,  rise=0.40),
    "DRIFT":  dict(rate=0.20, depth=6.0, tilt=5.0, pan=0.15, level=-3.0, rise=0.38),
    "ONSET":  dict(rate=0.25, depth=5.0, tilt=4.0, pan=0.25, level=-8.0, rise=0.35),
    "HOLD":   dict(rate=0.25, depth=3.0, tilt=3.0, pan=0.15, level=-10.0, rise=0.35),
}
GLIDE_TAU_S = dict(rate=20.0, depth=10.0, tilt=10.0, pan=15.0, level=40.0, rise=10.0)

# Per-soundscape scaling: how hard each texture can be swelled before it
# sounds processed. 'xover' = brightness crossover (Hz).
PROFILES = {
    "ocean":   dict(depth=1.0, tilt=1.0, pan=1.0, xover=1200.0),
    "river":   dict(depth=0.8, tilt=0.8, pan=1.0, xover=1500.0),
    "rain":    dict(depth=0.7, tilt=1.0, pan=0.8, xover=2500.0),
    "wind":    dict(depth=0.8, tilt=0.6, pan=1.0, xover=900.0),
    "default": dict(depth=0.8, tilt=0.8, pan=0.8, xover=1500.0),
    # event-rich beds (birds, crickets, fire, thunder): don't swell — use
    # them only as quiet garnish in SETTLE, or exclude.
    "events":  dict(depth=0.0, tilt=0.0, pan=0.0, xover=1500.0),
}

# Controller thresholds (on your (theta+delta)/(alpha+beta) slowing index,
# expressed in MAD units relative to the session baseline).
CTRL = dict(
    drift_enter=1.0,       # slowing index above this ...
    drift_exit=0.3,        # ... hysteresis: fall below this to back off
    dwell_s=30.0,          # sustained this long before advancing
    backoff_s=45.0,        # sustained this long before backing off
    onset_confirm_s=30.0,  # transition detector must hold this long
    hold_after_s=180.0,    # ONSET -> HOLD after this long
    drift_timeout_s=360.0, # SETTLE -> DRIFT anyway after 6 min (don't stall)
)


# --------------------------------------------------------------------------
# DSP helpers
# --------------------------------------------------------------------------
def butter2_lp(fc, fs):
    """RBJ 2nd-order Butterworth low-pass (b, a)."""
    w0 = 2 * np.pi * fc / fs
    al = np.sin(w0) / (2 * np.sqrt(0.5))
    c = np.cos(w0)
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2])
    a = np.array([1 + al, -2 * c, 1 - al])
    return b / a[0], a / a[0]


def peaking(fc, fs, gain_db, bw_oct):
    """RBJ peaking EQ (negative gain = notch-like cut)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    al = np.sin(w0) * np.sinh(np.log(2) / 2 * bw_oct * w0 / np.sin(w0))
    c = np.cos(w0)
    b = np.array([1 + al * A, -2 * c, 1 - al * A])
    a = np.array([1 + al / A, -2 * c, 1 - al / A])
    return b / a[0], a / a[0]


def wave_shape(phase, rise):
    """Asymmetric raised cosine: 0 trough, 1 crest; rise may be a vector."""
    ph = np.mod(phase, 1.0)
    rise = np.broadcast_to(rise, ph.shape)
    up = ph < rise
    out = np.empty_like(ph)
    out[up] = 0.5 - 0.5 * np.cos(np.pi * ph[up] / rise[up])
    dn = ~up
    out[dn] = 0.5 + 0.5 * np.cos(np.pi * (ph[dn] - rise[dn]) / (1.0 - rise[dn]))
    return out


class Smoother:
    """Per-sample one-pole glide, evaluated in closed form per block."""

    def __init__(self, value, tau_s, fs, log=False):
        self.log = log
        self.cur = np.log(value) if log else value
        self.target = self.cur
        self.a = np.exp(-1.0 / (tau_s * fs))

    def set(self, value):
        self.target = np.log(value) if self.log else value

    def block(self, n):
        k = np.arange(1, n + 1)
        v = self.target + (self.cur - self.target) * self.a ** k
        self.cur = v[-1]
        return np.exp(v) if self.log else v


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class SwellEngine:
    def __init__(self, fs=48000, carrier=144.0, profile="default", state="SETTLE",
                 notch_db=6.0):
        self.fs = fs
        self.prof = PROFILES[profile]
        self.state = state
        t = STATE_TARGETS[state]
        self.sm = {
            "rate": Smoother(t["rate"], GLIDE_TAU_S["rate"], fs, log=True),
            "depth": Smoother(t["depth"] * self.prof["depth"], GLIDE_TAU_S["depth"], fs),
            "tilt": Smoother(t["tilt"] * self.prof["tilt"], GLIDE_TAU_S["tilt"], fs),
            "pan": Smoother(t["pan"] * self.prof["pan"], GLIDE_TAU_S["pan"], fs),
            "level": Smoother(t["level"], GLIDE_TAU_S["level"], fs),
            "rise": Smoother(t["rise"], GLIDE_TAU_S["rise"], fs),
        }
        self.phase = 0.75   # start just past the crest's fall-off -> gentle entry
        self.n_done = 0
        # filters (stereo state)
        self.xb, self.xa = butter2_lp(self.prof["xover"], fs)
        self.xz = np.stack([lfilter_zi(self.xb, self.xa)] * 2, 1) * 0.0
        self.notch = carrier and carrier > 0
        if self.notch:
            self.nb, self.na = peaking(carrier, fs, -notch_db, 1.0 / 3.0)
            self.nz = np.stack([lfilter_zi(self.nb, self.na)] * 2, 1) * 0.0
        self.log = []

    # ---- control --------------------------------------------------------
    def set_state(self, state):
        if state == self.state:
            return
        self.state = state
        t = STATE_TARGETS[state]
        self.sm["rate"].set(t["rate"])
        self.sm["depth"].set(t["depth"] * self.prof["depth"])
        self.sm["tilt"].set(t["tilt"] * self.prof["tilt"])
        self.sm["pan"].set(t["pan"] * self.prof["pan"])
        self.sm["level"].set(t["level"])
        self.sm["rise"].set(t["rise"])

    # ---- audio ----------------------------------------------------------
    def process(self, x):
        """x: (n, 2) float block of the bed. Returns processed (n, 2)."""
        n = len(x)
        rate = self.sm["rate"].block(n)
        depth = self.sm["depth"].block(n)
        tilt = self.sm["tilt"].block(n)
        pan_w = self.sm["pan"].block(n)
        level = self.sm["level"].block(n)
        rise = self.sm["rise"].block(n)

        ph = self.phase + np.cumsum(rate) / self.fs
        self.phase = ph[-1] % 1.0
        env = wave_shape(ph, rise)

        if self.notch:
            x, self.nz = lfilter(self.nb, self.na, x, axis=0, zi=self.nz)
        low, self.xz = lfilter(self.xb, self.xa, x, axis=0, zi=self.xz)
        high = x - low                                         # sums back exactly
        hi_g = 10 ** (-tilt * (1.0 - env) / 20.0)              # darker in the trough
        y = low + high * hi_g[:, None]

        g = 10 ** ((-depth * (1.0 - env) + level) / 20.0)
        y = y * g[:, None]

        pan = pan_w * np.sin(2 * np.pi * ph)                   # one L-R sway per wave
        th = (pan + 1) * np.pi / 4
        mid = 0.5 * (y[:, 0] + y[:, 1])
        side = 0.5 * (y[:, 0] - y[:, 1])
        y = np.stack([np.sqrt(2) * np.cos(th) * mid + side,
                      np.sqrt(2) * np.sin(th) * mid - side], 1)

        self.log.append((self.n_done / self.fs, self.state, float(rate[-1]), float(ph[-1] % 1.0),
                         float(env[-1]), float(20 * np.log10(g[-1])), float(pan[-1])))
        self.n_done += n
        return y


# --------------------------------------------------------------------------
# EEG controller
# --------------------------------------------------------------------------
class NapInductionController:
    """Maps EEG epochs to induction states with dwell + hysteresis, so the
    audio never chases single-epoch noise (closed loops oscillate otherwise).

    update() inputs per epoch:
      t                 session time (s)
      slowing           (theta+delta)/(alpha+beta) slowing index, MAD units vs baseline
      transition        bool: your alpha-dropout + theta-onset detector fired
      quality_ok        bool: signal quality usable
    """

    ORDER = ["SETTLE", "DRIFT", "ONSET", "HOLD"]

    def __init__(self, cfg=CTRL):
        self.c = cfg
        self.state = "SETTLE"
        self.entered = 0.0
        self.above_since = None
        self.below_since = None
        self.trans_since = None

    def _go(self, s, t):
        self.state, self.entered = s, t
        self.above_since = self.below_since = self.trans_since = None

    def update(self, t, slowing, transition=False, quality_ok=True):
        c = self.c
        if not quality_ok:                      # bad signal -> freeze, never push
            return self.state
        self.above_since = (self.above_since or t) if slowing >= c["drift_enter"] else None
        self.below_since = (self.below_since or t) if slowing <= c["drift_exit"] else None
        self.trans_since = (self.trans_since or t) if transition else None

        s = self.state
        if s == "SETTLE":
            if (self.above_since and t - self.above_since >= c["dwell_s"]) or \
               (t - self.entered >= c["drift_timeout_s"]):
                self._go("DRIFT", t)
        elif s == "DRIFT":
            if self.trans_since and t - self.trans_since >= c["onset_confirm_s"]:
                self._go("ONSET", t)
            elif self.below_since and t - self.below_since >= c["backoff_s"]:
                self._go("SETTLE", t)
        elif s == "ONSET":
            if t - self.entered >= c["hold_after_s"]:
                self._go("HOLD", t)
            elif self.below_since and t - self.below_since >= c["backoff_s"]:
                self._go("DRIFT", t)            # woke up a bit: step back one
        elif s == "HOLD":
            if self.below_since and t - self.below_since >= c["backoff_s"] * 2:
                self._go("DRIFT", t)
        return self.state


# --------------------------------------------------------------------------
# Offline pre-conditioning of a source recording
# --------------------------------------------------------------------------
def condition_source(x, fs, crest_db=3.0):
    """Flatten the recording's own slow rhythm (0.5 s RMS) and cap transients
    at +crest_db above it, so the engine's rhythm is the only rhythm."""
    p = (x ** 2).mean(1)
    win = lambda s: np.ones(int(s * fs)) / int(s * fs)
    slow = np.sqrt(np.convolve(p, win(0.5), "same")) + 1e-6
    fast = np.sqrt(np.convolve(p, win(0.02), "same")) + 1e-6
    lim = np.minimum(1.0, slow * 10 ** (crest_db / 20) / fast)
    lim = np.convolve(lim, win(0.01), "same")
    y = x * (lim / slow)[:, None]
    return y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))


# --------------------------------------------------------------------------
# I/O + CLI
# --------------------------------------------------------------------------
def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, fs, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw == 2:
        x = np.frombuffer(raw, "<i2") / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        i = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(i >= 1 << 23, i - (1 << 24), i) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, "<i4") / 2 ** 31
    else:
        raise ValueError("unsupported sample width")
    x = x.reshape(-1, ch).astype(np.float64)
    if ch == 1:
        x = np.repeat(x, 2, 1)
    return x[:, :2], fs


def write_wav24(path, x, fs):
    x = np.clip(x, -1.0, 1.0 - 2 ** -23)
    i = (x * 2 ** 23).astype("<i4")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(fs)
        w.writeframes(i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())


def simulated_eeg(t):
    """Toy trajectory: relaxes over ~4 min, transition detector fires ~7 min."""
    slowing = 2.2 / (1 + np.exp(-(t - 200) / 60)) + 0.25 * np.sin(t / 7.0)
    return dict(slowing=float(slowing), transition=bool(t > 420), quality_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--profile", default="default", choices=list(PROFILES))
    ap.add_argument("--carrier", type=float, default=144.0)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--condition", action="store_true", help="pre-flatten source rhythm/transients")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--script", help='state timeline, e.g. "0:SETTLE,150:DRIFT,420:ONSET,600:HOLD"')
    g.add_argument("--simulate", action="store_true", help="drive the controller with a toy EEG")
    ap.add_argument("--log", default=None, help="CSV log path (default: <output>.csv)")
    a = ap.parse_args()

    src, fs = read_wav(a.input)
    if a.condition:
        src = condition_source(src, fs)
    n = int(a.minutes * 60 * fs)
    reps = int(np.ceil(n / len(src)))
    bed = np.concatenate([src] * reps, 0)[:n]        # (use a seamless loop in the app)

    eng = SwellEngine(fs, a.carrier, a.profile)
    ctl = NapInductionController()
    timeline = []
    if a.script:
        timeline = sorted((float(k), v.strip().upper())
                          for k, v in (p.split(":") for p in a.script.split(",")))
    out = np.zeros_like(bed)
    epoch_s, next_epoch = 2.0, 0.0
    for start in range(0, n, a.block):
        t = start / fs
        if a.script:
            for ts, st in timeline:
                if t >= ts:
                    eng.set_state(st)
        elif a.simulate and t >= next_epoch:
            eng.set_state(ctl.update(t, **simulated_eeg(t)))
            next_epoch += epoch_s
        out[start:start + a.block] = eng.process(bed[start:start + a.block])

    peak = np.max(np.abs(out))
    if peak > 0.89:
        out *= 0.89 / peak
    write_wav24(a.output, out, fs)
    log_path = a.log or a.output.rsplit(".", 1)[0] + ".csv"
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "state", "rate_hz", "wave_phase", "envelope", "gain_db", "pan"])
        w.writerows(eng.log)
    states = []
    for row in eng.log:
        if not states or states[-1][1] != row[1]:
            states.append((round(row[0], 1), row[1]))
    print(f"wrote {a.output} ({a.minutes} min), log {log_path}; state changes: {states}")


if __name__ == "__main__":
    main()

"""Audio I/O shared by every ss_gen stage: 48 kHz, stereo, float64 arrays of shape (n, 2)."""

import os
import shutil
import subprocess
import tempfile
import wave
from math import gcd

import numpy as np
from scipy.signal import lfilter, resample_poly

FS = 48000
TARGET_RMS_DBFS = -26.0
PEAK_CEIL_DBFS = -3.0


def resample(x, sr):
    if sr == FS:
        return x
    g = gcd(FS, sr)
    return resample_poly(x, FS // g, sr // g, axis=0)


def pseudo_stereo(mono, seed=0):
    """Decorrelate a mono take into a natural stereo pair: two different short
    diffuse IRs (~25 ms) blended with the dry signal. No comb/phasey Haas tricks."""
    rng = np.random.default_rng(seed)
    m = int(0.025 * FS)
    t = np.arange(m) / FS
    out = []
    for _ in range(2):
        ir = rng.standard_normal(m) * np.exp(-t / 0.006)
        ir /= np.sqrt(np.sum(ir ** 2))
        out.append(0.7 * mono + 0.714 * lfilter(ir, [1.0], mono))
    return np.stack(out, 1)


def to_stereo(x, seed=0):
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        return pseudo_stereo(x[:, 0], seed)
    return x[:, :2]


def pcm_channels(n_values, sr, duration=None, values=None):
    """Raw PCM has no header, so infer mono vs interleaved stereo.

    Primary: the value count against the requested duration (stereo = 2x). Cross-check:
    stereo read as mono strings L,R,L,R into one channel, which mirrors the L-R difference
    up to just below Nyquist; real audio has almost nothing there."""
    if duration:
        ratio = n_values / (duration * sr)
        for ch in (1, 2):
            if abs(ratio - ch) < 0.05 * ch:
                return ch
    if values is not None and len(values) >= 4096:
        f = np.fft.rfftfreq(len(values), 1 / sr)
        P = np.abs(np.fft.rfft(values - values.mean())) ** 2
        return 2 if P[f > 0.96 * sr / 2].sum() / (P.sum() + 1e-30) > 0.02 else 1
    raise ValueError(f"cannot tell mono from stereo ({n_values} values at {sr} Hz)")


def decode_audio(blob, output_format, seed=0, duration=None):
    """Decode an API response: WAV (header), raw PCM s16le (mono or interleaved stereo),
    or anything ffmpeg reads. duration = the requested length, used to detect PCM channels."""
    if blob[:4] == b"RIFF":
        import io
        with wave.open(io.BytesIO(blob), "rb") as w:
            ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
        if sw != 2:
            raise ValueError(f"unsupported WAV sample width {sw}")
        x = (np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0).reshape(-1, ch)
        return to_stereo(resample(x, sr), seed)
    if output_format.startswith("pcm_"):
        sr = int(output_format.split("_")[1])
        v = np.frombuffer(blob, "<i2").astype(np.float64) / 32768.0
        ch = pcm_channels(len(v), sr, duration, v)
        x = v[: len(v) // ch * ch].reshape(-1, ch)
        return to_stereo(resample(x, sr), seed)
    if not shutil.which("ffmpeg"):
        try:                                           # libsndfile >= 1.1 decodes MP3 too
            import io
            import soundfile as sf
            x, sr = sf.read(io.BytesIO(blob), dtype="float64", always_2d=True)
            return to_stereo(resample(x, sr), seed)
        except ImportError:
            raise RuntimeError("compressed audio needs ffmpeg or the soundfile package") from None
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(blob)
        src = f.name
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-f", "f32le", "-ac", "2",
                              "-ar", str(FS), "-"], check=True, capture_output=True).stdout
    finally:
        os.unlink(src)
    return np.frombuffer(raw, "<f4").astype(np.float64).reshape(-1, 2)


def read_wav(path, seed=0):
    with wave.open(path, "rb") as w:
        ch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
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
        raise ValueError(f"{path}: unsupported sample width {sw}")
    x = x.reshape(-1, ch).astype(np.float64)
    return to_stereo(resample(x, sr), seed)


def load_any(path, seed=0):
    if path.lower().endswith(".wav"):
        return read_wav(path, seed)
    if path.lower().endswith((".m4a", ".aac", ".mp4", ".caf")) and not shutil.which("ffmpeg") \
            and shutil.which("afconvert"):                 # macOS: AAC without ffmpeg
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "x.wav")
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI24", path, out], check=True, capture_output=True)
            return read_wav(out, seed)
    with open(path, "rb") as fh:
        return decode_audio(fh.read(), "mp3", seed)


def write_wav24(path, x):
    x = np.clip(np.asarray(x, np.float64), -1.0, 1.0 - 2 ** -23)
    i = (x * 2 ** 23).astype("<i4")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(3)
        w.setframerate(FS)
        w.writeframes(i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def normalise(x, rms_dbfs=TARGET_RMS_DBFS, peak_dbfs=PEAK_CEIL_DBFS):
    """Remove DC, set RMS, then pull down if the peak exceeds the ceiling."""
    x = x - x.mean(0)
    x = x * 10 ** (rms_dbfs / 20) / (rms(x) + 1e-12)
    pk, ceil = np.max(np.abs(x)), 10 ** (peak_dbfs / 20)
    return x * (ceil / pk) if pk > ceil else x

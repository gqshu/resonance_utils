"""Stable Audio 3, run locally through the MLX build in a stable-audio-3 checkout.

The model runs in that checkout's own venv (its `./sa3` wrapper), so this package needs no MLX
or torch: each take is one subprocess. Output is 16-bit stereo 44.1 kHz; read_wav resamples it
to 48 kHz.

SA3_DIR points at <stable-audio-3>/optimized/mlx (default: ~/work/stable-audio-3/optimized/mlx).
"""

import os
import shutil
import subprocess

DEFAULT_DIR = os.path.expanduser("~/work/stable-audio-3/optimized/mlx")
DECODER = {"medium": "same-l", "sm-music": "same-s", "sm-sfx": "same-s"}


def sa3_dir():
    d = os.environ.get("SA3_DIR", DEFAULT_DIR)
    if not os.path.exists(os.path.join(d, "sa3")):
        raise SystemExit(f"no ./sa3 in {d}: set SA3_DIR to <stable-audio-3>/optimized/mlx")
    return d


def model_tag(dit, steps, cfg):
    """What a take's sidecar records as its model: every setting that changes the audio
    besides the prompts and seed, so changing one never silently reuses old takes."""
    return f"sa3-{dit}/steps{steps}/cfg{cfg:g}"


def command(prompt, out_wav, seconds, seed, dit="medium", steps=8, cfg=1.0, negative=None,
            init_wav=None, init_noise=None):
    cmd = ["./sa3", "--prompt", prompt, "--dit", dit, "--decoder", DECODER[dit],
           "--seconds", f"{seconds:g}", "--seed", str(int(seed)), "--steps", str(int(steps)),
           "--cfg", f"{cfg:g}", "--out", os.path.abspath(out_wav)]
    if negative and cfg != 1.0:
        cmd += ["--negative-prompt", negative]
    if init_wav:
        cmd += ["--init-audio", os.path.abspath(init_wav), "--init-noise-level", f"{init_noise:g}"]
    return cmd


def write_init(x, path, fs):
    """Write x (n, 2 float at fs) as the plain 16-bit 44.1 kHz stereo WAV ./sa3 --init-audio reads
    (its reader rejects afconvert's headers and anything but 16-bit 44.1 kHz)."""
    import wave
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly
    g = gcd(44100, fs)
    y = resample_poly(x, 44100 // g, fs // g, axis=0) if fs != 44100 else x
    y = np.clip(y / max(1e-9, float(np.abs(y).max())) * 0.9, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def generate(prompt, out_wav, seconds, seed, dit="medium", steps=8, cfg=1.0, negative=None, log=print,
             init_wav=None, init_noise=None):
    """Render one clip to out_wav (44.1 kHz, 16-bit). Raises RuntimeError with the tail of the
    output on failure."""
    d = sa3_dir()
    if shutil.which("uv") is None and not os.path.isdir(os.path.join(d, ".venv")):
        raise SystemExit(f"{d} is not installed: run ./install.sh there")
    p = subprocess.run(command(prompt, out_wav, seconds, seed, dit, steps, cfg, negative, init_wav, init_noise), cwd=d,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if p.returncode or not os.path.exists(out_wav):
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"sa3 failed (exit {p.returncode}):\n{tail}")
    done = next((ln.strip() for ln in p.stdout.splitlines() if ln.strip().startswith("done")), "")
    if done:
        log(f"    {' '.join(done.split())}")
    return out_wav

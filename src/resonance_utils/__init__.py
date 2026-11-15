"""resonance_utils: soundscape generation for an EEG-adaptive power-nap app.

Entry point: ss_gen (bed -> swell -> optional tone). Modules:
    bed        ElevenLabs takes -> screen -> seamless loop -> condition
    swell      offline, loop-exact swell at 0.1 / 0.2 / 0.25 Hz
    tone       sedative ambient tone layer that breathes with the swell
    scenarios  prompts + per-texture swell profiles
    audio_io   WAV I/O and normalisation
    legacy     the earlier tools (real-time EEG SwellEngine, ocean_gen, bed-builder)
"""

__version__ = "0.2.0"

from .bed import build_bed, generate_takes, screen  # noqa: F401
from .swell import apply_swell  # noqa: F401
from .tone import mix_tone, render_tone_stems  # noqa: F401

__all__ = ["build_bed", "generate_takes", "screen", "apply_swell", "render_tone_stems", "mix_tone",
           "__version__"]

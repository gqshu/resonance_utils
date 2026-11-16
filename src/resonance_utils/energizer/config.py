"""Energizer parameters and config (plan §4). YAML configs live in energizer/config/."""

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
GENERATOR_VERSION = "0.1.0"


@dataclass
class EnergizerParams:
    # identity
    seed: int = 0
    arc_profile: str = "wake_to_work_4m"     # wake_to_work_2m30 | _4m | _5m
    plateau_bpm: float = 118
    entry_tempo_ratio: float = 0.85
    key_root: str = "D"
    mode: str = "major"                      # major | lydian | mixolydian | dorian
    # AI musical layer
    music_source: str = "none"               # elevenlabs | none (all-procedural; the M1 default)
    music_style: str = "ambient_electronic_keys"
    motif: bool = True
    # procedural layers
    pad_timbre: str = "warm_additive"        # warm_additive | glass | string_ensemble
    pad_tuning: str = "just_per_chord"       # just_per_chord | equal
    pulse_style: str = "soft_four"           # none | soft_four | broken_soft | shaker_only
    # texture
    texture_source: str = "procedural_air"   # elevenlabs_sfx | cc0_file | procedural_air | none
    texture_prompt: str = "soft steady rain on leaves, no thunder, no birds"
    length_s: float = 240


def load_yaml(name):
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML (pip install -e '.[energizer]')") from e
    with open(os.path.join(CONFIG_DIR, name)) as fh:
        return yaml.safe_load(fh)


def lever_slots():
    """The engine's slots, plus the derived monaural-beat tones and keep-clear ranges."""
    s = load_yaml("lever_slots.yaml")
    mb = s["mb_slot"]
    c = float(mb["carrier_hz"])
    s["mb_tones_hz"] = [c] + [c + b for b in mb["beats_hz"]]
    s["mb_keep_clear_hz"] = merge_ranges([[f - mb["guard_hz"], f + mb["guard_hz"]] for f in s["mb_tones_hz"]])
    return s


def merge_ranges(rs):
    out = []
    for lo, hi in sorted(rs):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def partial_clash(f, slots):
    """Why a partial at f Hz may not be used in any energizer stem, or None.

    Two rules: it sits in a monaural-beat keep-clear range, or it beats against one of the
    lever tones at a spacing inside native_mod_bands_hz (e.g. 180 or 300 Hz: 40 Hz from 220 / 260)."""
    for lo, hi in slots["mb_keep_clear_hz"]:
        if lo <= f <= hi:
            return f"in the monaural slot {lo:g}-{hi:g} Hz"
    for t in slots["mb_tones_hz"]:
        d = abs(f - t)
        for lo, hi in slots["native_mod_bands_hz"]:
            if lo <= d <= hi:
                return f"beats at {d:g} Hz with the {t:g} Hz lever tone"
    return None


def bed_id(params, slots, model_ids=None):
    """sha1(params, generator version, lever slots, API model ids)[:12] (plan §4)."""
    blob = json.dumps(dict(p=dataclasses.asdict(params), v=GENERATOR_VERSION, s=slots, m=model_ids or {}),
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]

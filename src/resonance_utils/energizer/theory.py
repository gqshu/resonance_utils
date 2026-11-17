"""Modes, diatonic triads, seeded progressions and the bar-quantised chord timeline.

In the all-procedural path (music_source = none) the harmony comes from here. In the hybrid
path (M2) the same timeline format is produced by chord detection on the AI track instead.
"""

import numpy as np

NOTE_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
           "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
PC_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MODES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
}
# Scale-degree progressions (0-based), each a cycle of 4 chords. Bright, unhurried, no dim chords.
PROGRESSIONS = {
    "major":      [[0, 4, 5, 3], [0, 3, 5, 4], [5, 3, 0, 4], [0, 5, 3, 4]],
    "lydian":     [[0, 1, 0, 1], [0, 1, 5, 4], [0, 1, 4, 0]],
    "mixolydian": [[0, 6, 3, 0], [0, 6, 3, 3], [0, 3, 6, 0]],
    "dorian":     [[0, 3, 0, 6], [0, 6, 3, 0], [0, 3, 6, 3]],
}
JUST = {0: 1.0, 2: 9 / 8, 3: 6 / 5, 4: 5 / 4, 7: 3 / 2, 12: 2.0, 14: 9 / 4, 19: 3.0}   # semitones over the chord root


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def triad(mode, degree):
    sc = MODES[mode]
    pcs = [sc[(degree + k) % 7] + 12 * ((degree + k) // 7) for k in (0, 2, 4)]
    third, fifth = pcs[1] - pcs[0], pcs[2] - pcs[0]
    return pcs[0], third, fifth


def chord_timeline(tempo, key_root, mode, seed, bars_per_chord=2, beats_per_bar=4):
    """Chords on bar boundaries of the tempo map. Returns a list of dicts (chords.json rows)."""
    rng = np.random.default_rng([seed, 11])
    prog = PROGRESSIONS[mode][int(rng.integers(len(PROGRESSIONS[mode])))]
    key_pc = NOTE_PC[key_root]
    span = bars_per_chord * beats_per_bar
    out, b, i = [], 0.0, 0
    while b < tempo.total_beats - 1e-9:
        deg = prog[i % len(prog)]
        root, third, fifth = triad(mode, deg)
        if fifth != 7:                                 # never a diminished triad
            deg, (root, third, fifth) = 0, triad(mode, 0)
        pc = (key_pc + root) % 12
        quality = "major" if third == 4 else "minor"
        e = min(b + span, tempo.total_beats)
        out.append(dict(
            bar=int(b // beats_per_bar) + 1, start_beat=b, end_beat=e,
            start_s=round(float(tempo.time_of_beat(b)), 6), end_s=round(float(tempo.time_of_beat(e)), 6),
            label=PC_NAME[pc] + ("" if quality == "major" else "m"), degree=deg + 1,
            root_pc=pc, third=third, quality=quality,
            pcs=sorted({pc, (pc + third) % 12, (pc + 7) % 12}), confidence=1.0))
        b, i = e, i + 1
    return out

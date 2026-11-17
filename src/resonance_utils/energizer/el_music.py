"""Eleven Music: composition plans, compose (cached), and the reference mix (plan §5.2, M2 (c)).

API as documented on 2026-11-15 (verify before changing): POST /v1/music with a
`composition_plan` of `chunks` for music_v2_5 (music_v1, whose plan has `sections`, is deprecated
and removed 2027-01-31). Instrumental chunks carry only a section tag in `text` (that is what
the free /v1/music/plan endpoint returns), and the content goes in positive/negative styles.
Each chunk is 3-120 s. Responses are cached by request hash in cache/el/, so a re-run is free.
"""

import json
import os

import numpy as np

from ..core import cache
from ..core import elevenlabs as el
from ..core.audio_io import FS, decode_audio, write_wav24
from .config import load_yaml

MODEL_ID = "music_v2_5"
CREDITS_PER_MIN = 900                          # pricing page, 2026-11
PREROLL_S = 8.0
FORMATS = ("pcm_48000", "mp3_48000_192")       # PCM needs a Pro plan; MP3 is the fallback
NEGATIVE = ["vocals", "voice", "choir", "spoken word", "lyrics", "drums", "percussion", "drops",
            "risers", "build-ups", "sidechain", "pumping", "tremolo", "stutter", "glitch",
            "sudden changes", "minor key"]


def _key(params):
    return f"{params.key_root} {params.mode}"


def main_plan(params, arc, tempo=None):
    """Call A: the main track at plateau tempo, covering pre-roll + ramp + plateau + hand-off.
    Variant "a" is the original 4-section plan (cached); the others come from VARIANTS."""
    return build_main(params, arc, tempo)[0]


def main_plan_phases(params, arc, tempo=None):
    """The bed phase (sparse | fuller | plateau | handoff) of each chunk of main_plan."""
    return build_main(params, arc, tempo)[1]


def build_main(params, arc, tempo=None):
    variant = getattr(params, "music_variant", "a")
    if variant == "a":
        return _plan_a(params, arc)
    if tempo is None:
        from .tempo import from_arc
        tempo = from_arc(arc, params.plateau_bpm, params.entry_tempo_ratio)
    return _plan_variant(params, arc, tempo, VARIANTS[variant])


def _plan_a(params, arc):
    style = load_yaml("music_styles.yaml")[params.music_style]["style"]
    bpm, key = f"{params.plateau_bpm:g} BPM", _key(params)
    (r0, r1), (p0, p1), (h0, h1) = arc.phases["ramp"], arc.phases["plateau"], arc.phases["handoff"]
    mid = (r0 + r1) / 2
    base = ["instrumental", style, bpm, key]
    neg = [n for n in NEGATIVE if not (params.mode == "dorian" and n == "minor key")]
    secs = [
        ("[Build Sparse]", PREROLL_S + mid - r0, base + ["sparse soft keys", "gentle", "warm", "steady"]),
        ("[Build Fuller]", r1 - mid, base + ["light plucked arpeggio", "brighter", "more motion", "steady energy"]),
        ("[Plateau]", p1 - p0, base + ["steady", "hypnotic", "consistent", "no lead melody",
                                       "background music for focus"]),
        ("[Handoff]", h1 - h0, base + ["gently thinning", "softer"]),
    ]
    plan = dict(chunks=[dict(text=t, duration_ms=int(round(d * 1000)), positive_styles=s, negative_styles=neg,
                             context_adherence="high") for t, d, s in secs])
    return plan, ["sparse", "fuller", "plateau", "handoff"]


# --- variants: more, shorter sections that rotate the arrangement, an explicit chord progression,
# and gentle lifts. No salient lead melody (plan: none during work), but motion, counter-lines and
# section contrast so the plateau doesn't loop. Risers/build-ups/sudden changes are no longer
# negative (an energizer needs lifts); what matters for the engine's AM stays out.
NEGATIVE_VARIANT = ["vocals", "voice", "choir", "spoken word", "lyrics", "drums", "percussion", "drops",
                    "sidechain", "pumping", "tremolo", "stutter", "glitch", "abrupt stops", "minor key"]
VARIANTS = {
    "b": dict(
        name="rotation", progression={"major": [0, 4, 5, 3]},
        chunks=[
            ("sparse", "[Intro]", 1, ["sparse soft electric piano chords", "warm and gentle", "slowly awakening"]),
            ("sparse", "[Arpeggio Enters]", 1, ["a soft plucked arpeggio joins the keys", "light and bright",
                                                "gentle forward motion"]),
            ("fuller", "[Mallets]", 1, ["warm marimba counter-pattern answering the keys", "more motion",
                                        "brighter"]),
            ("fuller", "[Lift]", 1, ["fuller layers", "bright synth pad swells in", "gentle lift into the next section"]),
            ("plateau", "[Groove A]", 1, ["full arrangement", "keys and arpeggio interlocking", "uplifting",
                                          "steady energy"]),
            ("plateau", "[Variation]", 1, ["the arpeggio changes pattern", "mallets take the foreground",
                                           "fresh chord voicings", "uplifting"]),
            ("plateau", "[Groove B]", 1, ["bright full arrangement", "soft background counter-melody",
                                          "never a lead melody", "positive and driving"]),
            ("handoff", "[Outro]", 1, ["gently thinning", "keys and arpeggio only", "softer"]),
        ]),
    "c": dict(
        name="evolving", progression={"major": [5, 3, 0, 4]},
        chunks=[
            ("sparse", "[Intro]", 1, ["gentle piano and soft synth", "evolving arpeggio pattern", "calm but bright"]),
            ("sparse", "[Call]", 1, ["call and response between piano and synth", "growing motion"]),
            ("fuller", "[Build]", 1, ["arpeggio patterns evolve every few bars", "warmer and fuller",
                                      "more motion"]),
            ("fuller", "[Rise]", 1, ["bright plucks join", "gentle lift", "rising energy"]),
            ("plateau", "[Theme]", 1, ["full evolving arrangement", "soft background counter-melody",
                                       "never a lead melody", "optimistic"]),
            ("plateau", "[Shift]", 1, ["harmonic shift", "new instrument colours", "the counter-line moves to plucks",
                                       "optimistic"]),
            ("plateau", "[Theme Return]", 1, ["full arrangement returns brighter", "interlocking patterns",
                                              "steady drive"]),
            ("handoff", "[Outro]", 1, ["gently thinning", "softer", "arpeggio fades into space"]),
        ]),
    "d": dict(
        name="contrast", progression={"major": [0, 3, 5, 4]},
        chunks=[
            ("sparse", "[Intro]", 1, ["soft warm keys", "gentle", "spacious"]),
            ("sparse", "[First Lift]", 1, ["light string pad and plucked guitar join", "gentle lift"]),
            ("fuller", "[Build]", 2, ["fuller arrangement", "strings and keys", "rhythmic plucked pattern",
                                      "rising energy"]),
            ("plateau", "[Full]", 2, ["full bright arrangement", "strings, keys and plucks", "uplifting",
                                      "driving forward"]),
            ("plateau", "[Breakdown]", 1, ["brief breakdown", "keys and soft pad only", "airy"]),
            ("plateau", "[Return]", 2, ["gentle lift back to full", "brighter than before", "new counter-pattern",
                                        "uplifting"]),
            ("handoff", "[Outro]", 1, ["gently thinning", "softer", "warm"]),
        ]),
}


def _progression_names(params, degrees):
    from .theory import NOTE_PC, PC_NAME, triad
    out = []
    for d in degrees:
        r, third, _ = triad(params.mode, d)
        pc = (NOTE_PC[params.key_root] + r) % 12
        out.append(PC_NAME[pc] + ("" if third == 4 else "m"))
    return out


def _plan_variant(params, arc, tempo, var):
    """Chunks sized in whole bars at plateau tempo so that, stretched onto the ramp, each phase
    holds exactly as many AI bars as bed bars (no bars dropped)."""
    from .theory import PROGRESSIONS
    style = load_yaml("music_styles.yaml")[params.music_style]["style"]
    bar_s = 4 * 60.0 / params.plateau_bpm
    (r0, r1), (p0, p1), (h0, h1) = arc.phases["ramp"], arc.phases["plateau"], arc.phases["handoff"]
    mid = (r0 + r1) / 2
    spans = dict(sparse=(r0 - PREROLL_S, mid), fuller=(mid, r1), plateau=(p0, p1), handoff=(h0, h1))
    bars = {k: max(2, int(round((tempo.beat_at(b) - tempo.beat_at(a)) / 4))) for k, (a, b) in spans.items()}
    degrees = var["progression"].get(params.mode, PROGRESSIONS[params.mode][0])
    prog = ", ".join(_progression_names(params, degrees))
    base = ["instrumental", style, f"{params.plateau_bpm:g} BPM", _key(params),
            f"chord progression {prog}", "chords change every two bars"]
    neg = [n for n in NEGATIVE_VARIANT if not (params.mode == "dorian" and n == "minor key")]
    chunks, phases = [], []
    for ph in ("sparse", "fuller", "plateau", "handoff"):
        items = [c for c in var["chunks"] if c[0] == ph]
        w = np.array([c[2] for c in items], float)
        raw = bars[ph] * w / w.sum()
        nb = np.floor(raw).astype(int)
        for i in np.argsort(raw - nb)[::-1][: bars[ph] - nb.sum()]:
            nb[i] += 1
        for (_, tag, _, styles), k in zip(items, nb):
            if k <= 0:
                continue
            chunks.append(dict(text=tag, duration_ms=int(round(k * bar_s * 1000)), positive_styles=base + styles,
                               negative_styles=neg, context_adherence="high"))
            phases.append(ph)
    return dict(chunks=chunks), phases


def motif_plan(params, timbre="felt piano"):
    """Call B: a short solo melodic line at entry tempo, same key and mode."""
    entry = params.plateau_bpm * params.entry_tempo_ratio
    pos = ["instrumental", "single gentle bright melodic line", timbre, "simple rising phrase",
           f"{entry:.0f} BPM", _key(params), "solo", "no accompaniment"]
    return dict(chunks=[dict(text="[Motif]", duration_ms=36000, positive_styles=pos,
                             negative_styles=list(NEGATIVE) + ["chords", "pads", "bass"],
                             context_adherence="high")])


def plan_seconds(plan):
    return sum(c["duration_ms"] for c in plan["chunks"]) / 1000


def compose(plan, seed, out_wav, dry_run=False, log=print):
    """Compose (or reuse the cached response), decode to 48 kHz stereo, write out_wav.
    Returns metadata: request hash, format, song id header, cache hit."""
    secs = plan_seconds(plan)
    body = dict(composition_plan=plan, model_id=MODEL_ID, seed=int(seed))
    if dry_run:
        log(f"    POST /v1/music  {secs:.0f} s, ~{secs / 60 * CREDITS_PER_MIN:,.0f} credits  "
            f"({len(plan['chunks'])} chunks: {', '.join(c['text'] for c in plan['chunks'])})")
        return None
    last = None
    for fmt in FORMATS:
        q = dict(output_format=fmt)
        hdr = {}

        def fetch():
            blob, h = el.post("/v1/music", body, q, seconds=secs, with_headers=True, timeout=600)
            hdr.update(h)
            return blob
        try:
            blob, path, hit = cache.cached_call("/v1/music", body, q, fetch, suffix="." + fmt.split("_")[0])
        except el.APIError as e:
            if fmt.startswith("pcm") and el.pcm_unavailable(e):
                log(f"    {fmt} not available on this plan ({e}); trying {FORMATS[-1]}")
                last = e
                continue
            raise
        meta_path = os.path.join(os.path.dirname(path), "headers.json")
        if hdr:
            with open(meta_path, "w") as fh:
                json.dump({k: v for k, v in hdr.items() if "key" not in k.lower()}, fh, indent=2)
        headers = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        x = decode_audio(blob, fmt, seed=seed, duration=secs)
        write_wav24(out_wav, x)
        log(f"    {'cached' if hit else 'composed'} {len(x) / FS:.1f} s ({fmt}) -> {out_wav}")
        song = next((v for k, v in headers.items() if "song" in k.lower()), None)
        return dict(request=el.request_hash("/v1/music", body, q), format=fmt, cache_hit=hit,
                    seconds=round(len(x) / FS, 3), song_id=song, model_id=MODEL_ID, seed=int(seed), plan=plan)
    raise last


STEM_VARIATION = "six_stems_v1"
DISCARD_STEMS = ("vocal", "voice", "drum", "percussion", "bass")   # bass: analysis reference only


def separate(wav_path, out_dir, variation=STEM_VARIATION, log=print):
    """POST /v1/music/stem-separation (multipart upload of the track). Cached by the audio's
    sha256 + variation. Writes every stem as <out_dir>/<name>.wav; returns {name: path} and meta."""
    import hashlib
    import io
    import zipfile
    with open(wav_path, "rb") as fh:
        audio = fh.read()
    digest = hashlib.sha256(audio).hexdigest()
    fields = {"stem_variation_id": variation}
    q = dict(output_format="pcm_48000")
    key_body = dict(audio_sha256=digest, **fields)

    def fetch():
        data, ctype = el.multipart(fields, {"file": (os.path.basename(wav_path), audio, "audio/wav")})
        return el.post("/v1/music/stem-separation", key_body, q, accept="application/zip", data=data,
                       content_type=ctype, timeout=900)
    blob, path, hit = cache.cached_call("/v1/music/stem-separation", key_body, q, fetch, suffix=".zip")
    os.makedirs(out_dir, exist_ok=True)
    stems = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        for name in names:
            if name.endswith("/"):
                continue
            raw = z.read(name)
            stem = os.path.splitext(os.path.basename(name))[0].lower().replace(" ", "_")
            fmt = "pcm_48000" if raw[:4] != b"RIFF" and not name.lower().endswith(".mp3") else "mp3"
            x = decode_audio(raw, fmt, duration=None if raw[:4] == b"RIFF" else
                             len(raw) / 2 / 2 / 48000)          # stereo s16 at 48 kHz, if raw PCM
            p = os.path.join(out_dir, stem + ".wav")
            write_wav24(p, x)
            stems[stem] = p
    log(f"    {'cached' if hit else 'separated'} {variation}: {', '.join(stems)}  (zip: {names})")
    return stems, dict(variation=variation, audio_sha256=digest, zip_names=names, cache_hit=hit,
                       request=el.request_hash("/v1/music/stem-separation", key_body, q))


def music_stem_names(stems):
    """The stems summed into `music`: everything except vocals, drums and bass."""
    keep = [k for k in stems if not any(d in k for d in DISCARD_STEMS)]
    return keep

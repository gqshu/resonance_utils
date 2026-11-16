"""resonance_utils: soundscape generation for the resonance EEG-adaptive app.

One command, `ss`, drives every soundscape type:

    ss list [type]                 types, or what a type can generate
    ss gen <type> ...              build into build/<type>/<id>/
    ss deploy [<type>|all] ...     collect ready builds into deploy/<type>/ + deploy/index.json
    ss migrate                     move a pre-0.3 build/ and deploy/ into the per-type layout

Types (core/types.py registry):
    inducer    power-nap inducer: texture loops with a baked slow swell (ss_gen / ss_deploy aliases)
    energizer  wake-to-work energizer beds (in development; see docs/)

Packages:
    core       audio I/O, analysis, loops, carrier slots, ElevenLabs client, cache, deploy helpers
    inducer    the inducer pipeline (bed, swell, tone, scenarios)
    energizer  the energizer pipeline
    legacy     the earlier tools (real-time EEG SwellEngine, ocean_gen, bed-builder)
"""

__version__ = "0.3.0"


def __getattr__(name):                     # lazy re-exports of the pre-0.3 top-level API
    if name in ("build_bed", "generate_takes", "screen"):
        from .inducer import bed
        return getattr(bed, name)
    if name == "apply_swell":
        from .inducer.swell import apply_swell
        return apply_swell
    if name in ("mix_tone", "render_tone_stems"):
        from .inducer import tone
        return getattr(tone, name)
    raise AttributeError(name)


__all__ = ["build_bed", "generate_takes", "screen", "apply_swell", "render_tone_stems", "mix_tone",
           "__version__"]

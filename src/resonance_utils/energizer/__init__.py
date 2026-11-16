"""Energizer soundscapes (wake-to-work ramp): hybrid ElevenLabs + procedural beds with lever-ready
stems for the app's music engine. Spec: docs/ENERGIZER_SOUNDSCAPE_GENERATOR_PLAN.md.

Status: skeleton. Params, config and the type hooks exist; the pipeline lands milestone by
milestone (M1 procedural core first). Shared pieces come from resonance_utils.core:
audio_io, analysis, loop, slots (carriers), elevenlabs (client), cache (request cache), deploy.
"""

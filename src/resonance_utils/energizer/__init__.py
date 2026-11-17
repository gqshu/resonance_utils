"""Energizer soundscapes (wake-to-work ramp): hybrid ElevenLabs + procedural beds with lever-ready
stems for the app's music engine. Spec: docs/ENERGIZER_SOUNDSCAPE_GENERATOR_PLAN.md.

Status: M1 (procedural core): `ss gen energizer` renders all-procedural beds (music_source=none)
with tempo map, arc, chords, pad/bass/pulse/air stems, preview mix and provisional QA. Next: M2
(ElevenLabs music + texture, analysis, conforming), M3 (QA gate + failure suite), deploy. Shared pieces come from resonance_utils.core:
audio_io, analysis, loop, slots (carriers), elevenlabs (client), cache (request cache), deploy.
"""

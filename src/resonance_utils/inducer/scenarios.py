"""Scenario presets: prompt variants plus how hard the swell may process that texture.

prompt              the variant ss_gen builds beds from ("a")
variants            extra prompts to compare with `ss_gen --probe N` before promoting one
depth / tilt / pan  scale the per-rate swell defaults (swell.RATE_DEFAULTS)
xover               brightness crossover in Hz: treble above it dips in each trough

{flat}   = bed.FLAT_LINE, the original negatively phrased constant-level line
{steady} = STEADY_LINE, the same request phrased positively

What screening has taught us about prompts (see README, "Prompting for pass rate"):
  * close, discrete sources (drops on a tent roof) come back as transients and 1-3 Hz patter;
    distance + an enclosure ("through a closed window") fuses them into a smooth wash
  * naturally periodic sources (surf) come back with their own ~0.15 Hz swell, in level and
    in brightness; conditioning removes the level, only partly the brightness
  * name only what you want to hear: text-to-audio models tend to add things that are
    mentioned even when negated ("no seagulls"); ElevenLabs v3 guidance is to describe the
    setting and atmosphere instead
"""

STEADY_LINE = "Steady and unchanging from start to finish: one even, continuous texture."

SCENARIOS = {
    "train_smooth": dict(
        prompt=("Interior of a modern high-speed electric train cruising at steady speed at night, "
                "recorded from a passenger seat. Deep continuous low rumble, soft broadband air hiss, "
                "faint smooth wheel-on-rail hum, gentle cabin resonance. {flat} No rhythmic clicking. "
                "Quiet empty carriage, no voices, no announcements, no doors, no horn, no brakes."),
        variants=dict(
            b=("Inside an empty modern high-speed train at night, heard from a padded seat. A deep, "
               "smooth rumble under a soft broadband air hiss, the faint even hum of wheels on welded "
               "rails and a warm cabin resonance. {steady}"),
            c=("The low, warm drone of a high-speed train interior at cruising speed, muffled by thick "
               "double glazing and upholstered seats; enveloping and seamless. {steady}")),
        depth=0.8, tilt=0.8, pan=0.8, xover=1200.0),
    "airplane": dict(
        prompt=("Airplane cabin interior during smooth cruise at night, steady deep engine drone and "
                "broadband air noise, muffled and even. {flat} No voices, no announcements, no chimes, "
                "no seatbelt sign dings."),
        variants=dict(
            b=("A long-haul airplane cabin at night in smooth cruise, heard from a seat over the wing. "
               "A deep, even engine drone blended into soft broadband air noise, muffled and warm, "
               "cabin lights dimmed. {steady}")),
        depth=0.8, tilt=0.8, pan=0.8, xover=1000.0),
    "car_night": dict(
        prompt=("Inside a quiet car driving on a smooth highway at night, recorded from the back seat. "
                "Soft tire roar on asphalt, low engine hum, gentle wind around the windows. {flat} "
                "No music, no voices, no indicators, no horn, no passing trucks."),
        variants=dict(
            b=("The back seat of a well-insulated car cruising at constant speed on fresh, smooth "
               "highway asphalt at night. A soft continuous tire roar, a low steady engine hum and a "
               "gentle hush of air around the closed windows. {steady}")),
        depth=0.8, tilt=0.8, pan=0.8, xover=1200.0),
    "rain_soft": dict(
        # 2/6 pass; the rest miss only on 1-12 Hz flicker (0.32-0.57 dB vs 0.3). The old close
        # "tent roof" prompt failed 6/6 on transients (+50-120/min).
        prompt=("Heavy steady rain heard from inside a quiet house through a closed window, muffled and "
                "distant. One dense, continuous, soft roar in which the drops blend together into an even "
                "wash; no single drops, drips or taps stand out. {flat} No thunder, no wind gusts, "
                "no dripping gutters, no splashes, no birds."),
        variants=dict(
            b=("Heavy steady rain heard from inside a quiet house through a closed window, muffled and "
               "distant: one dense, soft roar in which countless drops blend into an even wash. {steady}"),
            c=("Steady heavy rain on a roof, heard from a cozy room on the floor below, so muffled that "
               "it becomes one smooth, low, enveloping hush. {steady}")),
        depth=0.7, tilt=1.0, pan=0.8, xover=2500.0),
    "river": dict(
        prompt=("A wide, calm river flowing steadily past, heard from the bank. Continuous smooth "
                "water wash and soft gurgling texture, even and dense. {flat} No splashes, no birds, "
                "no insects, no wind, no voices."),
        variants=dict(
            b=("A wide, deep river flowing steadily, heard from a grassy bank a short way back: a smooth, "
               "dense water wash with a soft, even gurgle woven through it. {steady}"),
            c=("The constant low rush of a large river heard from a cabin porch nearby, smooth and "
               "broadband, like soft natural white noise. {steady}")),
        depth=0.8, tilt=0.8, pan=1.0, xover=1500.0),
    "ocean_distant": dict(
        # 1/3 pass: real surf rhythm (~0.15 Hz) leaves a brightness swell after conditioning.
        # The variants push the listener further away so many waves average into one roar.
        prompt=("Distant ocean surf heard from inside a beach house with the window open, a continuous "
                "soft wash of far-away waves blending into one steady roar. {flat} No individual crashing "
                "waves, no seagulls, no wind gusts, no voices."),
        variants=dict(
            b=("The steady low roar of the sea heard from a hilltop far inland, so distant that countless "
               "waves have merged into one smooth, continuous hush. {steady}"),
            c=("The deep, even rumble of a vast ocean heard through the closed windows of a house on a "
               "cliff, a wide soft wall of sound with no single wave standing out. {steady}")),
        depth=1.0, tilt=1.0, pan=1.0, xover=1200.0),
    "wind_pines": dict(
        prompt=("Gentle steady wind through a pine forest at night, soft airy rustle of needles, "
                "even and continuous. {flat} No gusts, no creaking branches, no birds, no insects, "
                "no animals."),
        variants=dict(
            b=("A soft, constant breeze high in tall pine trees at night, heard from a sheltered "
               "clearing: a smooth airy hush of countless needles. {steady}")),
        depth=0.8, tilt=0.6, pan=1.0, xover=900.0),
}

# For --from-dir beds with no scenario of their own.
DEFAULT_PROFILE = dict(depth=0.8, tilt=0.8, pan=0.8, xover=1500.0)


def prompt_for(name, variant="a"):
    from .bed import FLAT_LINE
    sc = SCENARIOS[name]
    text = sc["prompt"] if variant == "a" else sc["variants"][variant]
    return text.format(flat=FLAT_LINE, steady=STEADY_LINE)


def variant_names(name):
    return ["a"] + list(SCENARIOS[name].get("variants", {}))


def profile_for(name):
    return {k: v for k, v in SCENARIOS[name].items() if k not in ("prompt", "variants")}


# --- Stable Audio 3 (local; `ss gen inducer --engine sa3`) -----------------------------------
# One approach, variant "r": every take is SA3 audio-to-audio from a REAL recording of the scene,
# steered by a "signature" prompt that names the 3-4 sounds that make the scene recognisable.
#
# Why (2026-11-19, see core/identity.py): text-only prompts, ElevenLabs or SA3, that asked for
# "distant, muffled, smooth, continuous" (what the screens reward) turned every scene into one of
# two noise colours, vehicles or water, with the same texture; SA3 "rain through a closed window"
# came back centred at 157 Hz. Signature prompts made the scenes distinct but still missed some
# (car sat closer to the other vehicles than to real car recordings). Starting from a real
# recording that itself passes the screens gave the takes closest to real recordings (median
# distance 3.1 vs 6.7 for the shipped ElevenLabs set) and the widest margin over the other scenes.
#
# Each take uses the scene's references in turn, and a different window of each per seed.
# SA3_INIT_NOISE: 0.5 stays close to the reference, 0.7 keeps its timbre but recomposes it.
# The prompt is SA3-style (Freesound/AudioSparx metadata: `TrackType: SFX` first), positive only.
#
# refs/ is local only: scripts/freesound_refs.py fetches the CC0 Freesound part (credits.json
# records every licence); the Pixabay files were added by hand: fine as references, but they may
# not be redistributed on their own. Only references that pass the screens are listed here.
SA3_INIT_NOISE = 0.6
_FS = "refs/inducer/freesound"

SA3_SCENES = {
    "train_smooth": dict(
        prompt=("Field recording inside a passenger train carriage at cruising speed. The deep rumble of the "
                "bogies under the floor, the metallic singing hiss of steel wheels on the rails, the low hum "
                "of the traction motor, and the faint rattle and creak of the carriage body."),
        # Strict screens kept for train (2026-11-20), so no clickety-clack: every reference segment
        # with a measurable rhythm failed (the steam trains up to +13 dB at 1.1-1.6 Hz, +19-61
        # transients/min), and asking SA3 for "wheels over rail joints" added 8-65 clacks/min to
        # 25 of 32 takes (2 passed). Identity comes from the wheel-on-rail hiss, motor and carriage
        # instead, from cabin recordings whose segments pass: two smooth ones plus three with
        # sparse, irregular clatter. 36736 "trainride" gave 0/14 (a 90.8 Hz line beating the 110 Hz
        # carrier).
        refs=[f"{_FS}/train_smooth/{f}" for f in (
            "372890_inside-train.mp3", "619188_ambiente-tren-oil-train-ambience.mp3",
            "151867_hungarian-train-ride.mp3", "511306_train-cabin-by-night-wheels-grind.mp3",
            "616533_120618-train-interior-moving-clacks-rumb.mp3")],
        # SA3 adds knocks to anything train-like: 25 of 30 takes failed on transients alone. Capping
        # them at +4 dB before screening (the bed caps at +6 dB anyway) let 17 of those 25 pass the
        # unchanged screens; takes with a real beat still fail on rhythm. Approved 2026-11-20.
        precap_db=4.0),
    "airplane": dict(
        prompt=("Recorded in the cabin of a jet airliner in flight. Loud rushing air noise streaming over the "
                "fuselage, the steady low drone of the turbofan engines with a faint high whine, and the hiss "
                "of the overhead air vents."),
        refs=[f"{_FS}/airplane/{f}" for f in ("341591_plane-interior.mp3", "447825_737-plane-flight-passenger-cabin.mp3",
                                              "438972_airliner-jet-interior-ambience-idle-in-f.mp3",
                                              "156645_intinflight.mp3")]),
    "car_night": dict(
        prompt=("Recorded inside a car driving on a motorway. The coarse roar of tyres on rough asphalt, a low "
                "engine drone, and wind rushing past the windows and mirrors."),
        # Built with --engine sa3 only: as they are, real car recordings fail the screens (1 of 16
        # windows from 9 recordings, 2026-11-20). Engine tones sit at 222-231 Hz and 454 Hz, next to
        # the 220/440 Hz carriers, and city traffic adds transients.
        refs=[f"{_FS}/car_night/{f}" for f in ("397115_car-driving-interior-perspective.mp3",
                                               "383476_car-interior-driving-02.mp3", "164621_car-interior-driving-1.mp3")]),
    "rain_soft": dict(
        # No "splashing" or "trickle of running water": that is a stream, and the first rain bed
        # sounded like the river in listening (2026-11-19).
        prompt=("Steady rain falling on broad leaves and a wooden porch roof, recorded outdoors under the "
                "porch. The dense pitter-patter of countless raindrops tapping on leaves and wood over an "
                "even, airy hiss of rainfall."),
        refs=[f"{_FS}/rain_soft/217236_steady-rain-in-the-city.mp3",
              f"{_FS}/rain_soft/725603_rain-falling-on-the-greenhouse.mp3",
              "refs/inducer/donrain-lluvia-relajante-rain-2-210937.m4a"]),
    "river": dict(
        # A wide river, not a brook: under the strict screens a stream's gurgle and rain's patter both
        # end up as smooth bright hiss and sound alike (listening, 2026-11-19), so the two are kept
        # apart by register: the river low-mid and full, the rain high and airy.
        prompt=("Recorded on the bank of a wide, deep river. A full, low rushing flow of a large body of "
                "moving water, with a soft swirling wash close to the bank."),
        # Only 726340, a wide river strong below 125 Hz. Streams stay bright even at noise 0.75, and
        # mixed with them the wide-river takes (4/6 passing) were dropped as tonal outliers, 10-21 dB
        # off the stream majority. The recording fails the screens itself (swell, rhythm); only the
        # output takes must pass.
        refs=[f"{_FS}/river/726340_water-flowing-on-a-river-nature-ambient-.mp3"],
        # SA3 drops the lows as noise rises (63 Hz on one clip, 2026-11-19: ref -1, 0.35 -1, 0.45 -7,
        # 0.55 -14, 0.65 -29 dB): 0.75 rendered the wide river as a bright stream again, and at 0.45 only 4/20 takes
        # passed (the recording's own clatter, swell and a 90.8 Hz line came through).
        init_noise=0.55,
        # a stream has almost nothing below 250 Hz: the tone layer's 88 Hz drone at the default
        # -12 dB made it measure closer to other scenes than to real streams (margin +7.4 -> -4.5)
        tone_db=-24.0),
    "ocean_distant": dict(
        prompt=("Gentle ocean waves rolling onto a sandy beach, recorded from the shore. The deep, soft wash of "
                "each wave arriving and the long fizzing hiss of foam drawing back over the sand."),
        refs=[f"{_FS}/ocean_distant/463250_beach-waves-04.mp3"]),
    "wind_pines": dict(
        prompt=("Wind blowing through a pine forest at night. The airy whooshing hush of countless pine needles, "
                "slowly rising and falling, with a soft high whistle through the branches."),
        refs=[f"{_FS}/wind_pines/335889_wind-through-trees.mp3",
              f"{_FS}/wind_pines/460178_forest-ambience-steady-breeze-summer-swe.mp3",
              f"{_FS}/wind_pines/403051_wind-forest-08-strong-l-01.mp3"]),
}
SA3_SCENARIOS = {k: {"r": dict(prompt=f"TrackType: SFX. {v['prompt']}", cfg=1.0, init=v["refs"],
                               init_noise=v.get("init_noise", SA3_INIT_NOISE))}
                 for k, v in SA3_SCENES.items()}


def sa3_tone_db(name, default):
    """The scene's tone-layer level (dB re the bed) for SA3 builds, if it overrides --tone-db."""
    return SA3_SCENES.get(name, {}).get("tone_db", default)


def sa3_precap_db(name):
    """The scene's pre-screening transient cap (dB) for SA3 builds, or None (bed.PRECAP_DB)."""
    return SA3_SCENES.get(name, {}).get("precap_db")


def sa3_spec(name, variant="r"):
    return SA3_SCENARIOS[name][variant]


def sa3_variant_names(name):
    return list(SA3_SCENARIOS[name])

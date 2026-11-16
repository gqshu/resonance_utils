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

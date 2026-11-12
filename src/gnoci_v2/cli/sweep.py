import time

import click
import numpy as np

from gnoci_v2.cli.run import countdown
from gnoci_v2.control.gait_cycle import cycle_targets, fit_cycle, schedule
from gnoci_v2.control.main import FALL_STEPS, Main
from gnoci_v2.control.record import Recorder

DEFAULT_CYCLE = "logs/floor/cat-fire-potential/raw-accel-tests/r1-small.npz"
DEFAULT_FREQS = "0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.3,2.6,3.0"
# The cycle's amplitude ramps up over this long at the start, from its mean pose.
RAMP_SECONDS = 1.0


@click.command()
@click.option(
    "--cycle-from",
    "cycle_from",
    multiple=True,
    default=[DEFAULT_CYCLE],
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Recording(s) whose targets, averaged over the gait phase, make the cycle"
    " that's played. Repeat for several.",
)
@click.option("--freqs", default=DEFAULT_FREQS, show_default=True,
              help="Gait frequencies (Hz per leg) to play, in order, comma separated.")
@click.option("--seconds", default=5.0, show_default=True, type=click.FloatRange(min=1),
              help="Seconds at each frequency.")
@click.option("--amp", default=1.0, show_default=True, type=click.FloatRange(min=0),
              help="Scales the cycle's motion about its mean pose (1 = as recorded).")
@click.option("--record", "record", required=True, type=click.Path(dir_okay=False),
              help="Save every step to this .npz, as gnoci run --record does.")
@click.option("--start-delay", default=5.0, show_default=True, type=click.FloatRange(min=0),
              help="Seconds to hold the cycle's mean pose before starting, to set the robot down.")
@click.option("--fall-tilt", default=40.0, show_default=True, type=click.FloatRange(min=0),
              help="Stop once the torso is tilted more than this many degrees for 0.1 s. 0 never stops.")
def sweep(cycle_from, freqs, seconds, amp, record, start_delay, fall_tilt):
    """Plays one gait cycle open loop at a range of frequencies, to measure rocking.

    No policy runs: the joint targets are a recorded policy's average gait
    cycle, replayed at each frequency in --freqs for --seconds, with the phase
    continuous across the changes. Analyse with scripts/gait_sweep.py --real.
    """
    freqs = [float(f) for f in freqs.split(",")]
    coef = fit_cycle(cycle_from)
    plan = schedule(freqs, seconds)
    total = len(freqs) * seconds
    print(f"Cycle from {', '.join(cycle_from)}; {len(freqs)} frequencies x {seconds:g} s = {total:g} s")

    # Main sets up the servos, the IMU thread and the observation; its policy isn't used.
    main = Main(fall_tilt=fall_tilt)
    recorder = Recorder(
        record,
        model="gait-sweep",
        action_scale=main.policy.action_scale,
        control_freq=main.control_freq,
        close_loop=True,
        sweep_freqs=np.array(freqs),
        sweep_seconds=seconds,
        sweep_amp=amp,
        sweep_ramp_seconds=RAMP_SECONDS,
        cycle_from=np.array(cycle_from),
        cycle_coef=coef,
    )
    dt = main.dt
    try:
        main.init_main_process()
        main.servo_hwi.move_to(cycle_targets(coef, 0.0, amp=0.0))
        countdown(start_delay)
        phase, tilted, step = 0.0, 0, 0
        next_time = time.perf_counter()
        while True:
            t = step * dt
            if t >= total:
                print("Sweep done")
                break
            freq = [f for start, f in plan if start <= t][-1]
            ramp = min(1.0, t / RAMP_SECONDS)
            target = main.servo_hwi.clamp(cycle_targets(coef, phase, amp=amp * ramp))
            start_time = time.perf_counter()
            obs = main.get_observation()
            main.servo_hwi.set_positions(target)
            recorder.add(start_time, obs, target / main.policy.action_scale, target,
                         main.gravity, main.accel)
            if fall_tilt:
                tilted = tilted + 1 if main.tilt() > fall_tilt else 0
                if tilted >= FALL_STEPS:
                    print(f"Fell: tilted {main.tilt():.0f} degrees at {t:.1f} s ({freq:g} Hz)")
                    break
            if step % 50 == 0:
                print(f"t {t:5.1f} s  {freq:g} Hz")
            # Advance the phase at this segment's frequency, wrapped to [-pi, pi).
            phase = (phase + 2 * np.pi * freq * dt + np.pi) % (2 * np.pi) - np.pi
            step += 1
            next_time += dt
            now = time.perf_counter()
            if next_time <= now:
                next_time = now + dt
            time.sleep(max(0.0, next_time - now))
    finally:
        main.stop()
        path = recorder.save()
        print(f"Recorded {len(recorder.t)} steps to {path}" if path else "Nothing recorded")

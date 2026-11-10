import math
import time

import click
from gnoci_v2.control.main import GAIT_FREQ, Main
from gnoci_v2.control.record import Recorder
from gnoci_v2.control.predict import DEFAULT_MODEL
from gnoci_v2.networking.channel import Channel


@click.command()
@click.option(
    "--model",
    "model_name",
    default=DEFAULT_MODEL,
    show_default=True,
    help="The experiment the policy was trained in. Loads models/<model>/policy.onnx.",
)
@click.option(
    "--close-loop",
    "close_loop",
    default=False,
    show_default=True,
    help="Whether to close the loop and move the servos.",
    is_flag=True,
)
@click.option(
    "--torque-off",
    "torque_off",
    default=False,
    show_default=True,
    help="Turn the servo torque off after moving home, so the legs can be moved by hand.",
    is_flag=True,
)
@click.option(
    "--verbose",
    "verbose",
    default=False,
    show_default=True,
    help="Whether to print verbose output.",
    is_flag=True,
)
@click.option(
    "--host",
    "host",
    default="0.0.0.0",
    show_default=True,
    help="The address to listen on for a stream client. 0.0.0.0 accepts any network.",
)
@click.option(
    "--port",
    "port",
    default=5000,
    show_default=True,
    help="The port to listen on for a stream client.",
)
@click.option(
    "--stream",
    "stream",
    default=False,
    show_default=True,
    help="Stream each step's observation to a connected client.",
    is_flag=True,
)
@click.option(
    "--record",
    "record",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save every step's observation, action and target to this .npz when"
    " stopped (Ctrl-C). Compare with the sim using scripts/hang_rollout_sim.py --real.",
)
@click.option(
    "--start-delay",
    "start_delay",
    default=5.0,
    show_default=True,
    type=click.FloatRange(min=0),
    help="Seconds to hold the home pose before the policy starts, e.g. to set the robot down.",
)
def run(model_name, close_loop, torque_off, verbose, host, port, stream, record, start_delay):
    if close_loop and torque_off:
        raise click.UsageError("--close-loop and --torque-off can't be used together.")

    print("Running...")
    main = Main(
        model_name=model_name,
        close_loop=close_loop,
        torque_off=torque_off,
        verbose=verbose
    )
    print(
        f"Loaded policy {main.policy.name} from {main.policy.path}"
        f" (action_scale {main.policy.action_scale})"
    )

    channel = None
    if stream:
        channel = Channel(host, port)
        channel.start()

    recorder = None
    if record is not None:
        recorder = Recorder(
            record,
            model=main.policy.name,
            action_scale=main.policy.action_scale,
            control_freq=main.control_freq,
            gait_freq=GAIT_FREQ,
            close_loop=close_loop,
        )

    # Torque goes off however this ends, including an error or Ctrl-C.
    try:
        main.init_main_process()
        countdown(start_delay)
        main.start(channel, recorder)
    finally:
        main.stop()
        if channel is not None:
            channel.stop()
        if recorder is not None:
            path = recorder.save()
            if path is None:
                print("Nothing recorded")
            else:
                print(f"Recorded {len(recorder.t)} steps to {path}")
        print("Stopped")


def countdown(seconds):
    """Waits seconds, printing the time left each second. The servos hold home."""
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        print(f"Starting the policy in {math.ceil(left)} s")
        time.sleep(min(1.0, left))
    print("Policy started")

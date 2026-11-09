import click
from gnoci_v2.control.main import Main
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
def run(model_name, close_loop, torque_off, verbose, host, port, stream):
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

    # Torque goes off however this ends, including an error or Ctrl-C.
    try:
        main.init_main_process()
        main.start(channel)
    finally:
        main.stop()
        if channel is not None:
            channel.stop()
        print("Stopped")

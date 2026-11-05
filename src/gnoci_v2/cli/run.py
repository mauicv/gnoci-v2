import click
from gnoci_v2.control.main import Main
from gnoci_v2.control.predict import DEFAULT_MODEL


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
    "--verbose",
    "verbose",
    default=False,
    show_default=True,
    help="Whether to print verbose output.",
    is_flag=True,
)
def run(model_name, close_loop, verbose):
    print("Running...")
    main = Main(
        model_name=model_name,
        close_loop=close_loop,
        verbose=verbose
    )
    print(f"Loaded policy {main.policy.name} from {main.policy.path}")
    # Torque goes off however this ends, including an error or Ctrl-C.
    try:
        main.init_main_process()
        main.start()
    finally:
        main.stop()
        print("Stopped")

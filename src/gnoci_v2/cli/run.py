import click
from gnoci_v2.control.main import Main


@click.command()
def run():
    print("Running...")
    main = Main()
    # Torque goes off however this ends, including an error or Ctrl-C.
    try:
        main.init_main_process()
        main.start()
    finally:
        main.stop()
        print("Stopped")
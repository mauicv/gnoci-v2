import logging
import click

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


try:
    from gnoci_v2.cli.testing import (
        test_servos,
    )
    cli.add_command(test_servos)
except ImportError as error:
    logger.error(error)
    raise error


if __name__ == "__main__":
    cli()
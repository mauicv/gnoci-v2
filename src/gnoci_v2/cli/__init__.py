import logging
import click

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


try:
    from gnoci_v2.cli.configure import measure_imu_offsets
    from gnoci_v2.cli.testing import (
        test_contacts,
        test_imu,
        test_servos,
    )
    cli.add_command(measure_imu_offsets)
    cli.add_command(test_contacts)
    cli.add_command(test_imu)
    cli.add_command(test_servos)
except ImportError as error:
    logger.error(error)
    raise error


if __name__ == "__main__":
    cli()
import click
from gnoci_v2.control.imu_hwi import IMUHWI


@click.command()
def measure_imu_offsets():
    """Measures the IMU's offsets. The robot must be still with its torso level."""
    print("Measuring IMU offsets, keep the robot still...")
    imu_hwi = IMUHWI(use_offsets=False)
    try:
        offsets = imu_hwi.measure_offsets()
    except ValueError as error:
        raise click.ClickException(str(error))
    print(f"Gyro offset (rad/s): {offsets['gyro']}")
    print(f"Accel offset (g): {offsets['accel']}")
    print(f"Saved to {imu_hwi.offsets_path}")

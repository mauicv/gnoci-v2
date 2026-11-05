import click
from gnoci_v2.control.hardware import ServoController
import numpy as np
import time


@click.command()
@click.option("--servo-name", type=str, help="The ID of the servo to test")
@click.option(
    "--amplitude",
    type=click.FloatRange(min=0.0),
    default=0.1,
    show_default=True,
    help="How far to move either side of home, in radians. Joint limits still apply.",
)
def test_servos(servo_name, amplitude):
    print("Testing servo...")
    servo_controller = ServoController()
    servo_id = servo_controller.get_id_from_name(servo_name)

    upper_limit = min(amplitude, servo_controller.UPPER_LIMITS[servo_id])
    lower_limit = max(-amplitude, servo_controller.LOWER_LIMITS[servo_id])

    # Torque goes off however this ends, including an error or Ctrl-C.
    try:
        servo_controller.init_servos()

        positions = servo_controller.get_positions()
        velocities = servo_controller.get_velocities()

        print(f"Upper limit: {upper_limit}")
        print(f"Lower limit: {lower_limit}")
        print(f"Servo ID: {servo_id}")
        print(f"Positions: {positions}")
        print(f"Velocities: {velocities}")

        for position in (upper_limit, lower_limit, 0.0):
            goal_position = [
                0.0 if i != servo_id
                else position for i in range(len(servo_controller.SERVO_IDS))
            ]
            servo_controller.move_to(goal_position, speed=0.2)
            time.sleep(0.5)
    finally:
        servo_controller.disable()

    print("Test complete")

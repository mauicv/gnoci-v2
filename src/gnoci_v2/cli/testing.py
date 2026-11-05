import click
from evdev import ecodes
from gnoci_v2.control.servo_hwi import ServoHWI
from gnoci_v2.control.contact_hwi import ContactHWI
from gnoci_v2.control.imu_hwi import IMUHWI
from gnoci_v2.control.joystick_hwi import JoystickHWI
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
    servo_controller = ServoHWI()
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


@click.command()
def test_contacts():
    print("Testing contacts...")
    contact_hwi = ContactHWI()
    while True:
        left, right = contact_hwi.get_contacts()
        print(f"Left: {left}, Right: {right}")
        time.sleep(0.01)


@click.command()
def test_imu():
    print("Testing IMU...")
    imu_hwi = IMUHWI()
    while True:
        gyro, gravity = imu_hwi.read()
        print(f"Gyro (rad/s): {np.round(gyro, 3)}, Gravity: {np.round(gravity, 3)}")
        time.sleep(imu_hwi.filter.dt)


@click.command()
def test_joystick():
    print("Testing joystick...")
    joystick_hwi = JoystickHWI()
    axes = [ecodes.ABS[axis] for axis, _ in joystick_hwi.axes]
    print(f"Device: {joystick_hwi.dev.name}, axes for vx, vy, yaw_rate: {axes}")
    while True:
        print(f"Command [vx, vy, yaw_rate]: {np.round(joystick_hwi.read(), 3)}")
        time.sleep(0.02)
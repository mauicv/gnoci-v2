import click
from gnoci_v2.control.hardware import ServoController


@click.command()
@click.option("--servo-name", type=str, help="The ID of the servo to test")
def test_servos(servo_name):
    print("Testing servo...")
    servo_controller = ServoController()
    servo_id = servo_controller.get_id_from_name(servo_name)
    servo_controller.enable()
    servo_controller.init_servos()

    upper_limit = servo_controller.UPPER_LIMITS[servo_id]
    lower_limit = servo_controller.LOWER_LIMITS[servo_id]


    positions = servo_controller.get_positions()
    velocities = servo_controller.get_velocities()

    print(f"Upper limit: {upper_limit}")
    print(f"Lower limit: {lower_limit}")
    print(f"Servo ID: {servo_id}")
    print(f"Positions: {positions}")
    print(f"Velocities: {velocities}")

    for position in np.linspace(0, upper_limit, 10):
        goal_position = [
            0.0 if i != servo_id 
            else position for i in range(len(servo_controller.SERVO_IDS))
        ]
        servo_controller.set_positions(goal_position)
        time.sleep(0.2)

    for position in np.linspace(upper_limit, lower_limit, 10):
        goal_position = [
            0.0 if i != servo_id 
            else position for i in range(len(servo_controller.SERVO_IDS))
        ]
        servo_controller.set_positions(goal_position)
        time.sleep(0.2)

    for position in np.linspace(lower_limit, 0, 10):
        goal_position = [
            0.0 if i != servo_id 
            else position for i in range(len(servo_controller.SERVO_IDS))
        ]
        servo_controller.set_positions(goal_position)
        time.sleep(0.2)

    servo_controller.disable()
    print("Test complete")
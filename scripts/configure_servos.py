from rustypot import Sts3215PyController
import numpy as np
import time

def read_motor_data(controller, current_id):
    i_pos = controller.read_present_position(current_id)[0]
    kp = controller.read_p_coefficient(current_id)[0]
    ki = controller.read_i_coefficient(current_id)[0]
    kd = controller.read_d_coefficient(current_id)[0]
    max_acceleration = controller.read_maximum_acceleration(current_id)[0]
    acceleration = controller.read_acceleration(current_id)[0]
    mode = controller.read_mode(current_id)[0]

    print(f"Motor {current_id}")
    print(f"position: {i_pos:.3f} \nkp: {kp}, \nki: {ki}, \nkd: {kd}, \nmax_acceleration: {max_acceleration:.3f}, \nacceleration: {acceleration:.3f}, \nmode: {mode}")
    print("--------------------------------")

if __name__ == "__main__":
    controller = Sts3215PyController(
        serial_port='/dev/serial0',
        baudrate=1_000_000,
        timeout=0.1
    )
    
    for current_id in range(1, 7):
        controller.write_goal_position(current_id, 0)

    time.sleep(2)

    for current_id in range(1, 7):
        print("================================================")
        read_motor_data(controller, current_id)

        controller.write_lock(current_id, False)               # lock = 0 (bool register, addr 55)
        controller.write_mode(current_id, 0)                   # 0 = position mode (addr 33)
        controller.write_maximum_acceleration(current_id, 0)   # addr 85
        controller.write_acceleration(current_id, 0)           # addr 41
        controller.write_p_coefficient(current_id, 32)         # addr 21
        controller.write_i_coefficient(current_id, 0)          # addr 23
        controller.write_d_coefficient(current_id, 0)          # addr 22
        controller.write_lock(current_id, True)                # re-lock

        read_motor_data(controller, current_id)
        print("================================================")
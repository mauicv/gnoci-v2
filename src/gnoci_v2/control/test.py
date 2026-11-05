
from rustypot import Sts3215PyController
import numpy as np
import time
from evdev import InputDevice, list_devices, ecodes
import gpiod
from gpiod.line import Direction, Bias


# ### Servo controller code: ####

# c = Sts3215PyController(serial_port='/dev/serial0', baudrate=1_000_000, timeout=0.1)

# servo_ids = [1, 2, 3, 4, 5, 6]

# for servo_id in servo_ids:
#     pos = c.read_present_position(servo_id)
#     print(f"Servo {servo_id}: {pos}")

# for servo_id in servo_ids:
#     c.write_goal_position(servo_id, np.deg2rad(120.0))


# time.sleep(2)

# for servo_id in servo_ids:
#     pos = c.read_present_position(servo_id)
#     print(f"Servo {servo_id}: {pos}")



# ### Joystick controller code: ####

# print(list_devices())

# dev = InputDevice("/dev/input/event0")

# for e in dev.read_loop():
#     if e.type == ecodes.EV_ABS:
#         print("axis", ecodes.ABS[e.code], e.value)
#     elif e.type == ecodes.EV_KEY:
#         print("button", ecodes.BTN.get(e.code, e.code), e.value)


# ### Read gpio feet contact ####

# req = gpiod.request_lines(
#     "/dev/gpiochip0", consumer="feet",
#     config={(6, 13): gpiod.LineSettings(
#         direction=Direction.INPUT,
#         bias=Bias.PULL_UP,
#         active_low=True)},
# )

# while True:
#     # in control loop:
#     left, right = req.get_values([6, 13])
#     print(f"Left: {left}, Right: {right}")
#     time.sleep(0.01)
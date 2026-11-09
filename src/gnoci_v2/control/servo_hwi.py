from rustypot import Sts3215PyController
import numpy as np
import time
from evdev import InputDevice, list_devices, ecodes
import gpiod
from gpiod.line import Direction, Bias

# 1, 2 left and right lateral hip servos
# 3, 4 left and right hip servos
# 5, 6 left and right knee servos

"""
1: 0.245 left lat hip inward flex, -0.506 left lat hip outward flex, +ve is inward
2: -0.245 right lat hip inward flex, 0.53 right lat hip outwards flex, +ve is outward
3:  0.85 left hip forward, -1.110 left hip back, +ve is forward
4: 1.18 right hip back, -0.842 right hip forward, +ve is back
5: 1.3 lower left leg forward, -2.3 lower left leg back
6: -1.32 lower right legt rowarad, 2.19 lower right leg back, +ve is back
"""

class ServoHWI:
    SERVO_IDS = [1, 3, 5, 2, 4, 6]
    SERVO_NAMES = [
        'left_hip_lat',
        'left_hip',
        'left_knee',
        'right_hip_lat',
        'right_hip',
        'right_knee'
    ]
    INITIAL_POSITIONS = np.array([0,0,0,0,0,0])

    # The arrays below follow the sim's joint order, the same as SERVO_IDS:
    # left_hip_lat, left_hip, left_knee, right_hip_lat, right_hip, right_knee.

    # -1 where a positive servo angle turns the opposite way to the sim joint.
    SIGNS = np.array([-1, -1, -1, -1, 1, 1])

    # Measured hard stops from the note above, converted to sim joint angles.
    LOWER_LIMITS = np.array([-0.245, -0.85, -1.3, -0.53, -0.842, -1.32])
    UPPER_LIMITS = np.array([0.506, 1.11, 2.3, 0.245, 1.18, 2.19])
    # Keep targets this far (radians) inside the hard stops.
    LIMIT_MARGIN = 0.05

    def __init__(self,
            port='/dev/serial0',
            baudrate=1_000_000,
            timeout=0.1,
        ):
        self.controller = Sts3215PyController(
            serial_port=port,
            baudrate=baudrate,
            timeout=timeout
        )
        self.kps = [32 for _ in range(len(self.SERVO_IDS))]  # default kp
        self.kds = [0 for _ in range(len(self.SERVO_IDS))]  # default kp
        self.low_torque_kps = [1 for _ in range(len(self.SERVO_IDS))]


    def get_id_from_name(self, name):
        return self.SERVO_NAMES.index(name)

    def init_servos(self):
        """Turns torque on without the servos jumping, then moves slowly to home."""
        self.controller.sync_write_p_coefficient(self.SERVO_IDS, self.low_torque_kps)
        # Hold the current pose, so torque doesn't drive to an old goal position.
        present = self.controller.sync_read_present_position(self.SERVO_IDS)
        self.controller.sync_write_goal_position(self.SERVO_IDS, present)
        self.enable()
        self.controller.sync_write_p_coefficient(self.SERVO_IDS, self.kps)
        self.move_to(np.zeros(len(self.SERVO_IDS)))

    def enable(self):
        self.controller.sync_write_torque_enable(
            self.SERVO_IDS, [True for _ in self.SERVO_IDS]
        )

    def disable(self):
        self.controller.sync_write_torque_enable(
            self.SERVO_IDS, [False for _ in self.SERVO_IDS]
        )

    def move_to(self, targets, speed=0.5, dt=0.02):
        """Moves in a straight line from the current pose at up to speed rad/s."""
        start = np.array(self.get_positions())
        targets = np.array(targets, dtype=float)
        distance = np.max(np.abs(targets - start))
        steps = max(1, int(np.ceil(distance / (speed * dt))))
        for step in range(1, steps + 1):
            self.set_positions(start + (targets - start) * step / steps)
            time.sleep(dt)

    def clamp(self, positions):
        """The joint angles set_positions actually sends: kept inside the hard stops."""
        return np.clip(
            positions,
            self.LOWER_LIMITS + self.LIMIT_MARGIN,
            self.UPPER_LIMITS - self.LIMIT_MARGIN,
        )

    def set_positions(self, positions):
        """Moves the servos to joint angles given in sim order and sim convention."""
        clamped = self.clamp(positions)
        update_pos = self.SIGNS * clamped + self.INITIAL_POSITIONS
        self.controller.sync_write_goal_position(self.SERVO_IDS, update_pos.tolist())

    def get_positions(self):
        """Returns joint angles in sim order and sim convention."""
        values = self.controller.sync_read_present_position(self.SERVO_IDS)
        return (self.SIGNS * (np.array(values) - self.INITIAL_POSITIONS)).tolist()

    def get_velocities(self):
        values = self.controller.sync_read_present_velocity(self.SERVO_IDS)
        return (self.SIGNS * np.array(values)).tolist()

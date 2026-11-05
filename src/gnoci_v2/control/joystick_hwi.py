import numpy as np
from evdev import InputDevice, ecodes


class JoystickHWI:
    """Turns a controller's sticks into the policy's [vx, vy, yaw_rate] command.

    Left stick: up is forward (+vx), left is +vy. Right stick: left is turning
    left (+yaw_rate). The units are m/s, m/s and rad/s, in the robot's frame.
    """

    # The ranges the policy trained on (lin_vel_x, lin_vel_y, ang_vel_yaw in joystick.py).
    TRAINING_LIMITS = np.array([0.3, 0.2, 1.0])
    # The command at full stick. Starts at half of the training ranges.
    DEFAULT_MAX_COMMAND = TRAINING_LIMITS / 2
    DEAD_ZONE = 0.1  # Fraction of the stick's travel that counts as centred.

    def __init__(self, device="/dev/input/event0", max_command=DEFAULT_MAX_COMMAND):
        self.dev = InputDevice(device)
        self.max_command = np.clip(max_command, 0.0, self.TRAINING_LIMITS)
        self.connected = True

        available = self.dev.capabilities(absinfo=False).get(ecodes.EV_ABS, [])
        # USB adapters differ on the right stick: usually ABS_Z, sometimes ABS_RX.
        right_x = ecodes.ABS_RX if ecodes.ABS_RX in available else ecodes.ABS_Z
        # (axis, sign), in command order. Sticks read lowest when up or left.
        self.axes = [(ecodes.ABS_Y, -1.0), (ecodes.ABS_X, -1.0), (right_x, -1.0)]

        missing = [
            ecodes.ABS[axis] for axis, _ in self.axes if axis not in available
        ]
        if missing:
            raise ValueError(f"{device} ({self.dev.name}) has no axes {missing}")

        self.axis_info = {axis: self.dev.absinfo(axis) for axis, _ in self.axes}
        self.raw = {axis: info.value for axis, info in self.axis_info.items()}

    def _normalize(self, axis):
        """Returns the axis from -1 to 1, with the dead zone removed."""
        info = self.axis_info[axis]
        centre = (info.max + info.min) / 2
        half_range = (info.max - info.min) / 2
        value = np.clip((self.raw[axis] - centre) / half_range, -1.0, 1.0)
        # Rescale, so the command rises smoothly from 0 at the dead zone's edge.
        magnitude = max(abs(value) - self.DEAD_ZONE, 0.0) / (1.0 - self.DEAD_ZONE)
        return np.sign(value) * magnitude

    def read(self):
        """Returns the latest command without blocking. Zeros if disconnected."""
        if not self.connected:
            return np.zeros(3)
        try:
            for event in self.dev.read():
                if event.type == ecodes.EV_ABS and event.code in self.raw:
                    self.raw[event.code] = event.value
        except BlockingIOError:
            pass  # No new events since the last read.
        except OSError as error:
            print(f"Joystick disconnected, command is now zero: {error}")
            self.connected = False
            return np.zeros(3)

        sticks = np.array([sign * self._normalize(axis) for axis, sign in self.axes])
        # The +0.0 turns a -0.0 from a centred stick into 0.0.
        return sticks * self.max_command + 0.0

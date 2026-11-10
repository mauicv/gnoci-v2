import math
import numpy as np


class ComplementaryFilter:
    """Estimates the torso's tilt from the gyro and accelerometer.

    roll is about x and pitch is about y, both in radians.
    """

    def __init__(self, alpha=0.98, dt=0.02):
        self.alpha = alpha
        # Must match how often update is called (ctrl_dt in the policy's env).
        self.dt = dt
        self.roll = None
        self.pitch = None

    def update(self, accel, gyro):
        """Takes accel in any unit (only its direction is used) and gyro in rad/s."""
        ax, ay, az = accel
        gx, gy, _ = gyro

        accel_roll = math.atan2(ay, az)
        accel_pitch = math.atan2(-ax, math.hypot(ay, az))

        # Start from the first reading, so it needn't settle after starting tilted.
        if self.roll is None:
            self.roll = accel_roll
            self.pitch = accel_pitch
            return

        self.roll = self.alpha * (self.roll + gx * self.dt) + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * (self.pitch + gy * self.dt) + (1 - self.alpha) * accel_pitch

    @property
    def gravity(self):
        """Unit gravity vector in the IMU frame; [0, 0, -1] when upright.

        This is the policy's gravity observation: site_xmat.T @ [0, 0, -1].
        """
        return np.array([
            math.sin(self.pitch),
            -math.sin(self.roll) * math.cos(self.pitch),
            -math.cos(self.roll) * math.cos(self.pitch),
        ])

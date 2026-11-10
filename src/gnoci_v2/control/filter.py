import math
import numpy as np

# Filter time constant, in seconds. Tilt changes faster than this come from the
# gyro, slower ones from the accelerometer. The old value, 0.16 s (alpha 0.98 at
# 300 Hz), followed the torso's sway while walking and read it as tilt: in the
# sim (scripts/imu_filter_sim.py) that gave 6-7 degrees of false roll and most
# episodes fell, while 0.5 s and longer didn't fall. Longer isn't free: a gyro
# bias of b rad/s reads as about b * tau rad of tilt, so the gyro offset is
# re-measured at startup (IMUHWI.measure_gyro_bias).
DEFAULT_TAU = 1.0


class ComplementaryFilter:
    """Estimates the torso's tilt from the gyro and accelerometer.

    roll is about x and pitch is about y, both in radians.
    """

    def __init__(self, tau=DEFAULT_TAU, dt=0.02):
        self.tau = tau
        # Must match how often update is called.
        self.dt = dt
        self.alpha = tau / (tau + dt)
        self.reset()

    def reset(self):
        """Restarts from the next reading's accelerometer tilt."""
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

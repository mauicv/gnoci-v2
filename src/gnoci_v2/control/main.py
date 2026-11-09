from gnoci_v2.control.imu_hwi import IMUHWI
from gnoci_v2.control.joystick_hwi import JoystickHWI
from gnoci_v2.control.contact_hwi import ContactHWI
from gnoci_v2.control.servo_hwi import ServoHWI
from gnoci_v2.control.loop import Loop
from gnoci_v2.control.predict import DEFAULT_MODEL, PolicyRunner

import numpy as np
import time

# Stop if the newest IMU reading is older than this, in seconds (5 IMU ticks).
IMU_MAX_AGE = 0.05

# Steps per second of each leg. Training sampled U(1.25, 1.5) per episode, so use the middle.
GAIT_FREQ = 1.375

# With --verbose, print the action and timing every this many steps (5 times a second).
PRINT_EVERY = 10


class Main:
    control_freq = 50

    def __init__(
            self,
            model_name=DEFAULT_MODEL,
            close_loop=False,
            torque_off=False,
            verbose=False
        ):
        # Load the policy first: a wrong model name fails before any hardware opens.

        self.policy = PolicyRunner(model_name)
        self.joystick_hwi = JoystickHWI()
        self.contact_hwi = ContactHWI()
        self.servo_hwi = ServoHWI()
        self.last_action = np.zeros(6)
        # Phase advance per control step, in radians.
        self.phase_dt = 2 * np.pi * GAIT_FREQ / self.control_freq
        self.dt = 1 / self.control_freq
        self.reset_phase()
        self.close_loop = close_loop
        self.torque_off = torque_off
        self.verbose = verbose
        # (gyro, gravity, perf_counter time of the reading), or None before the first.
        self.imu_data = None
        self.imu_hwi = IMUHWI(dt=0.00333333)

        def read_imu():
            gyro, gravity = self.imu_hwi.read()
            # One assignment, so the main thread never sees a half-updated reading.
            self.imu_data = (gyro, gravity, time.perf_counter())

        self.imu_loop = Loop(
            func=read_imu,
            hz=300,
        )
        self.ready = False

    def init_main_process(self):
        # Start the IMU first, so its filter settles while the servos move home.
        self.imu_loop.start()
        self.servo_hwi.init_servos()  # Blocks until the servos reach home.
        self.get_imu()  # Fails here, before walking, if the IMU isn't running.
        self.reset_phase()
        if self.torque_off:
            self.servo_hwi.disable()
        self.ready = True

    def get_imu(self):
        """Returns the latest (gyro, gravity). Raises if the IMU thread has stopped."""
        if self.imu_loop.failed:
            raise RuntimeError("The IMU loop failed, see the traceback above")
        if self.imu_data is None:
            raise RuntimeError("No IMU reading yet")
        gyro, gravity, timestamp = self.imu_data
        age = time.perf_counter() - timestamp
        if age > IMU_MAX_AGE:
            raise RuntimeError(f"IMU reading is stale ({age * 1000:.0f} ms old)")
        return gyro, gravity

    def reset_phase(self):
        """Starts the gait cycle as training did: [left, right], half a cycle apart."""
        self.phase = np.array([0.0, np.pi])

    def step_phase(self):
        """Returns the phase obs [cos_l, cos_r, sin_l, sin_r], then advances one step.

        Call once per control step. Matches joystick.py: the obs uses the phase
        before it advances, and it keeps advancing even with a zero command.
        """
        phase_obs = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
        # Wrap to [-pi, pi), the same as the sim.
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
        return phase_obs

    def get_observation(self):
        gyro, gravity = self.get_imu()
        joystick_data = self.joystick_hwi.read()
        joint_pos = self.servo_hwi.get_positions()
        joint_vel = self.servo_hwi.get_velocities()
        last_actions = self.last_action
        phase_obs = self.step_phase()
        return np.concatenate((
            gyro,
            gravity,
            joystick_data,
            joint_pos,
            joint_vel,
            last_actions,
            phase_obs
        ))

    def start(self, channel=None):
        if not self.ready:
            raise RuntimeError("Main process not ready")
        # Deadlines are fixed in advance (t0 + k*dt), so sleep overshoot on one
        # step shortens the next sleep instead of building up as drift.
        next_time = time.perf_counter()
        prev_start = next_time
        step = 0
        while True:
            start_time = time.perf_counter()
            observation = self.get_observation()
            action = self.policy.predict(observation)
            # The sim feeds back the raw action, before scaling.
            self.last_action = action

            if self.close_loop:
                # Targets are the default pose (all zeros) + action_scale * action,
                # with the scale the model was trained with (its config.json).
                self.servo_hwi.set_positions(self.policy.action_scale * action)

            end_time = time.perf_counter()
            if self.verbose and step % PRINT_EVERY == 0:
                rate = 1 / (start_time - prev_start) if step else float("nan")
                print(
                    f"Action: {np.round(action, 3)}, step took"
                    f" {(end_time - start_time) * 1000:.1f} ms, rate {rate:.1f} Hz"
                )
            prev_start = start_time
            step += 1

            next_time += self.dt
            now = time.perf_counter()
            if next_time <= now:
                print(f"Control budget exceeded by {(now - next_time) * 1000:.1f} ms")
                # Resync: hold this late action for a full step, rather than
                # replacing it straight away to catch up.
                next_time = now + self.dt

            if channel is not None:
                channel.publish(observation.tolist())

            time.sleep(next_time - now)

    def stop(self):
        self.imu_loop.stop()
        self.servo_hwi.disable()
        self.ready = False
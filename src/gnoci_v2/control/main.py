from gnoci_v2.control.imu_hwi import IMUHWI
from gnoci_v2.control.joystick_hwi import JoystickHWI
from gnoci_v2.control.contact_hwi import ContactHWI
from gnoci_v2.control.servo_hwi import ServoHWI
from gnoci_v2.control.loop import Loop
import time

# Stop if the newest IMU reading is older than this, in seconds (5 IMU ticks).
IMU_MAX_AGE = 0.05


class Main:
    def __init__(self):
        self.joystick_hwi = JoystickHWI()
        self.contact_hwi = ContactHWI()
        self.servo_hwi = ServoHWI()

        # (gyro, gravity, perf_counter time of the reading), or None before the first.
        self.imu_data = None
        self.imu_hwi = IMUHWI(dt=0.01)

        def read_imu():
            gyro, gravity = self.imu_hwi.read()
            # One assignment, so the main thread never sees a half-updated reading.
            self.imu_data = (gyro, gravity, time.perf_counter())

        self.imu_loop = Loop(
            func=read_imu,
            hz=100,
        )
        self.ready = False

    def init_main_process(self):
        # Start the IMU first, so its filter settles while the servos move home.
        self.imu_loop.start()
        self.servo_hwi.init_servos()  # Blocks until the servos reach home.
        self.get_imu()  # Fails here, before walking, if the IMU isn't running.
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

    def start(self):
        if not self.ready:
            raise RuntimeError("Main process not ready")
        while True:
            joystick_data = self.joystick_hwi.read()
            if joystick_data is not None:
                print(f"Joystick data: {joystick_data}")
            gyro, gravity = self.get_imu()
            print(f"IMU gyro: {gyro}, gravity: {gravity}")
            contact_data = self.contact_hwi.get_contacts()
            if contact_data is not None:
                print(f"Contact data: {contact_data}")
            servo_data = self.servo_hwi.get_positions()
            if servo_data is not None:
                print(f"Servo data: {servo_data}")
            velocities = self.servo_hwi.get_velocities()
            if velocities is not None:
                print(f"Velocities: {velocities}")
            time.sleep(1)

    def stop(self):
        self.imu_loop.stop()
        self.servo_hwi.disable()
        self.ready = False
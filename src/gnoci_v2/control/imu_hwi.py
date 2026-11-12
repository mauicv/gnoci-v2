import json
from pathlib import Path
import threading as th
import time

import numpy as np
import smbus2 as smbus

from gnoci_v2.control.filter import ComplementaryFilter


IMU_ADDR = 0x68

# MPU6050 registers.
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
PWR_MGMT_1 = 0x6B

CLOCK_GYRO_PLL = 0x01  # Also wakes the chip, which starts asleep.
LOW_PASS_44HZ = 0x03
GYRO_RANGE_500_DPS = 0x08
ACCEL_RANGE_4G = 0x08

# These depend on the ranges above.
GYRO_LSB_PER_DPS = 65.5
ACCEL_LSB_PER_G = 8192.0

OFFSETS_PATH = Path(__file__).with_name("imu_offsets.json")

# measure_offsets refuses to save if the robot is further than this from level
# (about 9 degrees) or the gyro is noisier than this, which means it was moving.
MAX_LEVEL_ERROR_G = 0.15
MAX_STILL_GYRO_STD = 0.05  # rad/s

bus_lock = th.Lock()


def init_mpu6050(bus, addr=IMU_ADDR):
    bus.write_byte_data(addr, PWR_MGMT_1, CLOCK_GYRO_PLL)
    time.sleep(0.1)  # settle
    bus.write_byte_data(addr, CONFIG, LOW_PASS_44HZ)
    bus.write_byte_data(addr, GYRO_CONFIG, GYRO_RANGE_500_DPS)
    bus.write_byte_data(addr, ACCEL_CONFIG, ACCEL_RANGE_4G)


def read_imu_data(bus):
    with bus_lock:
        return bus.read_i2c_block_data(IMU_ADDR, ACCEL_XOUT_H, 14)


def decode_imu(raw):
    """Returns (gyro in rad/s, accel in g), without any offsets applied."""
    # ax, ay, az, temperature, gx, gy, gz
    values = np.frombuffer(bytes(raw), dtype=">i2").astype(float)
    accel = values[:3] / ACCEL_LSB_PER_G
    gyro = np.deg2rad(values[4:7] / GYRO_LSB_PER_DPS)
    return gyro, accel


class IMUHWI:
    def __init__(self, dt=0.02, offsets_path=OFFSETS_PATH, use_offsets=True):
        self.offsets_path = Path(offsets_path)
        self.gyro_offset = np.zeros(3)
        self.accel_offset = np.zeros(3)
        if use_offsets:
            self.load_offsets()
        self.bus = smbus.SMBus(1)
        init_mpu6050(self.bus)
        self.filter = ComplementaryFilter(dt=dt)

    def load_offsets(self):
        try:
            offsets = json.loads(self.offsets_path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No IMU offsets at {self.offsets_path}. With the robot still and"
                " level, run: uv run --group control gnoci measure-imu-offsets"
            ) from None
        self.gyro_offset = np.array(offsets["gyro"])
        self.accel_offset = np.array(offsets["accel"])

    def read_raw(self):
        """Returns (gyro in rad/s, accel in g), without any offsets applied."""
        return decode_imu(read_imu_data(self.bus))

    def read(self):
        """Returns the policy's (gyro in rad/s, unit gravity vector), in the IMU frame.

        Call this once per control step, since each call advances the filter by dt.
        """
        gyro, _, gravity = self.read_full()
        return gyro, gravity

    def read_full(self):
        """As read, but returns (gyro in rad/s, accel in g, unit gravity vector).

        gyro and accel have the offsets removed: they're what the filter gets.
        """
        gyro, accel = self.read_raw()
        gyro = gyro - self.gyro_offset
        accel = accel - self.accel_offset
        self.filter.update(accel, gyro)
        return gyro, accel, self.filter.gravity

    def measure_offsets(self, samples=500, sample_dt=0.005):
        """Averages readings with the robot still and level, then saves the offsets."""
        readings = []
        for _ in range(samples):
            readings.append(self.read_raw())
            time.sleep(sample_dt)
        gyro, accel = (np.array(values) for values in zip(*readings))

        accel_offset = accel.mean(axis=0) - [0.0, 0.0, 1.0]  # Keeps gravity in accel.
        if np.max(np.abs(accel_offset)) > MAX_LEVEL_ERROR_G:
            raise ValueError(
                f"Mean accel {accel.mean(axis=0)} g is too far from [0, 0, 1]."
                " Is the torso level and the right way up?"
            )
        if np.max(gyro.std(axis=0)) > MAX_STILL_GYRO_STD:
            raise ValueError("The gyro is too noisy. Keep the robot still.")

        offsets = {"gyro": gyro.mean(axis=0).tolist(), "accel": accel_offset.tolist()}
        self.offsets_path.write_text(json.dumps(offsets, indent=2) + "\n")
        self.gyro_offset = np.array(offsets["gyro"])
        self.accel_offset = np.array(offsets["accel"])
        return offsets

import time
import numpy as np
from gnoci_v2.control.imu_hwi import IMUHWI

N = 2000

def time_reads(fn, n=N):
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        times[i] = time.perf_counter() - t0
    return times * 1000  # ms

def report(name, ms):
    print(f"{name}: mean {ms.mean():.3f}  median {np.median(ms):.3f}  "
          f"p99 {np.percentile(ms, 99):.3f}  max {ms.max():.3f} ms  "
          f"-> max rate ~{1000 / np.percentile(ms, 99):.0f} Hz (using p99)")

if __name__ == "__main__":
    imu = IMUHWI()
    time_reads(imu.read_raw, 100)  # warm up
    report("read_raw (I2C only)", time_reads(imu.read_raw))
    report("read (I2C + filter)", time_reads(imu.read))

    # How often the sensor actually produces new data: count reads that differ from the last one.
    raw = [imu.read_raw()[0].copy() for _ in range(N)]
    changed = sum(not np.array_equal(a, b) for a, b in zip(raw, raw[1:]))
    print(f"{changed}/{N - 1} back-to-back reads returned new gyro values")
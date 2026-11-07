from gnoci_v2.control.servo_hwi import ServoHWI
import numpy as np
import time


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
    # Torque stays off: reading doesn't need it.
    servo_hwi = ServoHWI()
    controller = servo_hwi.controller
    servo_id = ServoHWI.SERVO_IDS[0]

    delays = [controller.read_return_delay_time(i)[0] for i in ServoHWI.SERVO_IDS]
    print(f"return_delay_time per servo (ids {ServoHWI.SERVO_IDS}): {delays}")

    time_reads(servo_hwi.get_positions, 100)  # warm up
    report("get_positions (all 6)", time_reads(servo_hwi.get_positions))
    report("get_velocities (all 6)", time_reads(servo_hwi.get_velocities))
    report(f"read_present_position (id {servo_id})",
           time_reads(lambda: controller.read_present_position(servo_id)))

    # How often the servo updates its position: count back-to-back reads that differ.
    # Move the joint by hand while this runs, or every read will be the same.
    print("Move the first joint by hand for the next few seconds...")
    reads = [controller.read_present_position(servo_id)[0] for _ in range(N)]
    changed = sum(a != b for a, b in zip(reads, reads[1:]))
    print(f"{changed}/{N - 1} back-to-back reads returned a new position")
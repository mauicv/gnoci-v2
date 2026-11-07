"""Step one joint on the robot and record its response as fast as the bus allows.

Put the robot on a stand with the legs free. The servos move home, the joint
is held there for --pre seconds, then its target steps to --step. Only that
servo is read during the test (one servo is faster to read than all six), and
every sample is timestamped by the middle of its read.

  uv run --group control python scripts/measure_servo_response_real.py --joint left_hip_lat --step 0.3

Then copy the .npz to the laptop and compare with the sim:
  uv run python scripts/measure_servo_response_sim.py --real logs/servo_step/real_left_hip_lat.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np

from gnoci_v2.control.servo_hwi import ServoHWI

OUT_DIR = Path("logs/servo_step")


def main() -> None:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--joint", default="left_hip_lat", choices=ServoHWI.SERVO_NAMES)
  parser.add_argument("--step", type=float, default=0.3, help="Target after the step, rad (sim convention).")
  parser.add_argument("--pre", type=float, default=0.3, help="Seconds recorded at rest before the step.")
  parser.add_argument("--duration", type=float, default=0.5, help="Seconds recorded after the step.")
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  servo_hwi = ServoHWI()
  index = servo_hwi.get_id_from_name(args.joint)
  servo_id = ServoHWI.SERVO_IDS[index]
  sign = ServoHWI.SIGNS[index]
  low = ServoHWI.LOWER_LIMITS[index] + ServoHWI.LIMIT_MARGIN
  high = ServoHWI.UPPER_LIMITS[index] - ServoHWI.LIMIT_MARGIN
  if not low <= args.step <= high:
    raise SystemExit(f"--step {args.step} is outside {args.joint}'s clamp [{low:.3f}, {high:.3f}]")

  targets = np.zeros(len(ServoHWI.SERVO_IDS))
  targets[index] = args.step
  controller = servo_hwi.controller

  def read():
    """(time of the sample, joint angle in sim convention)."""
    t0 = time.perf_counter()
    raw = controller.read_present_position(servo_id)[0]
    t1 = time.perf_counter()
    return (t0 + t1) / 2, sign * raw

  times, positions = [], []
  try:
    servo_hwi.init_servos()  # Torque on, then a slow move home.
    time.sleep(0.5)  # Let it settle.

    end = time.perf_counter() + args.pre
    while time.perf_counter() < end:
      t, p = read()
      times.append(t)
      positions.append(p)

    servo_hwi.set_positions(targets)
    t_command = time.perf_counter()  # A write gets no reply: the packet has gone.

    end = t_command + args.duration
    while time.perf_counter() < end:
      t, p = read()
      times.append(t)
      positions.append(p)

    servo_hwi.move_to(np.zeros(len(ServoHWI.SERVO_IDS)))
  finally:
    servo_hwi.disable()

  t = np.array(times) - t_command
  positions = np.array(positions)
  intervals = np.diff(t) * 1000
  print(f"{len(t)} samples, interval median {np.median(intervals):.2f} ms,"
        f" p99 {np.percentile(intervals, 99):.2f} ms, max {intervals.max():.2f} ms")
  print(f"Start {positions[t < 0].mean():+.4f} rad, end {positions[-20:].mean():+.4f} rad,"
        f" target {args.step:+.4f} rad")

  out = args.out or OUT_DIR / f"real_{args.joint}.npz"
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, t=t, position=positions, target=args.step, joint=args.joint)
  print(f"Saved {out}")


if __name__ == "__main__":
  main()

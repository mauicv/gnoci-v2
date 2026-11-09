"""Step one joint on the robot and record its response as fast as the bus allows.

Put the robot on a stand with the legs free. The servos move home, the joint
is held there for --pre seconds, then its target steps to --step. Only that
servo is read during the test (one servo is faster to read than all six), and
every sample is timestamped by the middle of its read.

Each sample reads the position, then the servo's own velocity (Present Speed)
through the same rustypot conversion ServoHWI.get_velocities uses, so the sim
script can check it against the differentiated position (units) and against
the sim's joint velocity. Each read has its own timestamp. --no-velocity reads
the position only, which halves the sample interval.

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
  parser.add_argument("--no-velocity", action="store_true", help="Don't read the servo's velocity.")
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

  times, positions = [], []
  velocity_times, velocities = [], []

  def read():
    """Records the joint angle (and velocity) in sim convention, with read times."""
    t0 = time.perf_counter()
    raw = controller.read_present_position(servo_id)[0]
    t1 = time.perf_counter()
    times.append((t0 + t1) / 2)
    positions.append(sign * raw)
    if not args.no_velocity:
      raw = controller.read_present_velocity(servo_id)[0]
      t2 = time.perf_counter()
      velocity_times.append((t1 + t2) / 2)
      velocities.append(sign * raw)

  try:
    servo_hwi.init_servos()  # Torque on, then a slow move home.
    time.sleep(0.5)  # Let it settle.

    end = time.perf_counter() + args.pre
    while time.perf_counter() < end:
      read()

    servo_hwi.set_positions(targets)
    t_command = time.perf_counter()  # A write gets no reply: the packet has gone.

    end = t_command + args.duration
    while time.perf_counter() < end:
      read()

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

  extra = {}
  if velocities:
    velocities = np.array(velocities)
    extra = {"velocity_t": np.array(velocity_times) - t_command, "velocity": velocities}
    print(f"Servo-reported velocity: peak {np.abs(velocities).max():.2f} rad/s,"
          f" at rest std {velocities[extra['velocity_t'] < 0].std():.3f} rad/s")

  out = args.out or OUT_DIR / f"real_{args.joint}.npz"
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, t=t, position=positions, target=args.step, joint=args.joint, **extra)
  print(f"Saved {out}")


if __name__ == "__main__":
  main()

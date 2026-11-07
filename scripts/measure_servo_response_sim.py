"""Step one joint in the sim with the torso fixed, and compare with the real robot.

The sim counterpart of measure_servo_response_real.py, which steps the same
joint on the robot (on a stand) and saves the response. The model is the
training model (scene.xml) loaded unchanged, except that in memory the
torso's freejoint is removed, so the torso is fixed in the air like the robot
on its stand. The XML files aren't modified.

  # Sim only:
  uv run python scripts/measure_servo_response_sim.py --joint left_hip_lat --step 0.3
  # Compare with a real recording (copied from the Pi), and fit the extra delay:
  uv run python scripts/measure_servo_response_sim.py --real logs/servo_step/real_left_hip_lat.npz

Both responses are measured the same way: t = 0 is when the step is commanded,
and "dead time" is when the joint has moved --threshold rad. The sim has no
delay, so the real minus sim difference in dead time is the extra delay the
sim is missing. --real also fits that delay directly, by finding the time
shift that best lines the sim curve up with the real one.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from gnoci_v2.training.constants import SIM_PATH

SCENE = SIM_PATH / "desc" / "scene.xml"
OUT_DIR = Path("logs/servo_step")


def fixed_base_model(height, gravity=True):
  """The training model, with the torso fixed at height (m) instead of free."""
  spec = mujoco.MjSpec.from_file(str(SCENE))
  torso = spec.body("body_main")
  spec.delete(torso.joints[0])  # The freejoint.
  # The keyframes include the freejoint's 7 qpos, so they no longer fit.
  for key in list(spec.keys):
    spec.delete(key)
  torso.pos = [0.0, 0.0, height]
  model = spec.compile()
  if not gravity:
    model.opt.gravity[:] = 0.0
  return model


def simulate_step(model, joint, step, pre, duration, delay=0.0):
  """Holds all joints at 0 for pre seconds, then steps joint's target to step.

  Returns (t, position) with t = 0 at the command; the target changes delay
  seconds later. Sampled every physics step (sim_dt, 2 ms in training).
  """
  data = mujoco.MjData(model)
  actuator = model.actuator(joint).id
  qpos_adr = model.jnt_qposadr[model.joint(joint).id]
  dt = model.opt.timestep
  n_pre = int(round(pre / dt))
  n_post = int(round(duration / dt))

  times, positions = [], []
  for i in range(-n_pre, n_post):
    t = i * dt
    data.ctrl[:] = 0.0
    if t >= delay - 1e-9:
      data.ctrl[actuator] = step
    times.append(t)
    positions.append(data.qpos[qpos_adr])
    mujoco.mj_step(model, data)
  return np.array(times), np.array(positions)


def analyse(t, position, step, threshold):
  """Response metrics. t = 0 is the command."""
  pre = position[t < 0]
  p0 = pre.mean() if len(pre) else position[0]
  noise = pre.std() if len(pre) > 1 else 0.0
  moved = np.abs(position - p0) > threshold
  moved &= t >= 0
  dead = t[np.argmax(moved)] if moved.any() else np.nan

  after = t >= 0
  tail = position[after][-max(1, after.sum() // 5):]
  final = tail.mean()
  span = final - p0

  def crossing(fraction):
    reached = after & ((position - p0) / span >= fraction) if span else after & False
    return t[np.argmax(reached)] if reached.any() else np.nan

  peak = position[after].max() if span > 0 else position[after].min()
  # Differentiating raw samples amplifies encoder steps and noise, so resample
  # onto a 1 ms grid and smooth over 5 ms first.
  grid = np.arange(0.0, t[-1], 0.001)
  smooth = np.convolve(np.interp(grid, t, position), np.ones(5) / 5, mode="valid")
  speed = np.abs(np.gradient(smooth, 0.001))
  return {
      "start": p0,
      "noise_std": noise,
      "dead_time_ms": 1000 * dead,
      "t50_ms": 1000 * crossing(0.5),
      "rise_10_90_ms": 1000 * (crossing(0.9) - crossing(0.1)),
      "final": final,
      "steady_state_error": step - final,
      "overshoot_pct": 100 * (peak - final) / span if span else np.nan,
      "peak_speed": speed.max(),
  }


def fit_delay(sim_t, sim_p, real_t, real_p, max_shift=0.05, window=0.15):
  """The time shift of the sim curve that best matches the real one (RMS).

  Shifting works because both start at rest: a delayed step is the same
  response, later. Only the first `window` seconds after the command count,
  so the fit is about timing, not steady-state error.
  """
  mask = (real_t >= -0.02) & (real_t <= window)
  shifts = np.arange(0.0, max_shift + 1e-9, 0.0005)
  rms = []
  for shift in shifts:
    sim_on_real = np.interp(real_t[mask] - shift, sim_t, sim_p)
    rms.append(np.sqrt(np.mean((sim_on_real - real_p[mask]) ** 2)))
  rms = np.array(rms)
  best = int(np.argmin(rms))
  return shifts[best], rms[best], rms[0]


def print_metrics(name, metrics):
  print(
      f"{name:>6}: dead {metrics['dead_time_ms']:6.1f} ms  t50 {metrics['t50_ms']:6.1f} ms"
      f"  rise {metrics['rise_10_90_ms']:6.1f} ms  final {metrics['final']:+.3f} rad"
      f"  (error {metrics['steady_state_error']:+.3f})"
      f"  overshoot {metrics['overshoot_pct']:5.1f}%  peak speed {metrics['peak_speed']:.2f} rad/s"
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--real", type=Path, default=None,
                      help="A recording from measure_servo_response_real.py. Its joint, step and"
                           " timings are used, and the two are compared.")
  parser.add_argument("--joint", default="left_hip_lat")
  parser.add_argument("--step", type=float, default=0.3, help="Target after the step, rad.")
  parser.add_argument("--pre", type=float, default=0.3, help="Seconds held at 0 before the step.")
  parser.add_argument("--duration", type=float, default=0.5, help="Seconds recorded after the step.")
  parser.add_argument("--height", type=float, default=0.5,
                      help="Torso height (m), high enough that the feet never touch the floor.")
  parser.add_argument("--no-gravity", action="store_true", help="Turn gravity off.")
  parser.add_argument("--threshold", type=float, default=0.005,
                      help="Movement (rad) that counts as the joint starting to move. ~3 encoder counts.")
  parser.add_argument("--out", type=Path, default=None, help="Plot path (.png).")
  args = parser.parse_args()

  real = None
  if args.real:
    real = np.load(args.real)
    args.joint = str(real["joint"])
    args.step = float(real["target"])
    args.duration = float(real["t"][-1])
    print(f"Real recording {args.real}: {args.joint} to {args.step:+.3f} rad,"
          f" {len(real['t'])} samples, median interval {1000 * np.median(np.diff(real['t'])):.2f} ms")

  model = fixed_base_model(args.height, gravity=not args.no_gravity)
  sim_t, sim_p = simulate_step(model, args.joint, args.step, args.pre, args.duration)
  sim = analyse(sim_t, sim_p, args.step, args.threshold)

  print(f"{args.joint}: step 0 -> {args.step:+.3f} rad, torso fixed, gravity"
        f" {'off' if args.no_gravity else 'on'}, sim_dt {1000 * model.opt.timestep:g} ms")
  print_metrics("sim", sim)

  shift = None
  if real is not None:
    real_metrics = analyse(real["t"], real["position"], args.step, args.threshold)
    print_metrics("real", real_metrics)
    print(f"Dead time difference (real - sim): "
          f"{real_metrics['dead_time_ms'] - sim['dead_time_ms']:.1f} ms")
    shift, rms, rms0 = fit_delay(sim_t, sim_p, real["t"], real["position"])
    print(f"Best-fit extra delay: {1000 * shift:.1f} ms"
          f" (RMS error {rms:.4f} rad, vs {rms0:.4f} rad with no delay)")
    print("If the RMS stays large at the best shift, the curves differ in shape, not just timing:"
          " compare peak speed (speed limit?) and rise time (kp?).")

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(9, 5))
  ax.axvline(0, color="grey", lw=0.8, ls=":")
  ax.axhline(args.step, color="grey", lw=0.8, ls="--", label="target")
  ax.plot(1000 * sim_t, sim_p, label="sim (no delay)")
  if real is not None:
    ax.plot(1000 * real["t"], real["position"], ".", ms=2, label="real")
    ax.plot(1000 * (sim_t + shift), sim_p, label=f"sim + {1000 * shift:.1f} ms")
  ax.set_xlabel("time since command (ms)")
  ax.set_ylabel(f"{args.joint} (rad)")
  ax.set_xlim(-50, 1000 * min(args.duration, 0.3))
  ax.legend()
  ax.grid(alpha=0.3)
  out = args.out or OUT_DIR / f"step_{args.joint}{'_vs_real' if real is not None else ''}.png"
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=120, bbox_inches="tight")
  print(f"Saved plot {out}")


if __name__ == "__main__":
  main()

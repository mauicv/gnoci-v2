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

Velocity is checked too, as joint speed in rad/s:
  - sim: the joint's qvel, which is what the policy observes in training.
  - real, from position: the recorded position, differentiated.
  - real, from the servo: the servo's own velocity reading (Present Speed), the
    value the robot's policy observes. Recordings made before velocity was
    added don't have it.
The servo's reading should match the differentiated position, with a scale of
1.0. A different scale means the rustypot conversion isn't rad/s, so the
policy's joint velocity observation is off by that factor. Its lag shows how
much the servo smooths or delays the reading. The sim's peak qvel is compared
with the real peak speed, to check the sim's speed limit (forcerange / damping).
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

  Returns (t, position, velocity) with t = 0 at the command; the target
  changes delay seconds later. velocity is the joint's qvel. Sampled every
  physics step (sim_dt, 2 ms in training).
  """
  data = mujoco.MjData(model)
  actuator = model.actuator(joint).id
  joint_id = model.joint(joint).id
  qpos_adr = model.jnt_qposadr[joint_id]
  dof_adr = model.jnt_dofadr[joint_id]
  dt = model.opt.timestep
  n_pre = int(round(pre / dt))
  n_post = int(round(duration / dt))

  times, positions, velocities = [], [], []
  for i in range(-n_pre, n_post):
    t = i * dt
    data.ctrl[:] = 0.0
    if t >= delay - 1e-9:
      data.ctrl[actuator] = step
    times.append(t)
    positions.append(data.qpos[qpos_adr])
    velocities.append(data.qvel[dof_adr])
    mujoco.mj_step(model, data)
  return np.array(times), np.array(positions), np.array(velocities)


def differentiate(t, position, smooth_ms=5):
  """Signed velocity from position samples, as (t, velocity) on a 1 ms grid.

  Differentiating raw samples amplifies encoder steps and noise, so resample
  onto a 1 ms grid and smooth over smooth_ms first. The moving average is
  centred, so the result isn't shifted in time.
  """
  grid = np.arange(t[0], t[-1], 0.001)
  kernel = np.ones(smooth_ms) / smooth_ms
  smooth = np.convolve(np.interp(grid, t, position), kernel, mode="valid")
  # Sample i of a "valid" convolution is centred on grid[i + (smooth_ms - 1) / 2].
  grid = grid[: len(smooth)] + 0.001 * (smooth_ms - 1) / 2
  return grid, np.gradient(smooth, 0.001)


def fit_velocity_reading(reading_t, reading, ref_t, ref, max_lag=0.03, window=(-0.02, 0.2)):
  """How the servo's velocity reading relates to the reference velocity.

  Finds the lag (s) at which reading(t) best matches scale * ref(t - lag),
  with scale fitted by least squares. Returns (scale, lag, rms of the fit,
  rms with scale 1 and no lag).
  """
  mask = (reading_t >= window[0]) & (reading_t <= window[1])
  r = reading[mask]
  best = None
  for lag in np.arange(0.0, max_lag + 1e-9, 0.0005):
    x = np.interp(reading_t[mask] - lag, ref_t, ref)
    scale = (x @ r) / (x @ x) if x @ x > 0 else np.nan
    rms = np.sqrt(np.mean((r - scale * x) ** 2))
    if best is None or rms < best[2]:
      best = (scale, lag, rms)
  x0 = np.interp(reading_t[mask], ref_t, ref)
  return (*best, np.sqrt(np.mean((r - x0) ** 2)))


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
  grid, velocity = differentiate(t, position)
  speed = np.abs(velocity[grid >= 0])
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
  sim_t, sim_p, sim_v = simulate_step(model, args.joint, args.step, args.pre, args.duration)
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

  # Velocity.
  print("\nVelocity (rad/s, signed peak in the direction of the step):")
  sign = np.sign(args.step) or 1.0
  print(f"   sim qvel:                peak {sign * (sign * sim_v).max():+.2f}"
        f"  (differentiated sim position: {sign * sim['peak_speed']:+.2f})")
  real_dt = real_dv = None
  has_reading = real is not None and "velocity" in real.files
  if real is not None:
    real_dt, real_dv = differentiate(real["t"], real["position"])
    after = real_dt >= 0
    print(f"  real from position:       peak {sign * (sign * real_dv[after]).max():+.2f}")
  if has_reading:
    rv_t, rv = real["velocity_t"], real["velocity"]
    print(f"  real from servo reading:  peak {sign * (sign * rv[rv_t >= 0]).max():+.2f}"
          f"  (at rest: mean {rv[rv_t < 0].mean():+.3f}, std {rv[rv_t < 0].std():.3f},"
          f" {len(np.unique(rv))} distinct values, smallest step"
          f" {np.diff(np.unique(rv)).min() if len(np.unique(rv)) > 1 else np.nan:.4f})")
    scale, lag, rms, rms0 = fit_velocity_reading(rv_t, rv, real_dt, real_dv)
    print(f"  servo reading vs differentiated position: scale {scale:.3f}, lag {1000 * lag:.1f} ms"
          f" (RMS {rms:.3f} rad/s, vs {rms0:.3f} with scale 1 and no lag)")
    if not 0.9 <= scale <= 1.1:
      print(f"  WARNING: the servo's velocity reading is {scale:.3f}x the real joint speed, so"
            " ServoHWI.get_velocities isn't rad/s. The policy's joint_vel observation is off by"
            " that factor.")
  elif real is not None:
    print("  (this recording has no servo velocity reading: re-record it with the updated"
          " measure_servo_response_real.py)")

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, (ax, ax_v) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
  ax.axvline(0, color="grey", lw=0.8, ls=":")
  ax.axhline(args.step, color="grey", lw=0.8, ls="--", label="target")
  ax.plot(1000 * sim_t, sim_p, label="sim (no delay)")
  if real is not None:
    ax.plot(1000 * real["t"], real["position"], ".", ms=2, label="real")
    ax.plot(1000 * (sim_t + shift), sim_p, label=f"sim + {1000 * shift:.1f} ms")
  ax.set_ylabel(f"{args.joint} (rad)")
  ax.legend()
  ax.grid(alpha=0.3)

  ax_v.axvline(0, color="grey", lw=0.8, ls=":")
  ax_v.axhline(0, color="grey", lw=0.8)
  ax_v.plot(1000 * sim_t, sim_v, label="sim qvel (no delay)")
  if real is not None:
    ax_v.plot(1000 * (sim_t + shift), sim_v, label=f"sim qvel + {1000 * shift:.1f} ms")
    ax_v.plot(1000 * real_dt, real_dv, label="real, differentiated position")
  if has_reading:
    ax_v.plot(1000 * real["velocity_t"], real["velocity"], ".-", ms=3, lw=0.8,
              label="real, servo velocity reading")
  ax_v.set_xlabel("time since command (ms)")
  ax_v.set_ylabel(f"{args.joint} velocity (rad/s)")
  ax_v.set_xlim(-50, 1000 * min(args.duration, 0.3))
  ax_v.legend()
  ax_v.grid(alpha=0.3)
  out = args.out or OUT_DIR / f"step_{args.joint}{'_vs_real' if real is not None else ''}.png"
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=120, bbox_inches="tight")
  print(f"Saved plot {out}")


if __name__ == "__main__":
  main()

"""How far the robot's gravity estimate is from the truth while walking, in the sim.

Training observes the exact gravity direction (site_xmat.T @ [0, 0, -1]). The
robot instead estimates it with ComplementaryFilter (control/filter.py) from
the MPU6050's accelerometer and gyro, at 300 Hz. While walking the
accelerometer also measures the torso's own acceleration, which the filter
partly reads as tilt. This runs the policy (the robot's ONNX file) on the full,
free-floating model in CPU MuJoCo, and runs the same filter on the sim's
accelerometer and gyro sensors, to measure that error.

  --feed truth   the policy sees the true gravity, as in training (default)
  --feed filter  the policy sees the filter's gravity, as on the robot

  uv run python scripts/imu_filter_sim.py --model cat-fire-potential
  uv run python scripts/imu_filter_sim.py --model cat-fire-potential --feed filter
  uv run python scripts/imu_filter_sim.py --model cat-fire-potential --feed filter \
      --episodes 1 --video videos/imu_filter.mp4

The filter runs every physics substep (2 ms) with the robot's time constant
(DEFAULT_TAU in control/filter.py; --tau 0.163 for the old filter). The
sim's accelerometer has no sensor noise and rigid-contact impacts, so its
spikes may differ from the real chip's.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from gnoci_v2.control.filter import DEFAULT_TAU, ComplementaryFilter
from gnoci_v2.control.main import GAIT_FREQ
from gnoci_v2.control.predict import PolicyRunner
from gnoci_v2.training.constants import SIM_PATH

SCENE = SIM_PATH / "desc" / "scene.xml"
CTRL_DT, SIM_DT = 0.02, 0.002
OLD_ROBOT_TAU = 0.98 * (1 / 300) / (1 - 0.98)  # alpha 0.98 at 300 Hz
JOINTS = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]
DEFAULT_COMMANDS = ([0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.15, 0.0])


AFTER_FALL_STEPS = 50  # Steps still rendered after a fall, so the fall is visible.


def episode(model, policy, command, steps, feed, rng, tau, frame_fn=None):
  """Returns per-step (true gravity, filter gravity) and the step it fell at (or None).

  frame_fn(data, true_gravity, filter_gravity) is called once per control step.
  """
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
  # As training's reset: joints +U(-0.1, 0.1), kept inside their limits.
  qadr = [model.jnt_qposadr[model.joint(j).id] for j in JOINTS]
  dadr = [model.jnt_dofadr[model.joint(j).id] for j in JOINTS]
  lo, hi = model.jnt_range[[model.joint(j).id for j in JOINTS]].T
  data.qpos[qadr] = np.clip(rng.uniform(-0.1, 0.1, 6), lo, hi)
  data.ctrl[:] = data.qpos[qadr]
  mujoco.mj_forward(model, data)

  imu = model.site("imu").id
  gyro_adr = model.sensor_adr[model.sensor("gyro").id]
  acc_adr = model.sensor_adr[model.sensor("accelerometer").id]
  act = [model.actuator(j).id for j in JOINTS]
  filt = ComplementaryFilter(tau=tau, dt=SIM_DT)

  def true_gravity():
    return data.site_xmat[imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

  def sensors():
    return data.sensordata[acc_adr:acc_adr + 3].copy(), data.sensordata[gyro_adr:gyro_adr + 3].copy()

  # Start the filter at the true tilt: on the robot it starts with the robot still,
  # whereas here the first accelerometer reading includes the jolt of the reset.
  g0 = true_gravity()
  filt.roll = float(np.arctan2(-g0[1], -g0[2]))
  filt.pitch = float(np.arcsin(np.clip(g0[0], -1, 1)))
  phase = np.array([0.0, np.pi])
  last_action = np.zeros(6)
  n_sub = int(round(CTRL_DT / SIM_DT))
  truth, est = [], []
  for k in range(steps):
    g_true, g_filt = true_gravity(), filt.gravity
    truth.append(g_true)
    est.append(g_filt)
    if frame_fn is not None:
      frame_fn(data, g_true, g_filt)
    if g_true[2] > -0.5 or data.qpos[2] < 0.15:  # The env's termination.
      if frame_fn is not None:
        for _ in range(AFTER_FALL_STEPS):
          for _ in range(n_sub):
            mujoco.mj_step(model, data)
            filt.update(*sensors())
          frame_fn(data, true_gravity(), filt.gravity)
      return np.array(truth), np.array(est), k
    acc, gyro = sensors()
    obs = np.concatenate([
        gyro, g_filt if feed == "filter" else g_true, command,
        data.qpos[qadr], data.qvel[dadr], last_action,
        np.cos(phase), np.sin(phase),
    ])
    action = policy.predict(obs)
    last_action = action
    phase = np.fmod(phase + 2 * np.pi * GAIT_FREQ * CTRL_DT + np.pi, 2 * np.pi) - np.pi
    data.ctrl[act] = policy.action_scale * action
    for _ in range(n_sub):
      mujoco.mj_step(model, data)
      filt.update(*sensors())
  return np.array(truth), np.array(est), None


def angles(g):
  """(roll, pitch) in degrees, from gravity vectors in the IMU frame."""
  roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2]))
  pitch = np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1)))
  return roll, pitch


def main():
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", required=True)
  parser.add_argument("--feed", choices=["truth", "filter"], default="truth")
  parser.add_argument("--episodes", type=int, default=16)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--skip", type=int, default=50, help="Steps left out at the start.")
  parser.add_argument("--command", type=float, nargs=3, action="append", metavar=("VX", "VY", "YAW"))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--plot", type=Path, default=None, help="Plot one episode per command (.png).")
  parser.add_argument("--video", type=Path, default=None,
                      help="Render the first episode of each command, one after another (.mp4).")
  parser.add_argument("--camera", default="track")
  parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                      help="Filter time constant (s). Default: the robot's. The old robot"
                           f" filter (alpha 0.98 at 300 Hz) was {OLD_ROBOT_TAU:.3f} s.")
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(str(SCENE))
  model.opt.timestep = SIM_DT
  tau = args.tau
  alpha = tau / (tau + SIM_DT)
  policy = PolicyRunner(args.model)
  print(f"{args.model}, policy fed the {args.feed} gravity, filter time constant {tau:.3f} s"
        f" (alpha {alpha:.4f} at {1000 * SIM_DT:g} ms)")

  frames, renderer = [], None
  if args.video:
    from PIL import Image, ImageDraw
    renderer = mujoco.Renderer(model, height=480, width=640)

  def make_frame_fn(command):
    def frame_fn(data, g_true, g_filt):
      renderer.update_scene(data, camera=args.camera)
      image = Image.fromarray(renderer.render())
      draw = ImageDraw.Draw(image)
      (tr,), (tp,) = angles(g_true[None])
      (fr,), (fp,) = angles(g_filt[None])
      draw.text((10, 10), f"{args.model}, policy sees {args.feed} gravity", fill="white")
      draw.text((10, 25), f"command vx {command[0]:g} vy {command[1]:g} yaw {command[2]:g}",
                fill="white")
      draw.text((10, 45), f"true   roll {tr:+6.1f}  pitch {tp:+6.1f} deg", fill=(120, 180, 255))
      draw.text((10, 60), f"filter roll {fr:+6.1f}  pitch {fp:+6.1f} deg", fill=(255, 170, 80))
      frames.append(np.asarray(image))
    return frame_fn

  plots = []
  for command in args.command or DEFAULT_COMMANDS:
    rng = np.random.default_rng(args.seed)
    errs, falls, true_rp = [], 0, []
    for e in range(args.episodes):
      frame_fn = make_frame_fn(command) if args.video and e == 0 else None
      truth, est, fell = episode(model, policy, np.array(command), args.steps, args.feed, rng,
                                 tau, frame_fn)
      falls += fell is not None
      if len(truth) > args.skip:
        tr, tp = angles(truth[args.skip:])
        er, ep = angles(est[args.skip:])
        errs.append(np.stack([er - tr, ep - tp], 1))
        true_rp.append(np.stack([tr, tp], 1))
      if e == 0:
        plots.append((command, truth, est))
    err = np.concatenate(errs)
    rp = np.concatenate(true_rp)
    print(f"\ncommand {command}: fell {falls}/{args.episodes}")
    print(f"  true tilt (deg)    roll std {rp[:, 0].std():5.2f}  pitch std {rp[:, 1].std():5.2f}"
          f"   (mean roll {rp[:, 0].mean():+.2f}, pitch {rp[:, 1].mean():+.2f})")
    print(f"  filter error (deg) roll rms {np.sqrt((err[:, 0] ** 2).mean()):5.2f}"
          f"  pitch rms {np.sqrt((err[:, 1] ** 2).mean()):5.2f}"
          f"   mean roll {err[:, 0].mean():+.2f}, pitch {err[:, 1].mean():+.2f}"
          f"   p99 |roll| {np.percentile(np.abs(err[:, 0]), 99):.2f}, |pitch|"
          f" {np.percentile(np.abs(err[:, 1]), 99):.2f}")
    # Error as gravity-vector components, to compare with the training noise (U(-0.05, 0.05)).
    print(f"  as gravity components: rms error x {np.sqrt(np.mean(np.radians(err[:, 1]) ** 2)):.3f},"
          f" y {np.sqrt(np.mean(np.radians(err[:, 0]) ** 2)):.3f}"
          f"  (training noise rms {0.05 / np.sqrt(3):.3f}, independent each step)")
    # How much of the error is at the gait frequency rather than white noise.
    e0 = errs[0]
    if len(e0) > 64:
      f = np.fft.rfftfreq(len(e0), CTRL_DT)
      p = np.abs(np.fft.rfft(e0 - e0.mean(0), axis=0)) ** 2
      print(f"  error peak freq (Hz) roll {f[1 + p[1:, 0].argmax()]:.2f}, pitch {f[1 + p[1:, 1].argmax()]:.2f}"
            f"  (gait {GAIT_FREQ} Hz); share below 5 Hz roll"
            f" {p[(f > 0) & (f < 5), 0].sum() / p[f > 0, 0].sum():.0%},"
            f" pitch {p[(f > 0) & (f < 5), 1].sum() / p[f > 0, 1].sum():.0%}")

  if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(plots), 2, figsize=(14, 3.2 * len(plots)), squeeze=False)
    for row, (command, truth, est) in zip(axes, plots):
      t = np.arange(len(truth)) * CTRL_DT
      for ax, idx, name in ((row[0], 0, "roll"), (row[1], 1, "pitch")):
        ax.plot(t, angles(truth)[idx], label="true")
        ax.plot(t, angles(est)[idx], label="complementary filter")
        ax.set_title(f"command {command}: {name} (deg)")
        ax.grid(alpha=0.3)
      row[0].legend()
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=110, bbox_inches="tight")
    print(f"\nSaved plot {args.plot}")

  if args.video:
    import imageio_ffmpeg
    import mediapy
    args.video.parent.mkdir(parents=True, exist_ok=True)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(args.video, frames, fps=1 / CTRL_DT)
    print(f"Wrote {len(frames)} frames ({len(frames) * CTRL_DT:.1f} s) to {args.video}")


if __name__ == "__main__":
  main()

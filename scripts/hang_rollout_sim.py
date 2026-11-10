"""Run a policy in the sim with the torso hung in the air, and compare with the robot.

The robot can't walk freely yet, so it's tested hanging from a stand, with its
feet off the floor. This does the same in the sim: the training model
(scene.xml) with the torso's freejoint removed (in memory) and the torso fixed
at --height, so the feet never touch the floor. The policy is the same ONNX
file the robot runs (models/<model>/policy.onnx), and the observation is built
as Main builds it on the robot.

Record on the robot (on its stand), stop with Ctrl-C, then copy the .npz over:
  gnoci run --close-loop --model serial-pollen-weep --record logs/hang/real.npz

Sim only:
  uv run python scripts/hang_rollout_sim.py --model serial-pollen-weep --steps 500
  uv run python scripts/hang_rollout_sim.py --command 0.2 0 0 --video videos/hang.mp4
Compare with the robot (uses its model, length and joystick commands):
  uv run python scripts/hang_rollout_sim.py --real logs/hang/real.npz
Replay the robot's actions open loop, so only the joint dynamics are compared,
not the policy's feedback:
  uv run python scripts/hang_rollout_sim.py --real logs/hang/real.npz --replay

Closed loop, small differences grow, because each side feeds its own joint
readings back into the policy, so expect the curves to drift apart in phase.
--replay removes that: if the sim joints still don't follow the same targets
the way the real ones do, the actuator model (kp, damping, speed limit, delay)
is off. If replay matches but closed loop doesn't, look at the observation
(joint velocity scale, noise, IMU, delay).

Saves an .npz with the same keys as the robot's recording, and a plot.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from gnoci_v2.control.main import GAIT_FREQ
from gnoci_v2.control.predict import DEFAULT_MODEL, PolicyRunner
from measure_servo_response_sim import fixed_base_model

OUT_DIR = Path("logs/hang")
CTRL_DT = 0.02  # Main.control_freq = 50 Hz, the same as training.
SIM_DT = 0.002  # Training's sim_dt.
JOINTS = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]
# Slices of the 31-value observation.
GYRO, ACCEL, COMMAND = slice(0, 3), slice(3, 6), slice(6, 9)
JOINT_POS, JOINT_VEL, LAST_ACT, PHASE = slice(9, 15), slice(15, 21), slice(21, 27), slice(27, 31)


def rollout(model, policy, steps, commands, action_delay=0.0, real_imu=None, actions=None,
            frame_fn=None):
  """Runs the policy (or replays actions) for steps control steps.

  commands: (steps, 3). real_imu: optional (steps, 6) gyro + accelerometer to put in
  the observation instead of the sim's. actions: optional (steps, 6) actions
  to apply instead of the policy's. action_delay: seconds between reading the
  observation and the target taking effect. frame_fn(data) is called once per
  control step, after the physics.
  """
  model.opt.timestep = SIM_DT
  n_sub = int(round(CTRL_DT / SIM_DT))
  delay = int(round(action_delay / SIM_DT))
  data = mujoco.MjData(model)
  data.qpos[:] = 0.0  # Home: every joint at 0, as the robot is after init_servos.
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)

  qpos_adr = [model.jnt_qposadr[model.joint(j).id] for j in JOINTS]
  dof_adr = [model.jnt_dofadr[model.joint(j).id] for j in JOINTS]
  act_ids = [model.actuator(j).id for j in JOINTS]
  imu = model.site("imu").id
  gyro_adr = model.sensor_adr[model.sensor("gyro").id]

  phase = np.array([0.0, np.pi])
  phase_dt = 2 * np.pi * GAIT_FREQ * CTRL_DT
  last_action = np.zeros(6)
  # (substep it takes effect, target), oldest first.
  pending = []
  current = np.zeros(6)
  out = {"t": [], "obs": [], "action": [], "target": []}

  for k in range(steps):
    if real_imu is not None:
      gyro, accel = real_imu[k, :3], real_imu[k, 3:]
    else:
      gyro = data.sensordata[gyro_adr:gyro_adr + 3].copy()
      # MuJoCo's accelerometer reads 0 on a body fixed to the world, so use what
      # a still torso measures: the reaction to gravity, in g.
      accel = data.site_xmat[imu].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    obs = np.concatenate([
        gyro, accel, commands[k],
        data.qpos[qpos_adr], data.qvel[dof_adr],
        last_action,
        np.cos(phase), np.sin(phase),
    ])
    action = actions[k] if actions is not None else policy.predict(obs)
    # As in training: no clamping, the joint limits stop the joints.
    target = policy.action_scale * np.asarray(action, dtype=float)
    last_action = np.asarray(action, dtype=float)
    phase = np.fmod(phase + phase_dt + np.pi, 2 * np.pi) - np.pi

    pending.append((k * n_sub + delay, target))
    for j in range(n_sub):
      s = k * n_sub + j
      while pending and pending[0][0] <= s:
        current = pending.pop(0)[1]
      data.ctrl[act_ids] = current
      mujoco.mj_step(model, data)
    if frame_fn is not None:
      frame_fn(data)

    out["t"].append(k * CTRL_DT)
    out["obs"].append(obs)
    out["action"].append(last_action)
    out["target"].append(target)
  return {key: np.array(value) for key, value in out.items()}


def dominant_freq(x, dt):
  """The strongest non-zero frequency (Hz) of each column of x."""
  x = x - x.mean(0)
  power = np.abs(np.fft.rfft(x, axis=0)) ** 2
  freqs = np.fft.rfftfreq(len(x), d=dt)
  return freqs[1 + power[1:].argmax(0)]


def summarise(name, run, skip):
  """Per-joint stats of position and action, after the first skip steps."""
  pos = run["obs"][skip:, JOINT_POS]
  vel = run["obs"][skip:, JOINT_VEL]
  act = run["action"][skip:]
  rows = {
      "pos mean (rad)": pos.mean(0),
      "pos p2p (rad)": np.ptp(pos, 0),
      "vel rms (rad/s)": np.sqrt((vel ** 2).mean(0)),
      "action mean": act.mean(0),
      "action std": act.std(0),
      "action d/step rms": np.sqrt((np.diff(act, axis=0) ** 2).mean(0)),
      "pos peak freq (Hz)": dominant_freq(pos, CTRL_DT),
  }
  print(f"\n{name}:")
  print(" " * 20 + " ".join(f"{j[:11]:>11}" for j in JOINTS))
  for label, values in rows.items():
    print(f"{label:<20}" + " ".join(f"{v:11.3f}" for v in values))


def plot(sim, real, path, title):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
  for i, joint in enumerate(JOINTS):
    ax_p, ax_a = axes[i]
    ax_p.plot(sim["t"], sim["obs"][:, JOINT_POS][:, i], color="C0", label="sim position")
    ax_p.plot(sim["t"], sim["target"][:, i], color="C0", ls="--", lw=0.8, label="sim target")
    ax_a.plot(sim["t"], sim["obs"][:, JOINT_VEL][:, i], color="C0", label="sim")
    if real is not None:
      ax_p.plot(real["t"], real["obs"][:, JOINT_POS][:, i], color="C1", label="real position")
      ax_p.plot(real["t"], real["target"][:, i], color="C1", ls="--", lw=0.8, label="real target")
      ax_a.plot(real["t"], real["obs"][:, JOINT_VEL][:, i], color="C1", label="real")
    ax_p.set_ylabel(f"{joint}\n(rad)")
    ax_a.set_ylabel("velocity (rad/s)")
    for ax in (ax_p, ax_a):
      ax.grid(alpha=0.3)
  axes[0, 0].legend(fontsize=8, ncol=2)
  axes[0, 1].legend(fontsize=8)
  axes[0, 0].set_title("joint position (solid) and target (dashed)")
  axes[0, 1].set_title("joint velocity (as observed)")
  axes[-1, 0].set_xlabel("time (s)")
  axes[-1, 1].set_xlabel("time (s)")
  fig.suptitle(title)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=110, bbox_inches="tight")
  print(f"Saved plot {path}")


def main() -> None:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--real", type=Path, default=None,
                      help="A recording from `gnoci run --record`. Its model, length and"
                           " joystick commands are used, and the two are compared.")
  parser.add_argument("--replay", action="store_true",
                      help="With --real: apply the robot's recorded actions instead of running"
                           " the policy.")
  parser.add_argument("--real-imu", action="store_true",
                      help="With --real: put the robot's recorded gyro and accelerometer in the"
                           " observation, e.g. if it hangs tilted or swings on the stand.")
  parser.add_argument("--model", default=None, help=f"Default: the recording's, or {DEFAULT_MODEL}.")
  parser.add_argument("--steps", type=int, default=500, help="Control steps (ignored with --real).")
  parser.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                      metavar=("VX", "VY", "YAW"), help="Fixed command (ignored with --real).")
  parser.add_argument("--action-delay", type=float, default=0.0,
                      help="Seconds from observation to the target taking effect. Training"
                           " sampled 0 to 0.03.")
  parser.add_argument("--height", type=float, default=0.5,
                      help="Torso height (m). Home is 0.2723 with the feet on the floor.")
  parser.add_argument("--skip", type=int, default=25,
                      help="Steps left out of the summary stats, while the policy starts up.")
  parser.add_argument("--out", type=Path, default=None, help="Output .npz (the plot is next to it).")
  parser.add_argument("--video", type=Path, default=None, help="Also render a video (.mp4).")
  parser.add_argument("--camera", default="track")
  args = parser.parse_args()

  real = None
  if args.real:
    real = dict(np.load(args.real))
    steps = len(real["t"])
    commands = real["obs"][:, COMMAND]
    model_name = args.model or str(real["model"])
    print(f"Real recording {args.real}: model {real['model']}, {steps} steps"
          f" ({steps * CTRL_DT:.1f} s), close_loop {bool(real['close_loop'])},"
          f" median step interval {1000 * np.median(np.diff(real['t'])):.1f} ms")
    if not bool(real["close_loop"]):
      print("  WARNING: recorded without --close-loop, so the robot's joints didn't follow"
            " its targets.")
  else:
    if args.replay or args.real_imu:
      parser.error("--replay and --real-imu need --real")
    steps = args.steps
    commands = np.tile(np.array(args.command), (steps, 1))
    model_name = args.model or DEFAULT_MODEL

  policy = PolicyRunner(model_name)
  if real is not None and not np.isclose(policy.action_scale, float(real["action_scale"])):
    print(f"  WARNING: the recording used action_scale {float(real['action_scale'])},"
          f" the model's config.json says {policy.action_scale}")

  model = fixed_base_model(args.height)
  frames, renderer = [], None
  frame_fn = None
  if args.video:
    renderer = mujoco.Renderer(model, height=480, width=640)

    def frame_fn(data):
      renderer.update_scene(data, camera=args.camera)
      frames.append(renderer.render())

  sim = rollout(
      model, policy, steps, commands,
      action_delay=args.action_delay,
      real_imu=np.concatenate([real["obs"][:, GYRO], real["obs"][:, ACCEL]], axis=1)
      if args.real_imu else None,
      actions=real["action"] if args.replay else None,
      frame_fn=frame_fn,
  )

  mode = "replay of the robot's actions" if args.replay else "policy closed loop"
  print(f"Sim: {model_name}, torso fixed at {args.height} m, {mode},"
        f" action delay {1000 * args.action_delay:g} ms")
  summarise("sim", sim, args.skip)
  if real is not None:
    summarise("real", real, args.skip)
    pos_err = sim["obs"][args.skip:, JOINT_POS] - real["obs"][args.skip:, JOINT_POS]
    print("\nsim - real position RMS (rad):"
          + " ".join(f"{v:11.3f}" for v in np.sqrt((pos_err ** 2).mean(0))))

  default_name = f"sim_{model_name}" + ("_replay" if args.replay else "")
  out = args.out or OUT_DIR / f"{default_name}.npz"
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **sim, model=model_name, action_scale=policy.action_scale,
           control_freq=round(1 / CTRL_DT), gait_freq=GAIT_FREQ, close_loop=True,
           action_delay=args.action_delay, replay=args.replay)
  print(f"Saved {out}")
  plot(sim, real, out.with_suffix(".png"),
       f"{model_name}, hung at {args.height} m, {mode}, delay {1000 * args.action_delay:g} ms")

  if args.video:
    import imageio_ffmpeg
    import mediapy
    args.video.parent.mkdir(parents=True, exist_ok=True)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(args.video, frames, fps=1 / CTRL_DT)
    print(f"Wrote {len(frames)} frames to {args.video}")


if __name__ == "__main__":
  main()

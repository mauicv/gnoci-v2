"""Measure how a trained policy copes with action and IMU latency in the sim.

Runs the deterministic policy in many envs for each combination of action
delay and IMU delay, with the same random seed for every combination, so
differences come from the latency alone. Prints a table and saves the results
as JSON.

  uv run python scripts/eval_latency.py
  uv run python scripts/eval_latency.py --action-delays-ms 0 10 20 30 40 --imu-delays-ms 0 10
  uv run python scripts/eval_latency.py --command 0 0 0 --video-dir videos/latency

Delays are in milliseconds and must be multiples of sim_dt (2 ms). An action
delay of D ms means an action takes effect D ms after the observation it was
computed from; an IMU delay of E ms means the gyro and gravity in the
observation are E ms old.

The env's observation reports the action from one control step back, as the
robot's Main does. Checkpoints trained before that fix saw the action from two
steps back, so they're evaluated here (and on the robot) slightly out of
distribution.
"""

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from gnoci_v2.training import joystick
from gnoci_v2.training.convert import load_checkpoint, make_deterministic_policy

CHECKPOINTS = Path("checkpoints")
COMMAND = slice(6, 9)


def latest_checkpoint():
  paths = sorted(p for p in CHECKPOINTS.iterdir() if p.is_dir())
  if not paths:
    raise FileNotFoundError(f"No checkpoints in {CHECKPOINTS}")
  return paths[-1]


def ms_to_substeps(ms, sim_dt):
  substeps = ms / 1000 / sim_dt
  if abs(substeps - round(substeps)) > 1e-6:
    raise ValueError(f"{ms} ms isn't a multiple of sim_dt ({sim_dt * 1000:g} ms)")
  return int(round(substeps))


def make_rollout(env, policy, num_envs, num_steps, fixed_command):
  """Returns a jitted fn (rng, action_delay, imu_delay) -> results."""
  reset = jax.vmap(env.reset)
  step = jax.vmap(env.step)
  dt = env.dt

  def set_command(state):
    if fixed_command is None:
      return state
    command = jp.broadcast_to(jp.array(fixed_command), (num_envs, 3))
    state.info["command"] = command
    return state

  def rollout(rng, action_delay, imu_delay):
    state = reset(jax.random.split(rng, num_envs))
    state.info["action_delay"] = jp.full(num_envs, action_delay, jp.int32)
    state.info["imu_delay"] = jp.full(num_envs, imu_delay, jp.int32)
    state = set_command(state)
    if fixed_command is not None:
      # The reset obs was built with the sampled command.
      obs = state.obs["state"].at[:, COMMAND].set(state.info["command"])
      state = state.replace(obs={**state.obs, "state": obs})

    def body(carry, _):
      state, prev_action, alive, sums = carry
      action = policy(state.obs["state"])

      command = state.info["command"]
      linvel = jax.vmap(env.get_local_linvel)(state.data)[:, :2]
      yaw_rate = jax.vmap(env.get_gyro)(state.data)[:, 2]
      lin_err = jp.linalg.norm(linvel - command[:, :2], axis=-1)
      yaw_err = jp.abs(yaw_rate - command[:, 2])
      joint_vel = state.data.qvel[:, 6:]
      a = alive.astype(jp.float32)
      sums = {
          "alive_steps": sums["alive_steps"] + a,
          "lin_vel_err": sums["lin_vel_err"] + a * lin_err,
          "yaw_rate_err": sums["yaw_rate_err"] + a * yaw_err,
          "action_abs": sums["action_abs"] + a * jp.abs(action).mean(-1),
          "action_rate": sums["action_rate"] + a * jp.abs(action - prev_action).mean(-1),
          "joint_vel_rms": sums["joint_vel_rms"] + a * jp.sqrt((joint_vel**2).mean(-1)),
      }

      state = step(set_command(state), action)
      alive = alive & (state.done == 0)
      return (state, action, alive, sums), (state.data.qpos[0], alive[0])

    zeros = jp.zeros(num_envs)
    sums = {k: zeros for k in (
        "alive_steps", "lin_vel_err", "yaw_rate_err",
        "action_abs", "action_rate", "joint_vel_rms",
    )}
    carry = (state, jp.zeros((num_envs, env.action_size)), jp.ones(num_envs, bool), sums)
    (_, _, alive, sums), (qpos0, alive0) = jax.lax.scan(body, carry, None, num_steps)

    steps = jp.maximum(sums["alive_steps"].sum(), 1.0)
    results = {
        "fall_rate": 1.0 - alive.mean(),
        "mean_survival_s": sums["alive_steps"].mean() * dt,
    }
    for key in ("lin_vel_err", "yaw_rate_err", "action_abs", "action_rate", "joint_vel_rms"):
      results[key] = sums[key].sum() / steps
    return results, qpos0, alive0

  return jax.jit(rollout)


def save_video(env, qpos, alive, path):
  import mediapy
  import imageio_ffmpeg
  import mujoco

  model = env.mj_model
  data = mujoco.MjData(model)
  # Stop a second after the first env falls, so the fall is visible.
  end = len(alive)
  if not alive.all():
    end = min(len(alive), int(np.argmin(alive)) + int(1 / env.dt))
  frames = []
  with mujoco.Renderer(model, height=480, width=640) as renderer:
    for q in qpos[:end]:
      data.qpos[:] = q
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera="track")
      frames.append(renderer.render())
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  mediapy.write_video(str(path), frames, fps=1 / env.dt)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--checkpoint", type=Path, default=None,
                      help="Checkpoint directory. Default: the latest in checkpoints/.")
  parser.add_argument("--action-delays-ms", type=float, nargs="+", default=[0, 10, 20, 30, 40])
  parser.add_argument("--imu-delays-ms", type=float, nargs="+", default=[0])
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=1000, help="Control steps per env (50 per second).")
  parser.add_argument("--command", type=float, nargs=3, default=None, metavar=("VX", "VY", "YAW"),
                      help="Fix the command. Default: the env's random commands.")
  parser.add_argument("--no-push", action="store_true", help="Disable the random pushes.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=Path, default=None, help="JSON results path.")
  parser.add_argument("--video-dir", type=Path, default=None,
                      help="Also save a video of env 0 for each combination.")
  args = parser.parse_args()

  checkpoint = (args.checkpoint or latest_checkpoint()).resolve()
  policy = make_deterministic_policy(*load_checkpoint(checkpoint))

  config = joystick.default_config()
  sim_dt = config.sim_dt
  action_delays = [ms_to_substeps(ms, sim_dt) for ms in args.action_delays_ms]
  imu_delays = [ms_to_substeps(ms, sim_dt) for ms in args.imu_delays_ms]
  # Size the env's buffers for the largest delay; each run then sets its own.
  config.latency_config.action_delay_range = [0, max(action_delays)]
  config.latency_config.imu_delay_range = [0, max(imu_delays)]
  if args.no_push:
    config.push_config.enable = False
  env = joystick.GnociV2Joystick(config=config)

  rollout = make_rollout(env, policy, args.num_envs, args.steps, args.command)
  rng = jax.random.PRNGKey(args.seed)

  print(f"Checkpoint: {checkpoint}")
  print(f"{args.num_envs} envs x {args.steps} steps ({args.steps * env.dt:.0f} s),"
        f" command {'random' if args.command is None else args.command},"
        f" pushes {'off' if args.no_push else 'on'}")
  header = (f"{'action ms':>9} {'imu ms':>6} | {'fall %':>6} {'survive s':>9}"
            f" {'vel err':>7} {'yaw err':>7} | {'|a|':>5} {'|da|':>5} {'qvel rms':>8}")
  print(header)
  print("-" * len(header))

  if args.video_dir:
    args.video_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for d, e in itertools.product(action_delays, imu_delays):
    results, qpos0, alive0 = rollout(rng, d, e)
    results = {k: float(v) for k, v in results.items()}
    row = {
        "action_delay_ms": d * sim_dt * 1000,
        "imu_delay_ms": e * sim_dt * 1000,
        **results,
    }
    rows.append(row)
    print(f"{row['action_delay_ms']:>9g} {row['imu_delay_ms']:>6g} |"
          f" {100 * results['fall_rate']:>6.1f} {results['mean_survival_s']:>9.1f}"
          f" {results['lin_vel_err']:>7.3f} {results['yaw_rate_err']:>7.3f} |"
          f" {results['action_abs']:>5.2f} {results['action_rate']:>5.2f}"
          f" {results['joint_vel_rms']:>8.2f}")
    if args.video_dir:
      path = args.video_dir / f"action{row['action_delay_ms']:g}ms_imu{row['imu_delay_ms']:g}ms.mp4"
      save_video(env, np.asarray(qpos0), np.asarray(alive0), path)

  out = args.out or Path("logs/latency") / f"{checkpoint.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps({
      "checkpoint": str(checkpoint),
      "num_envs": args.num_envs,
      "steps": args.steps,
      "command": args.command,
      "pushes": not args.no_push,
      "seed": args.seed,
      "results": rows,
  }, indent=2) + "\n")
  print(f"\nvel err: m/s, yaw err: rad/s, |a|: mean |action|, |da|: mean |action change| per step,"
        f" qvel rms: rad/s. All averaged over steps before each env's first fall.")
  print(f"Saved {out}")


if __name__ == "__main__":
  main()

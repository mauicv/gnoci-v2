"""Measure how smooth a trained policy's actions are in the sim.

Runs the deterministic policy in many envs for each fixed command and reports,
per joint:

  - how fast the motor target moves (|delta target| / ctrl_dt, in rad/s),
  - how fast the simulated joint actually moves (qvel, in rad/s),
  - the share of steps where either goes over --speed-limit,
  - the share of action power above --hf-cutoff Hz (jitter; the gait clock is
    about 1.4 Hz),
  - touchdowns per second per foot, and how many envs fell.

Stats only use the steps before an env falls. There's no domain randomization.

  uv run python scripts/action_stats.py
  uv run python scripts/action_stats.py --checkpoint checkpoints/state-unfold-marry
  uv run python scripts/action_stats.py --command 0.2 0 0 --command 0 0 0.5
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from gnoci_v2.training import joystick
from gnoci_v2.control.predict import load_model_config
from gnoci_v2.training.convert import (
    checkpoint_action_scale,
    load_checkpoint,
    make_deterministic_policy,
)

CHECKPOINTS = Path("checkpoints")
COMMAND = slice(6, 9)
DEFAULT_COMMANDS = ([0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.15, 0.0])
# Skip the first steps, while the policy settles from the reset pose.
WARMUP_STEPS = 50


def latest_checkpoint():
  paths = sorted(p for p in CHECKPOINTS.iterdir() if p.is_dir())
  if not paths:
    raise FileNotFoundError(f"No checkpoints in {CHECKPOINTS}")
  return paths[-1]


def trained_action_scale(checkpoint, override=None):
  """The action_scale the policy was trained with, and where it came from.

  A policy run with the wrong scale moves its joints too far or not far
  enough, so the stats would be meaningless.
  """
  if override is not None:
    return override, "--action-scale"
  scale = checkpoint_action_scale(checkpoint)
  if scale is not None:
    return scale, "checkpoint env_config.json"
  try:
    return float(load_model_config(checkpoint.name)["action_scale"]), "models/ config.json"
  except FileNotFoundError:
    default = joystick.default_config().action_scale
    return default, "WARNING: not recorded, using the env default. Pass --action-scale if it differs"


def make_rollout(env, policy, num_envs, num_steps):
  """Returns a jitted fn (rng, command) -> per-step arrays, each (T, N, ...)."""
  reset = jax.vmap(env.reset)
  step = jax.vmap(env.step)

  def set_command(state, command):
    state.info["command"] = jp.broadcast_to(command, (num_envs, 3))
    # Keep the step counter below 500, so the env never resamples the command.
    state.info["step"] = jp.zeros_like(state.info["step"])
    return state

  def rollout(rng, command):
    state = set_command(reset(jax.random.split(rng, num_envs)), command)
    # The reset obs was built with the sampled command.
    obs = state.obs["state"].at[:, COMMAND].set(state.info["command"])
    state = state.replace(obs={**state.obs, "state": obs})

    def body(carry, _):
      state, alive = carry
      action = policy(state.obs["state"])
      state = set_command(step(state, action), command)
      out = {
          "action": action,
          "target": state.info["motor_targets"],
          "joint_vel": state.data.qvel[:, 6:],
          "contact": state.info["last_contact"],
          "alive": alive,
      }
      alive = alive & (state.done == 0)
      return (state, alive), out

    carry = (state, jp.ones(num_envs, bool))
    _, out = jax.lax.scan(body, carry, None, num_steps)
    return out

  return jax.jit(rollout)


def percent(x):
  return " ".join(f"{v:6.1%}" for v in x)


def numbers(x, fmt="6.2f"):
  return " ".join(f"{v:{fmt}}" for v in x)


def report(env, out, speed_limit, hf_cutoff):
  dt = env.dt
  out = {k: np.asarray(v)[WARMUP_STEPS:] for k, v in out.items()}
  alive = out["alive"]  # (T, N)
  num_envs = alive.shape[1]
  fell = ~alive[-1]
  print(f"  envs fallen: {fell.sum()}/{num_envs}")

  # Per-step quantities, masked to steps where the env was still up.
  both_alive = alive[1:] & alive[:-1]
  target_rate = np.abs(np.diff(out["target"], axis=0)) / dt  # (T-1, N, 6)
  target_rate = target_rate[both_alive]  # (M, 6)
  joint_vel = np.abs(out["joint_vel"][alive])  # (M, 6)

  names = [env.mj_model.joint(i + 1).name for i in range(env.action_size)]
  print("  " + " " * 26 + " ".join(f"{n[:6]:>6}" for n in names)
        + "   (" + ", ".join(names) + ")")
  print(f"  target speed p50 (rad/s)  {numbers(np.percentile(target_rate, 50, axis=0))}")
  print(f"  target speed p95 (rad/s)  {numbers(np.percentile(target_rate, 95, axis=0))}")
  print(f"  target speed max (rad/s)  {numbers(target_rate.max(axis=0))}")
  print(f"  target > {speed_limit:g} rad/s       {percent((target_rate > speed_limit).mean(axis=0))}")
  print(f"  joint speed p95 (rad/s)   {numbers(np.percentile(joint_vel, 95, axis=0))}")
  print(f"  joint speed max (rad/s)   {numbers(joint_vel.max(axis=0))}")
  print(f"  joint > {speed_limit:g} rad/s        {percent((joint_vel > speed_limit).mean(axis=0))}")

  # Spectra only use envs that stayed up for the whole rollout.
  upright = ~fell
  if upright.any():
    action = out["action"][:, upright]  # (T, N', 6)
    freqs = np.fft.rfftfreq(action.shape[0], d=dt)
    power = np.abs(np.fft.rfft(action - action.mean(0), axis=0)) ** 2
    power = power.mean(axis=1)  # (F, 6)
    hf = power[freqs > hf_cutoff].sum(0) / power[freqs > 0].sum(0)
    print(f"  action power > {hf_cutoff:g} Hz      {percent(hf)}")
    print(f"  peak action freq (Hz)     {numbers(freqs[1 + power[1:].argmax(0)])}")
    contact = out["contact"][:, upright].astype(int)  # (T, N', 2)
    touchdowns = (np.diff(contact, axis=0) == 1).sum(0).mean(0)
    print(f"  touchdowns/s per foot     {numbers(touchdowns / (contact.shape[0] * dt))}")
  else:
    print("  every env fell, so no spectrum")


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--checkpoint", type=Path, default=None,
                      help="Checkpoint dir (default: the latest in checkpoints/)")
  parser.add_argument("--command", type=float, nargs=3, action="append",
                      metavar=("VX", "VY", "YAW"),
                      help="Fixed command; repeat for several (default: stand, forward, sideways)")
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=500, help="Control steps per rollout")
  parser.add_argument("--speed-limit", type=float, default=5.24,
                      help="rad/s; Open Duck's max_motor_velocity for the STS3215")
  parser.add_argument("--hf-cutoff", type=float, default=5.0, help="Hz")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--action-scale", type=float, default=None,
                      help="Default: the checkpoint's env_config.json, else"
                           " models/<checkpoint name>/config.json, else the env default")
  args = parser.parse_args()

  checkpoint = (args.checkpoint or latest_checkpoint()).resolve()
  print(f"Checkpoint: {checkpoint}")
  action_scale, source = trained_action_scale(checkpoint, args.action_scale)
  print(f"action_scale {action_scale} ({source})")
  config = joystick.default_config()
  config.action_scale = action_scale
  env = joystick.GnociV2Joystick(config=config)
  normalizer, params = load_checkpoint(checkpoint)
  policy = jax.vmap(make_deterministic_policy(normalizer, params))
  rollout = make_rollout(env, policy, args.num_envs, args.steps)

  for command in args.command or DEFAULT_COMMANDS:
    print(f"\nCommand vx={command[0]:g} vy={command[1]:g} yaw={command[2]:g}:")
    out = rollout(jax.random.PRNGKey(args.seed), jp.array(command))
    report(env, out, args.speed_limit, args.hf_cutoff)


if __name__ == "__main__":
  main()

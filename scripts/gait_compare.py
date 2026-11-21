"""Compare the gaits of several checkpoints in the sim: foot height and torso sway.

Runs each checkpoint's deterministic policy on the current model with a fixed
command, no pushes and no domain randomization, and writes two figures:

  - <out>_feet.png: sole centre height of both feet over a few gait cycles for
    one env, with the floor contacts drawn as squares below 0 (as
    foot_rollout.png), plus the stance mask of the feet_contact reward when the
    checkpoint was trained with it.
  - <out>_sway.png: torso roll and pitch over time for one env, and the
    amplitude spectrum of each averaged over all envs.

and prints, per checkpoint, the roll and pitch spread (RMS about the mean and
p5-p95 range), their mean, the dominant frequency, stance share per foot, the
share of time both feet are down / both up, and how many envs fell. Stats skip
the first --warmup seconds and only use envs that stayed up.

  uv run python scripts/gait_compare.py
  uv run python scripts/gait_compare.py --checkpoint slain-butter-mate saint-oil-fast --command 0.15 0 0
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mujoco_playground._src import gait

from gnoci_v2.training import joystick
from gnoci_v2.training.constants import ENV_CONFIG_FILE
from gnoci_v2.training.convert import load_checkpoint, make_deterministic_policy

CHECKPOINTS = Path("checkpoints")
DEFAULT = ["saint-oil-fast", "giddy-purple-honor", "slain-butter-mate"]
COMMAND = slice(6, 9)


def roll_pitch(quat):
  """(…, 4) wxyz torso quaternion -> roll, pitch in degrees (x forward, y left)."""
  w, x, y, z = (quat[..., i] for i in range(4))
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
  return np.degrees(roll), np.degrees(pitch)


def make_rollout(env, policy, num_envs, num_steps):
  reset = jax.vmap(env.reset)
  step = jax.vmap(env.step)

  def set_command(state, command):
    state.info["command"] = jp.broadcast_to(command, (num_envs, 3))
    state.info["step"] = jp.zeros_like(state.info["step"])
    return state

  def rollout(rng, command):
    state = set_command(reset(jax.random.split(rng, num_envs)), command)
    obs = state.obs["state"].at[:, COMMAND].set(state.info["command"])
    state = state.replace(obs={**state.obs, "state": obs})

    def body(carry, _):
      state, alive = carry
      state = set_command(step(state, policy(state.obs["state"])), command)
      out = {
          "foot_z": state.data.site_xpos[:, env._feet_site_id, 2],
          "contact": state.info["last_contact"],
          "quat": state.data.qpos[:, 3:7],
          "linvel": state.data.qvel[:, :3],
          "phase": state.info["phase"],
          "alive": alive,
      }
      return (state, alive & (state.done == 0)), out

    _, out = jax.lax.scan(body, (state, jp.ones(num_envs, bool)), None, num_steps)
    return out

  return jax.jit(rollout)


def describe(cfg):
  rc = cfg.get("reward_config", {})
  s = rc.get("scales", {})
  if s.get("feet_contact", 0):
    return (f"feet_contact (stance {rc.get('stance_fraction', 0.5):g}), "
            f"feet_height {s.get('feet_height', 0):g}, lift {rc['max_foot_height']*1e3:g} mm")
  frac = rc.get("swing_fraction", 1.0)
  return f"feet_phase, swing_fraction {frac:g}, lift {rc['max_foot_height']*1e3:g} mm"


def target_height(cfg, phase):
  """The feet_phase target the checkpoint was trained with, or None."""
  rc = cfg.get("reward_config", {})
  if not rc.get("scales", {}).get("feet_phase", 0):
    return None
  h = rc["max_foot_height"]
  frac = rc.get("swing_fraction", 1.0)
  if frac == 1.0:
    return np.asarray(gait.get_rz(phase, swing_height=h))
  x = (phase + np.pi) / (2 * np.pi)
  s = np.clip(x / frac, 0, 1)
  bez = lambda a, b, t: a + (b - a) * (t**3 + 3 * t**2 * (1 - t))
  return np.where(x >= frac, 0.0, np.where(s <= 0.5, bez(0, h, 2 * s), bez(h, 0, 2 * s - 1)))


def stance_mask(cfg, phase):
  rc = cfg.get("reward_config", {})
  if not rc.get("scales", {}).get("feet_contact", 0):
    return None
  x = (phase + np.pi) / (2 * np.pi)
  return np.abs(x - 0.5) >= (1 - rc.get("stance_fraction", 0.5)) / 2


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--checkpoint", nargs="+", default=DEFAULT)
  p.add_argument("--command", type=float, nargs=3, default=[0.15, 0.0, 0.0])
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--warmup", type=float, default=2.0)
  p.add_argument("--window", type=float, nargs=2, default=[3.0, 6.0],
                 help="Time window (s) of the single-env plots")
  p.add_argument("--env", type=int, default=0, help="Env shown in the single-env plots")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default="videos/gait_compare")
  args = p.parse_args()

  config = joystick.default_config()
  config.push_config.enable = False
  results = {}
  for name in args.checkpoint:
    path = (CHECKPOINTS / name).resolve()
    cfg = json.loads((path / ENV_CONFIG_FILE).read_text())
    config.action_scale = float(cfg["action_scale"])
    env = joystick.GnociV2Joystick(config=config)
    policy = jax.vmap(make_deterministic_policy(*load_checkpoint(path)))
    steps = int(round(args.seconds / env.dt))
    out = make_rollout(env, policy, args.num_envs, steps)(
        jax.random.PRNGKey(args.seed), jp.array(args.command))
    out = {k: np.asarray(v) for k, v in out.items()}
    out["roll"], out["pitch"] = roll_pitch(out["quat"])
    q = out["quat"]
    out["yaw"] = np.degrees(np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
                                       1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2)))
    results[name] = (cfg, out, env.dt)

  cmd = " ".join(f"{c:g}" for c in args.command)
  # Stats.
  print(f"command {cmd}, {args.num_envs} envs, no pushes, no randomization; "
        f"stats over t > {args.warmup:g} s\n")
  hdr = (f"{'checkpoint':20} {'fell':>5} {'roll rms':>8} {'p5-p95':>7} {'mean':>6} {'f':>5}"
         f" {'pitch rms':>9} {'p5-p95':>7} {'mean':>6} {'f':>5} {'stance L/R':>11} {'both dn':>7} {'both up':>7}")
  print(hdr + "\n" + "-" * len(hdr))
  spectra = {}
  for name, (cfg, out, dt) in results.items():
    w = int(args.warmup / dt)
    up = out["alive"][-1]
    row = f"{name:20} {np.sum(~up):>2}/{len(up):<2}"
    spectra[name] = {}
    for key in ("roll", "pitch"):
      a = out[key][w:, up]
      dev = a - a.mean(0)
      freqs = np.fft.rfftfreq(a.shape[0], dt)
      amp = 2 * np.abs(np.fft.rfft(dev, axis=0)).mean(1) / a.shape[0]
      spectra[name][key] = (freqs, amp)
      row += (f" {np.sqrt((dev**2).mean()):8.2f} {np.mean(np.percentile(a, 95, 0) - np.percentile(a, 5, 0)):7.2f}"
              f" {a.mean():6.2f} {freqs[1 + amp[1:].argmax()]:5.2f}")
    c = out["contact"][w:, up]
    stance = c.mean((0, 1))
    row += (f" {stance[0]:5.0%}/{stance[1]:<5.0%} {np.all(c, -1).mean():7.0%}"
            f" {np.all(~c, -1).mean():7.0%}")
    print(row)
  print(f"\n{'checkpoint':20} {'side vel rms':>12} {'side amp mm':>11} {'yaw rate rms':>12} {'fwd speed':>9}")
  for name, (cfg, out, dt) in results.items():
    w = int(args.warmup / dt)
    up = out["alive"][-1]
    yaw = np.radians(out["yaw"][w:, up])
    v = out["linvel"][w:, up]
    fwd = np.cos(yaw) * v[..., 0] + np.sin(yaw) * v[..., 1]
    side = -np.sin(yaw) * v[..., 0] + np.cos(yaw) * v[..., 1]
    side = side - side.mean(0)
    # Sideways sway amplitude at the stride frequency: the sideways velocity
    # component there, integrated (A_pos = A_vel / 2 pi f).
    freqs = np.fft.rfftfreq(side.shape[0], dt)
    amp = 2 * np.abs(np.fft.rfft(side, axis=0)).mean(1) / side.shape[0]
    k = 1 + amp[1:].argmax()
    yaw_rate = np.diff(np.unwrap(yaw, axis=0), axis=0) / dt
    print(f"{name:20} {np.sqrt((side**2).mean()):12.3f} {amp[k] / (2 * np.pi * freqs[k]) * 1e3:8.1f} @{freqs[k]:.1f}"
          f" {np.degrees(np.sqrt((yaw_rate**2).mean())):12.1f} {fwd.mean():9.3f}")
  print("side vel in m/s (body heading frame), side amp = sideways position swing amplitude"
        " at its dominant frequency (Hz), yaw rate in deg/s, fwd speed in m/s")
  print("\nroll/pitch in degrees (rms about each env's mean; p5-p95 range per env, averaged),"
        " f = dominant frequency in Hz")

  colors = {"left": "#2a6fdb", "right": "#e8622c"}
  t0, t1 = args.window

  # Feet.
  n = len(results)
  fig, axes = plt.subplots(n, 1, figsize=(13, 3.6 * n), sharex=True, squeeze=False)
  for ax, (name, (cfg, out, dt)) in zip(axes[:, 0], results.items()):
    t = (np.arange(out["foot_z"].shape[0]) + 1) * dt
    sel = (t >= t0) & (t <= t1)
    e = args.env
    for i, side in enumerate(("left", "right")):
      col = colors[side]
      tgt = target_height(cfg, out["phase"][sel, e, i])
      if tgt is not None:
        ax.plot(t[sel], tgt * 1e3, "--", color=col, lw=1.2, label=f"{side} target")
      ax.plot(t[sel], out["foot_z"][sel, e, i] * 1e3, color=col, lw=2, label=f"{side} foot")
      y = -6 - 4 * i
      c = out["contact"][sel, e, i]
      ax.plot(t[sel][c], np.full(c.sum(), y), "s", color=col, ms=2.5)
      mask = stance_mask(cfg, out["phase"][sel, e, i])
      if mask is not None:
        ax.plot(t[sel][mask], np.full(mask.sum(), y - 1.6), "s", color=col, ms=2.5, alpha=0.3)
    title = f"{name}: {describe(cfg)}, command {cmd}  (squares below 0: foot on the floor"
    title += "; faint: meant to be on the floor)" if stance_mask(cfg, out["phase"][:1, 0, 0]) is not None else ")"
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylabel("sole centre height (mm)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
  axes[-1, 0].set_xlabel("time (s)")
  fig.tight_layout()
  feet_path = f"{args.out}_feet.png"
  Path(feet_path).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(feet_path, dpi=110)

  # Sway.
  palette = ["#2a6fdb", "#e8622c", "#2ca02c", "#9467bd", "#8c564b"]
  fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [2, 1]})
  for (name, (cfg, out, dt)), col in zip(results.items(), palette):
    t = (np.arange(out["roll"].shape[0]) + 1) * dt
    sel = (t >= t0) & (t <= t1)
    for r, key in enumerate(("roll", "pitch")):
      axes[r, 0].plot(t[sel], out[key][sel, args.env], color=col, lw=1.8, label=name)
      freqs, amp = spectra[name][key]
      axes[r, 1].plot(freqs, amp, color=col, lw=1.6, label=name)
  for r, key in enumerate(("roll", "pitch")):
    axes[r, 0].set_ylabel(f"{key} (deg)")
    axes[r, 0].set_title(f"torso {key}, env {args.env}, command {cmd}", loc="left", fontsize=10)
    axes[r, 1].set_title(f"{key} amplitude spectrum, mean of upright envs", loc="left", fontsize=10)
    axes[r, 1].set_xlim(0, 8)
    axes[r, 1].set_ylabel("amplitude (deg)")
    for ax in axes[r]:
      ax.grid(alpha=0.3)
  axes[0, 0].legend(fontsize=8, frameon=False)
  axes[1, 0].set_xlabel("time (s)")
  axes[1, 1].set_xlabel("frequency (Hz)")
  fig.tight_layout()
  sway_path = f"{args.out}_sway.png"
  fig.savefig(sway_path, dpi=110)
  print(f"\nwrote {feet_path} and {sway_path}")


if __name__ == "__main__":
  main()

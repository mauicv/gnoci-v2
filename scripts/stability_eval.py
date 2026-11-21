"""Compare how often checkpoints fall in the sim, with and without pushes.

Runs each checkpoint's deterministic policy for --seconds in --num-envs envs per
scenario. The env samples an action and IMU delay per episode as in training
(latency_config), and the env's own pushes are off; pushes are applied here
instead, the same way (a sideways kick added to the torso velocity).

Scenarios (everything but "calm" uses domain_randomize, with the same
randomizations for every checkpoint):

  calm            nominal model, a fixed command per env (stand, forward, back,
                  sideways, turning), no pushes
  randomized      random commands resampled every 5 s, no pushes
  train pushes    as randomized, plus the training pushes: 0.05-0.5 m/s every
                  5-10 s
  strong pushes   as randomized, plus 0.3-0.8 m/s every 3-6 s

and two push sweeps, standing and walking forward at 0.15 m/s: a single kick at
t = 4 s of a set speed (0.1-1.6 m/s) forward, back, left or right, counted as a
fall if the robot falls within --sweep-window s after it.

A fall is the env's termination (tilted past 60 deg or torso below 0.15 m).
Writes <out>.png and <out>.json and prints a table.

  uv run python scripts/stability_eval.py
  uv run python scripts/stability_eval.py --checkpoint slain-butter-mate saint-oil-fast
"""

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

from gnoci_v2.training import joystick, randomize
from gnoci_v2.training.constants import ENV_CONFIG_FILE
from gnoci_v2.training.convert import load_checkpoint, make_deterministic_policy

CHECKPOINTS = Path("checkpoints")
DEFAULT = ["sassy-temple-bandit", "saint-oil-fast", "giddy-purple-honor", "slain-butter-mate"]
COMMAND = slice(6, 9)
RESAMPLE_S = 5.0
KICK_S = 4.0
MAGNITUDES = np.round(np.arange(0.1, 1.65, 0.1), 2)
DIRECTIONS = {"forward": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1)}
CALM_COMMANDS = np.array([
    [0, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [-0.15, 0, 0],
    [0, 0.15, 0], [0, -0.15, 0], [0, 0, 0.7], [0.15, 0, -0.7],
])
# Categorical slots 1-5 of the dataviz reference palette, in fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NO_PUSH = (np.inf, np.inf, 0.0, 0.0)


class _Batched:
  """Vmaps reset/step over envs without domain randomization."""

  def __init__(self, env):
    self.reset = jax.vmap(env.reset)
    self.step = jax.vmap(env.step)


def make_rollout(venv, sample_command, policy, num_envs, num_steps, dt):
  """Jitted fn(rng, scenario arrays) -> per-env first fall step (num_steps if none)."""
  resample_every = int(round(RESAMPLE_S / dt))

  def rollout(rng, sc):
    rng, reset_rng = jax.random.split(rng)
    state = venv.reset(jax.random.split(reset_rng, num_envs))
    cmd = sc["command"]

    def set_command(state, cmd):
      state.info["command"] = cmd
      state.info["step"] = jp.zeros_like(state.info["step"])  # No env resampling.
      return state

    state = set_command(state, cmd)
    state = state.replace(obs={**state.obs, "state": state.obs["state"].at[:, COMMAND].set(cmd)})
    rng, k = jax.random.split(rng)
    next_push = jax.random.uniform(k, (num_envs,), minval=sc["interval"][:, 0],
                                   maxval=sc["interval"][:, 1])
    next_push = jp.round(next_push / dt)

    def body(carry, t):
      state, cmd, next_push, fall_step, rng = carry
      rng, k_cmd, k_dir, k_mag, k_int = jax.random.split(rng, 5)
      # Commands.
      new_cmd = jax.vmap(sample_command)(jax.random.split(k_cmd, num_envs))
      resample = sc["resample"] & (t > 0) & (t % resample_every == 0)
      cmd = jp.where(resample[:, None], new_cmd, cmd)
      # Random pushes.
      push_now = t == next_push
      theta = jax.random.uniform(k_dir, (num_envs,), maxval=2 * jp.pi)
      mag = jax.random.uniform(k_mag, (num_envs,), minval=sc["magnitude"][:, 0],
                               maxval=sc["magnitude"][:, 1])
      kick = jp.stack([jp.cos(theta), jp.sin(theta)], -1) * mag[:, None] * push_now[:, None]
      interval = jax.random.uniform(k_int, (num_envs,), minval=sc["interval"][:, 0],
                                    maxval=sc["interval"][:, 1])
      next_push = jp.where(push_now, next_push + jp.round(interval / dt), next_push)
      # The sweep's single kick.
      kick += sc["kick"] * (t == sc["kick_step"])[:, None]
      qvel = state.data.qvel.at[:, :2].add(kick)
      state = state.replace(data=state.data.replace(qvel=qvel))
      state = set_command(state, cmd)
      state = venv.step(state, policy(state.obs["state"]))
      fell = (state.done > 0) & (fall_step == num_steps)
      fall_step = jp.where(fell, t, fall_step)
      return (state, cmd, next_push, fall_step, rng), None

    init = (state, cmd, next_push, jp.full(num_envs, num_steps), rng)
    (_, _, _, fall_step, _), _ = jax.lax.scan(body, init, jp.arange(num_steps))
    return fall_step

  return jax.jit(rollout)


def scenario(n, command, resample, push=NO_PUSH, kick=None, kick_step=None):
  i_lo, i_hi, m_lo, m_hi = push
  return {
      "command": jp.asarray(command, jp.float32),
      "resample": jp.full(n, resample),
      "interval": jp.tile(jp.array([i_lo, i_hi], jp.float32), (n, 1)),
      "magnitude": jp.tile(jp.array([m_lo, m_hi], jp.float32), (n, 1)),
      "kick": jp.zeros((n, 2)) if kick is None else jp.asarray(kick, jp.float32),
      "kick_step": jp.full(n, -1) if kick_step is None else jp.full(n, kick_step),
  }


def sweep_kicks(n):
  """Kick vectors for n envs: every (direction, magnitude) pair, repeated."""
  pairs = [(d, m) for d in DIRECTIONS for m in MAGNITUDES]
  reps = n // len(pairs)
  if reps * len(pairs) != n:
    raise ValueError(f"--num-envs must be a multiple of {len(pairs)}")
  kick = np.array([np.array(DIRECTIONS[d]) * m for d, m in pairs for _ in range(reps)])
  return kick, reps


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--checkpoint", nargs="+", default=DEFAULT)
  p.add_argument("--num-envs", type=int, default=512,
                 help=f"Per scenario; a multiple of {len(DIRECTIONS) * len(MAGNITUDES)}")
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--sweep-window", type=float, default=4.0, help="s after the kick")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default="videos/stability")
  args = p.parse_args()

  n = args.num_envs
  config = joystick.default_config()
  config.push_config.enable = False
  dt = config.ctrl_dt
  steps = int(round(args.seconds / dt))
  kick_step = int(round(KICK_S / dt))
  window = int(round(args.sweep_window / dt))
  kick, reps = sweep_kicks(n)
  rng = jax.random.PRNGKey(args.seed)
  rng, dr_rng, cmd_rng = jax.random.split(rng, 3)

  calm_cmd = np.tile(CALM_COMMANDS, (n // len(CALM_COMMANDS) + 1, 1))[:n]
  env0 = joystick.GnociV2Joystick(config=config)
  random_cmd = jax.vmap(env0.sample_command)(jax.random.split(cmd_rng, n))
  stand = np.zeros((n, 3))
  walk = np.tile([0.15, 0, 0], (n, 1))
  scenarios = {
      "calm": scenario(n, calm_cmd, False),
      "randomized": scenario(n, random_cmd, True),
      "train pushes": scenario(n, random_cmd, True, (5.0, 10.0, 0.05, 0.5)),
      "strong pushes": scenario(n, random_cmd, True, (3.0, 6.0, 0.3, 0.8)),
      "sweep standing": scenario(n, stand, False, kick=kick, kick_step=kick_step),
      "sweep walking": scenario(n, walk, False, kick=kick, kick_step=kick_step),
  }

  results = {}
  for name in args.checkpoint:
    path = (CHECKPOINTS / name).resolve()
    cfg = json.loads((path / ENV_CONFIG_FILE).read_text())
    config.action_scale = float(cfg["action_scale"])
    env = joystick.GnociV2Joystick(config=config)
    policy = jax.vmap(make_deterministic_policy(*load_checkpoint(path)))
    nominal = make_rollout(_Batched(env), env.sample_command, policy, n, steps, dt)
    dr_env = BraxDomainRandomizationVmapWrapper(
        env, functools.partial(randomize.domain_randomize, rng=jax.random.split(dr_rng, n)))
    randomized = make_rollout(dr_env, env.sample_command, policy, n, steps, dt)
    res = {}
    for sname, sc in scenarios.items():
      run = nominal if sname == "calm" else randomized
      res[sname] = np.asarray(run(rng, sc))
      print(f"{name:20} {sname:15} fell {np.mean(res[sname] < steps):6.1%}", flush=True)
    results[name] = res

  # Summary.
  summary = {}
  for name, res in results.items():
    s = {}
    for sname in ("calm", "randomized", "train pushes", "strong pushes"):
      fs = res[sname]
      fell = fs < steps
      s[sname] = {"fall_rate": float(fell.mean()),
                  "falls_per_min": float(fell.sum() / (np.minimum(fs + 1, steps).sum() * dt / 60))}
    for sname in ("sweep standing", "sweep walking"):
      fs = res[sname]
      pre = fs < kick_step  # Fell before the kick: left out.
      hit = (fs >= kick_step) & (fs < kick_step + window)
      table = {}
      for d_i, d in enumerate(DIRECTIONS):
        rates = []
        for m_i, m in enumerate(MAGNITUDES):
          sl = slice((d_i * len(MAGNITUDES) + m_i) * reps, (d_i * len(MAGNITUDES) + m_i + 1) * reps)
          ok = ~pre[sl]
          rates.append(float(hit[sl][ok].mean()) if ok.any() else float("nan"))
        table[d] = rates
      s[sname] = {"fell_before_kick": int(pre.sum()), "fall_rate": table}
      # Push that topples half the envs, per direction (linear interpolation).
      s[sname]["p50"] = {d: _crossing(MAGNITUDES, r) for d, r in table.items()}
    summary[name] = s

  print(f"\n{'checkpoint':20} {'calm':>6} {'random':>7} {'train':>7} {'strong':>7}"
        f"   push (m/s) that topples 50%, standing / walking: fwd back left right")
  for name, s in summary.items():
    row = f"{name:20}" + "".join(
        f" {s[k]['fall_rate']:6.1%}" for k in ("calm", "randomized", "train pushes", "strong pushes"))
    for k in ("sweep standing", "sweep walking"):
      row += "   " + " ".join(f"{s[k]['p50'][d]:4.2f}" if s[k]["p50"][d] else " >1.6"
                              for d in DIRECTIONS)
    print(row)
  print(f"(fall rate: share of envs that fell in {args.seconds:g} s)")

  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(f"{args.out}.json").write_text(json.dumps(
      {"magnitudes": MAGNITUDES.tolist(), "seconds": args.seconds, "num_envs": n,
       "results": summary}, indent=1))
  plot(summary, results, args, steps, dt)


def _crossing(x, y):
  y = np.asarray(y)
  idx = np.where(y >= 0.5)[0]
  if not len(idx):
    return None
  i = idx[0]
  if i == 0:
    return float(x[0])
  return float(x[i - 1] + (0.5 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def plot(summary, results, args, steps, dt):
  names = list(summary)
  colors = dict(zip(names, PALETTE))
  ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e4e3df"
  plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted,
                       "ytick.color": muted, "axes.titlecolor": ink, "font.size": 9})
  fig = plt.figure(figsize=(15, 10))
  gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1], hspace=0.55, wspace=0.25)

  # Fall rate per scenario.
  ax = fig.add_subplot(gs[0, :2])
  scen = ["calm", "randomized", "train pushes", "strong pushes"]
  width = 0.8 / len(names)
  for i, name in enumerate(names):
    vals = [100 * summary[name][s]["fall_rate"] for s in scen]
    x = np.arange(len(scen)) + (i - (len(names) - 1) / 2) * width
    ax.bar(x, vals, width * 0.9, color=colors[name], label=name)
    for xi, v in zip(x, vals):
      ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=muted)
  ax.set_xticks(np.arange(len(scen)), scen)
  ax.set_ylabel(f"envs that fell in {args.seconds:g} s (%)")
  ax.set_title("Fall rate by scenario", loc="left")
  ax.legend(frameon=False, fontsize=8)
  ax.grid(axis="y", color=grid)
  ax.set_axisbelow(True)

  # Survival under strong pushes.
  ax = fig.add_subplot(gs[0, 2:])
  t = np.arange(steps) * dt
  for name in names:
    for s, style in (("train pushes", "--"), ("strong pushes", "-")):
      fs = results[name][s]
      surv = (fs[None, :] > np.arange(steps)[:, None]).mean(1) * 100
      ax.plot(t, surv, style, color=colors[name], lw=2,
              label=f"{name}" if s == "strong pushes" else None)
  ax.set_ylim(0, 102)
  ax.set_xlabel("time (s)")
  ax.set_ylabel("still up (%)")
  ax.set_title("Survival: solid = strong pushes, dashed = training pushes", loc="left")
  ax.grid(color=grid)
  ax.legend(frameon=False, fontsize=8, loc="lower left")

  # Push sweeps.
  for r, s in enumerate(("sweep standing", "sweep walking"), start=1):
    for c, d in enumerate(DIRECTIONS):
      ax = fig.add_subplot(gs[r, c])
      for name in names:
        ax.plot(MAGNITUDES, 100 * np.array(summary[name][s]["fall_rate"][d]), "-o",
                color=colors[name], lw=2, ms=4, label=name)
      ax.axvspan(0.05, 0.5, color=grid, alpha=0.6, lw=0)
      ax.set_ylim(-3, 103)
      ax.set_xlim(0, 1.65)
      ax.grid(color=grid)
      ax.set_title(f"{s.split()[1]}: pushed {d}", loc="left")
      if c == 0:
        ax.set_ylabel(f"fell within {args.sweep_window:g} s (%)")
      if r == 2:
        ax.set_xlabel("push (m/s)")
  fig.text(0.01, 0.005, "Shaded: the training push range. Domain randomization and latency on "
           "except 'calm'. Same randomizations for every checkpoint.", color=muted, fontsize=8)
  fig.savefig(f"{args.out}.png", dpi=110, bbox_inches="tight")
  print(f"wrote {args.out}.png and {args.out}.json")


if __name__ == "__main__":
  main()

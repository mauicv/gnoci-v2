"""Estimates the robot's centre of mass from scripts/lean_test.py recordings.

For each robot recording it finds the lean at which the feet started to tip,
runs the same lean test in the
sim with the torso's centre of mass shifted by each of --com-x mm, and
interpolates the shift at which the sim tips at the robot's lean.

The robot's foot pitch drifts steadily as it leans (flex in the soles and legs,
a few degrees per rad, about 10x the sim's), so a fixed threshold catches the
drift, not the tip. Instead a line is fitted to the first 60% of the lean and
the tip is where the foot pitch leaves that line by more than --tip degrees.

The lean is measured from the joints (the measured angles projected onto the
lean direction), not the targets, so servos sagging under load don't count as
a centre-of-mass difference.

  uv run python scripts/lean_compare.py logs/lean/robot-forwards-1.npz logs/lean/robot-backwards-1.npz

Forwards and backwards give two independent estimates; if they disagree, the
difference is something else (foot shape, compliance, IMU offset). The shift is
of the torso's centre of mass (body_main, about half the robot's mass); the
whole robot's centre of mass moves by torso mass / total mass of that.
Sim runs are cached in logs/lean/sim-cache/: delete it after changing the model
or lean_test.py's settings. Saves a plot next to the first recording.
"""

import argparse
from pathlib import Path

import numpy as np

import lean_test

CACHE = Path("logs/lean/sim-cache")


def measured_lean(d):
  """The joints' measured angles projected onto the lean direction, per step."""
  direction = np.asarray(d["lean_direction"], dtype=float)
  pos = d["obs"][:, 9:15]
  return pos @ direction / (direction @ direction)


def tip_lean(d, tip):
  """(measured lean, target lean) where the foot pitch leaves its early drift line by tip deg."""
  idx = np.where(d["phase"] == 1)[0]
  x, y = measured_lean(d)[idx], d["foot_pitch"][idx]
  early = x < 0.6 * x.max()
  if early.sum() < 10:
    return None
  slope, offset = np.polyfit(x[early], y[early], 1)
  # Smoothed a little, against the robot's IMU noise.
  off_line = np.convolve(np.abs(y - (slope * x + offset)), np.ones(9) / 9, mode="same")
  hits = np.where((off_line > tip) & (x > 0.05))[0]
  if not len(hits):
    return None
  k = idx[hits[0]]
  return float(measured_lean(d)[k]), float(d["lean"][k])


def sim_run(way, com_x):
  path = CACHE / f"{way}-com{com_x:+g}-rate{lean_test.LEAN_RATE:g}.npz"
  if not path.exists():
    lean_test.run(way, sim=True, com_x_mm=com_x, out=path)
  return dict(np.load(path))


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("robot", type=Path, nargs="+")
  p.add_argument("--com-x", type=float, nargs="+", default=list(range(-50, 51, 10)),
                 help="Torso centre-of-mass shifts (mm, + forward) to run in the sim.")
  p.add_argument("--tip", type=float, default=1.0,
                 help="Degrees the foot pitch must leave its early drift line by to count as tipping.")
  args = p.parse_args()

  model = lean_test.sim_model(0.0)
  torso = model.body("body_main").id
  torso_share = model.body_mass[torso] / model.body_subtreemass[0]

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
  ax_curve = {"forwards": axes[0], "backwards": axes[1]}

  lo, hi = min(args.com_x), max(args.com_x) + 1e-9
  sims = {}
  plotted = set()
  for path in args.robot:
    real = dict(np.load(path))
    if bool(real.get("sim", False)):
      print(f"{path}: is a sim recording, skipped")
      continue
    way = str(real["way"])
    if way not in sims:
      sims[way] = {c: sim_run(way, c) for c in args.com_x}
    tips = {c: tip_lean(s, args.tip) for c, s in sims[way].items()}
    found = [(c, t[0]) for c, t in tips.items() if t is not None]

    got = tip_lean(real, args.tip)
    print(f"\n{path} ({way}):")
    print("  sim tip lean (measured, rad) by torso COM shift: "
          + ", ".join(f"{c:+g} mm: {t:.3f}" for c, t in found))
    if got is None:
      print(f"  the robot never tipped (foot pitch stayed within {args.tip} deg)")
    else:
      cs, ts = np.array([c for c, _ in found]), np.array([t for _, t in found])
      order = np.argsort(ts)
      if ts.min() <= got[0] <= ts.max():
        est = float(np.interp(got[0], ts[order], cs[order]))
        print(f"  robot tipped at lean {got[0]:.3f} (target {got[1]:.3f})"
              f" -> torso COM {est:+.0f} mm vs the model,"
              f" whole robot {est * torso_share:+.0f} mm (torso is {torso_share:.0%} of the mass)")
      else:
        print(f"  robot tipped at lean {got[0]:.3f}, outside the sim's range"
              f" {ts.min():.3f}-{ts.max():.3f}: widen --com-x")

    # Only the leaning phase: the return to home would draw a line back across.
    ax = ax_curve[way]
    if way not in plotted:
      plotted.add(way)
      for c, s in sims[way].items():
        m = s["phase"] == 1
        ax.plot(measured_lean(s)[m], s["foot_pitch"][m], color=plt.cm.coolwarm((c - lo) / (hi - lo)),
                lw=1, label=f"sim {c:+g} mm")
    m = real["phase"] == 1
    ax.plot(measured_lean(real)[m], real["foot_pitch"][m], color="k", lw=1.5,
            ls=["-", "--", ":", "-."][len(ax.lines) % 4], label=f"robot {path.stem}")
  for way, ax in ax_curve.items():
    ax.set_xlabel("lean, measured from the joints (rad)")
    ax.set_ylabel("foot pitch change (deg)")
    ax.grid(alpha=0.3)
    ax.set_title(f"{way}: foot pitch change vs lean (sim: torso COM shift vs the model)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
  out = args.robot[0].with_name("lean_compare.png")
  fig.tight_layout()
  fig.savefig(out, dpi=100)
  print(f"\nSaved {out}")


if __name__ == "__main__":
  main()

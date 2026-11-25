"""Compares a robot and a sim recording of scripts/stepping_test.py, segment by segment.

  uv run python scripts/stepping_compare.py logs/stepping/robot-1.npz logs/stepping/sim-default.npz

Both play the same open-loop targets, so they should move the same. For each
segment it uses the hold at full amplitude (the ramps are left out) and works
at the stepping frequency f (and 2f, where pitch and the vertical
accelerometer mostly move) with a lock-in: the amplitude and phase of each
signal's f component, the phase relative to the left hip_lat target.

  joints   how far each joint swings, and how far it lags its target
  gyro     body rotation speed; true tilt amplitude = gyro amplitude / (2 pi f)
  accel    in g; plus the rms above 5 Hz (impacts, rattle)
  filter   the complementary filter's roll/pitch amplitude at f, and its error
           against the true tilt at f (the gyro-derived one), as an amplitude

Roll/pitch here are the gravity observation's angles (roll about x, pitch about
y). Also prints the baseline (standing still) means and noise, and saves
<robot>_vs_<sim>.png next to the robot recording.
"""

import argparse
from pathlib import Path

import numpy as np

JOINTS = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]
SHORT = ["L hlat", "L hip", "L knee", "R hlat", "R hip", "R knee"]
GYRO, GRAV, POS = slice(0, 3), slice(3, 6), slice(9, 15)


def angles(g):
  """Roll, pitch (deg) from gravity vectors in the IMU frame."""
  return (np.degrees(np.arctan2(-g[:, 1], -g[:, 2])),
          np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1))))


def lockin(t, x, f):
  """Complex amplitude of x's component at f (Hz): |A| is the sine amplitude."""
  x = x - x.mean(0)
  e = np.exp(-2j * np.pi * f * (t - t[0]))
  if x.ndim > 1:
    e = e[:, None]
  return 2 * (x * e).mean(0)


def highpass_rms(x, dt, cutoff=5.0):
  x = x - x.mean(0)
  spec = np.fft.rfft(x, axis=0)
  freqs = np.fft.rfftfreq(len(x), dt)
  spec[freqs < cutoff] = 0
  return np.sqrt((np.fft.irfft(spec, n=len(x), axis=0) ** 2).mean(0))


def segment_stats(d, s):
  """Stats over segment s's hold (where the amplitude is at its maximum)."""
  m = (d["segment"] == s)
  amp = d["amplitude"]
  hold = m & np.isclose(amp, amp[m].max())
  idx = np.where(hold)[0]
  if len(idx) < 50:
    return None
  t = d["t"][idx] - d["t"][idx[0]]
  f = float(d["frequency"][idx[0]])
  obs = d["obs"][idx]
  w = 2 * np.pi * f
  ref = lockin(t, d["target"][idx, 0], f)
  rot = np.conj(ref) / abs(ref)  # Rotates phases so the left hip_lat target is at 0.

  def ph(a):
    return a * rot

  roll, pitch = angles(d["gravity"][idx])
  gyro = obs[:, GYRO]
  out = {"f": f, "amp": float(amp[idx[0]]), "n": len(idx)}
  out["pos"] = ph(lockin(t, obs[:, POS], f))
  out["tgt"] = ph(lockin(t, d["target"][idx], f))
  out["gyro"] = ph(lockin(t, gyro, f))
  out["gyro2"] = lockin(t, gyro, 2 * f)
  # True tilt at f from the gyro: angle = rate / (j w), in degrees.
  true_roll = out["gyro"][0] / (1j * w) * 180 / np.pi
  true_pitch = out["gyro"][1] / (1j * w) * 180 / np.pi
  true_pitch2 = out["gyro2"][1] / (2j * w) * 180 / np.pi
  out["roll"] = ph(lockin(t, roll, f))
  out["pitch"] = ph(lockin(t, pitch, f))
  out["pitch2"] = lockin(t, pitch, 2 * f)
  out["true_roll"], out["true_pitch"], out["true_pitch2"] = true_roll, true_pitch, true_pitch2
  out["roll_err"] = abs(out["roll"] - true_roll)
  out["pitch_err"] = abs(out["pitch"] - true_pitch)
  out["pitch2_err"] = abs(out["pitch2"] - true_pitch2)
  out["accel"] = ph(lockin(t, d["accel"][idx], f))
  out["accel2"] = lockin(t, d["accel"][idx], 2 * f)
  dt = float(np.median(np.diff(d["t"])))
  out["accel_hf"] = highpass_rms(d["accel"][idx], dt)
  out["gyro_hf"] = highpass_rms(gyro, dt)
  out["roll_mean"], out["pitch_mean"] = roll.mean(), pitch.mean()
  out["gyro_rms"] = np.sqrt((gyro ** 2).mean(0))
  return out


def baseline(d):
  m = np.where(d["segment"] == -1)[0]
  m = m[m < np.argmax(d["segment"] >= 0)]  # Only the first rest, before any stepping.
  m = m[10:]
  roll, pitch = angles(d["gravity"][m])
  return {"roll": roll.mean(), "pitch": pitch.mean(),
          "gyro_std": d["obs"][m, GYRO].std(0), "accel_std": d["accel"][m].std(0),
          "accel_mean": d["accel"][m].mean(0), "pos": d["obs"][m, POS].mean(0)}


def fmt(v, w=6, p=2):
  return f"{v:{w}.{p}f}"


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("robot", type=Path)
  p.add_argument("sim", type=Path)
  args = p.parse_args()
  real, sim = dict(np.load(args.robot)), dict(np.load(args.sim))

  b_r, b_s = baseline(real), baseline(sim)
  print("BASELINE (standing still before the first segment)      robot      sim")
  print(f"  filter roll mean (deg)                             {fmt(b_r['roll'],8,1)} {fmt(b_s['roll'],8,1)}")
  print(f"  filter pitch mean (deg)                            {fmt(b_r['pitch'],8,1)} {fmt(b_s['pitch'],8,1)}")
  print(f"  gyro noise std x/y/z (rad/s)   robot {np.round(b_r['gyro_std'],3)}  sim {np.round(b_s['gyro_std'],3)}")
  print(f"  accel noise std x/y/z (g)      robot {np.round(b_r['accel_std'],3)}  sim {np.round(b_s['accel_std'],3)}")
  print(f"  accel mean x/y/z (g)           robot {np.round(b_r['accel_mean'],3)}  sim {np.round(b_s['accel_mean'],3)}")
  print(f"  joint pos mean (rad)           robot {np.round(b_r['pos'],3)}\n"
        f"                                 sim   {np.round(b_s['pos'],3)}")

  n_seg = int(max(real["segment"].max(), sim["segment"].max())) + 1
  rows = []
  for s in range(n_seg):
    r, q = segment_stats(real, s), segment_stats(sim, s)
    if r is None or q is None:
      print(f"\nsegment {s}: missing in {'robot' if r is None else 'sim'}")
      continue
    rows.append((r, q))
    print(f"\n=== segment {s}: {r['f']} Hz x{r['amp']:g}  ({r['n'] * 0.02:.0f} s hold)   robot | sim")
    print("  joint swing at f (rad):    " + " ".join(f"{n:>7}" for n in SHORT))
    print("    target                   " + " ".join(fmt(abs(v), 7, 3) for v in r["tgt"]))
    print("    robot                    " + " ".join(fmt(abs(v), 7, 3) for v in r["pos"]))
    print("    sim                      " + " ".join(fmt(abs(v), 7, 3) for v in q["pos"]))
    lag = lambda st: np.degrees(np.angle(st["tgt"] / st["pos"])) / 360 / st["f"] * 1000
    print("    lag behind target (ms) r " + " ".join(fmt(v, 7, 0) for v in lag(r)))
    print("                           s " + " ".join(fmt(v, 7, 0) for v in lag(q)))
    print(f"  gyro rms x/y/z (rad/s)      {np.round(r['gyro_rms'],2)} | {np.round(q['gyro_rms'],2)}")
    print(f"  gyro >5 Hz rms x/y/z        {np.round(r['gyro_hf'],2)} | {np.round(q['gyro_hf'],2)}")
    print(f"  TRUE roll at f (deg, gyro)  {abs(r['true_roll']):6.2f} | {abs(q['true_roll']):6.2f}"
          f"   phase vs hip_lat target {np.degrees(np.angle(r['true_roll'])):5.0f} | {np.degrees(np.angle(q['true_roll'])):5.0f}")
    print(f"  TRUE pitch at f / 2f (deg)  {abs(r['true_pitch']):6.2f} / {abs(r['true_pitch2']):5.2f} | "
          f"{abs(q['true_pitch']):6.2f} / {abs(q['true_pitch2']):5.2f}")
    print(f"  FILTER roll at f (deg)      {abs(r['roll']):6.2f} | {abs(q['roll']):6.2f}"
          f"   error vs true {r['roll_err']:5.2f} | {q['roll_err']:5.2f}")
    print(f"  FILTER pitch at f / 2f      {abs(r['pitch']):6.2f} / {abs(r['pitch2']):5.2f} | "
          f"{abs(q['pitch']):6.2f} / {abs(q['pitch2']):5.2f}   error f {r['pitch_err']:4.2f} | {q['pitch_err']:4.2f}")
    print(f"  filter mean roll/pitch      {r['roll_mean']:5.1f} / {r['pitch_mean']:5.1f} | "
          f"{q['roll_mean']:5.1f} / {q['pitch_mean']:5.1f}")
    print(f"  accel at f x/y/z (g)        {np.round(abs(r['accel']),3)} | {np.round(abs(q['accel']),3)}")
    print(f"  accel at 2f x/y/z (g)       {np.round(abs(r['accel2']),3)} | {np.round(abs(q['accel2']),3)}")
    print(f"  accel >5 Hz rms x/y/z (g)   {np.round(r['accel_hf'],3)} | {np.round(q['accel_hf'],3)}")

  plot(real, sim, rows, args)


def plot(real, sim, rows, args):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  freqs = sorted({r["f"] for r, _ in rows})
  metrics = [
      ("true roll at f (deg)", lambda st: abs(st["true_roll"])),
      ("filter roll at f (deg)", lambda st: abs(st["roll"])),
      ("filter roll error at f (deg)", lambda st: st["roll_err"]),
      ("true pitch at f (deg)", lambda st: abs(st["true_pitch"])),
      ("lateral accel y at f (g)", lambda st: abs(st["accel"][1])),
      ("accel >5 Hz rms, |xyz| (g)", lambda st: np.linalg.norm(st["accel_hf"])),
      ("gyro >5 Hz rms, |xyz| (rad/s)", lambda st: np.linalg.norm(st["gyro_hf"])),
      ("knee swing at f, mean L/R (rad)", lambda st: abs(st["pos"][[2, 5]]).mean()),
  ]
  fig, axes = plt.subplots(3, 4, figsize=(17, 11))
  for ax, (title, fn) in zip(axes.flat, metrics):
    for i, f in enumerate(freqs):
      sel = [(r, q) for r, q in rows if r["f"] == f]
      a = [r["amp"] for r, _ in sel]
      ax.plot(a, [fn(r) for r, _ in sel], "o-", color=f"C{i}", label=f"robot {f} Hz")
      ax.plot(a, [fn(q) for _, q in sel], "s--", color=f"C{i}", alpha=0.6, label=f"sim {f} Hz")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("amplitude (x policy step)")
    ax.grid(alpha=0.3)
  axes[0, 0].legend(fontsize=8)

  # Time series of one segment: the largest amplitude at the first frequency.
  r, _ = max([rs for rs in rows if rs[0]["f"] == freqs[-1]], key=lambda rs: rs[0]["amp"])
  s = [i for i, (rr, _) in enumerate(rows) if rr is r][0]
  seg_id = [k for k in range(int(real["segment"].max()) + 1)
            if (real["segment"] == k).any() and real["frequency"][real["segment"] == k][0] == r["f"]
            and real["amplitude"][real["segment"] == k].max() == r["amp"]][0]
  for ax, (label, get) in zip(axes[2, :], [
      ("gyro x (rad/s)", lambda d, m: d["obs"][m, 0]),
      ("filter roll (deg)", lambda d, m: angles(d["gravity"][m])[0]),
      ("accel y (g)", lambda d, m: d["accel"][m, 1]),
      ("left hip_lat (rad)", lambda d, m: d["obs"][m, 9]),
  ]):
    for d, name, c in [(real, "robot", "C3"), (sim, "sim", "C0")]:
      m = np.where(d["segment"] == seg_id)[0]
      m = m[len(m) // 2: len(m) // 2 + 150]
      t = d["t"][m] - d["t"][m[0]]
      ax.plot(t, get(d, m), color=c, label=name)
    if label.startswith("left hip_lat"):
      ax.plot(t, real["target"][m, 0], "k--", lw=0.8, label="target")
    ax.set_title(f"{label}, segment {seg_id} ({r['f']} Hz x{r['amp']:g})", fontsize=10)
    ax.set_xlabel("s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
  fig.suptitle(f"{args.robot.name} vs {args.sim.name}")
  fig.tight_layout()
  out = args.robot.with_name(f"{args.robot.stem}_vs_{args.sim.stem}.png")
  fig.savefig(out, dpi=95)
  print(f"\nSaved {out}")


if __name__ == "__main__":
  main()

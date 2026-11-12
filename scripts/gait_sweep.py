"""Gait-frequency sweep: how much the torso rocks at each step frequency, sim vs robot.

Plays one gait cycle open loop (a policy's targets averaged over the gait
phase, control/gait_cycle.py) at a range of frequencies, holding each for a few
seconds, in the free-floating sim. With --real, also analyses a robot run of
the same sweep (`gnoci sweep --record ...`) and plots the two together.

For each frequency it fits the response at that frequency (the last
--analyse-seconds of the segment):
  roll amplitude (deg)     sim: true roll; robot: gyro integral + low-passed accel
  gyro x amplitude (rad/s) measured directly on both, so the most robust
  lateral accel amplitude (g)

  uv run python scripts/gait_sweep.py
  uv run python scripts/gait_sweep.py --real logs/sweep/r1.npz --plot gait_sweep.png
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import butter, filtfilt

from gnoci_v2.control.gait_cycle import cycle_targets, fit_cycle, schedule
from gnoci_v2.training.constants import SIM_PATH

SCENE = SIM_PATH / "desc" / "scene.xml"
DT, SIM_DT = 0.02, 0.002
JOINTS = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]
DEFAULT_CYCLE = "logs/floor/cat-fire-potential/raw-accel-tests/r1-small.npz"
DEFAULT_FREQS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0]
RAMP_SECONDS = 1.0  # as gnoci sweep
TRAIN_RANGE = (1.25, 1.5)  # gait_freq sampled in training


def roll_from_imu(gyro, accel):
    """Roll (rad): gyro x integral, with its slow part from the accelerometer (0.3 Hz)."""
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (gyro[1:, 0] + gyro[:-1, 0]) * DT)])
    accel_roll = np.arctan2(accel[:, 1], accel[:, 2])
    b, a = butter(2, 0.3 / (0.5 / DT))
    return integ + filtfilt(b, a, accel_roll - integ)


def sim_sweep(coef, freqs, seconds, amp):
    """Runs the sweep in the sim. Returns per-step t, gyro, accel (g), true roll, fell time."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    act = [m.actuator(j).id for j in JOINTS]
    start = cycle_targets(coef, 0.0, amp=0.0)
    d.qpos[qadr] = start
    d.ctrl[act] = start
    mujoco.mj_forward(m, d)
    for _ in range(500):  # settle 1 s on the floor
        mujoco.mj_step(m, d)
    imu = m.site("imu").id
    ga = m.sensor_adr[m.sensor("gyro").id]
    aa = m.sensor_adr[m.sensor("accelerometer").id]
    plan = schedule(freqs, seconds)
    total = len(freqs) * seconds
    out = {"t": [], "gyro": [], "accel": [], "roll": []}
    phase, fell, step = 0.0, None, 0
    while step * DT < total:
        t = step * DT
        g = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        if g[2] > -0.5:
            fell = t
            break
        out["t"].append(t)
        out["gyro"].append(d.sensordata[ga:ga + 3].copy())
        out["accel"].append(d.sensordata[aa:aa + 3] / 9.81)
        out["roll"].append(np.arctan2(-g[1], -g[2]))
        freq = [f for s, f in plan if s <= t][-1]
        d.ctrl[act] = cycle_targets(coef, phase, amp=amp * min(1.0, t / RAMP_SECONDS))
        for _ in range(int(round(DT / SIM_DT))):
            mujoco.mj_step(m, d)
        phase = (phase + 2 * np.pi * freq * DT + np.pi) % (2 * np.pi) - np.pi
        step += 1
    out = {k: np.array(v) for k, v in out.items()}
    out["fell"] = fell
    return out


def amplitude_at(x, t, freq):
    """Amplitude of x at freq (least-squares sine + cosine + offset)."""
    X = np.stack([np.ones_like(t), np.cos(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)], 1)
    c, *_ = np.linalg.lstsq(X, x, rcond=None)
    return np.hypot(c[1], c[2])


def analyse(run, freqs, seconds, analyse_seconds):
    """Per frequency: roll amp (deg), gyro x amp (rad/s), accel y amp (g); NaN once fallen."""
    rows = []
    for i, f in enumerate(freqs):
        t0, t1 = (i + 1) * seconds - analyse_seconds, (i + 1) * seconds
        sel = (run["t"] >= t0) & (run["t"] < t1)
        if sel.sum() < 0.8 * analyse_seconds / DT:
            rows.append((f, np.nan, np.nan, np.nan))
            continue
        t = run["t"][sel]
        rows.append((f, np.degrees(amplitude_at(run["roll"][sel], t, f)),
                     amplitude_at(run["gyro"][sel, 0], t, f),
                     amplitude_at(run["accel"][sel, 1], t, f)))
    return np.array(rows)


def load_real(path):
    rec = np.load(path)
    gyro, accel = rec["obs"][:, :3], rec["accel"]
    return rec, {"t": rec["t"], "gyro": gyro, "accel": accel, "roll": roll_from_imu(gyro, accel)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, default=None, help="a gnoci sweep recording")
    ap.add_argument("--cycle-from", nargs="+", default=[DEFAULT_CYCLE],
                    help="recording(s) for the cycle (ignored with --real: its own cycle is used)")
    ap.add_argument("--freqs", type=float, nargs="+", default=DEFAULT_FREQS)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--analyse-seconds", type=float, default=3.0,
                    help="the end of each segment that's analysed (after the change settles)")
    ap.add_argument("--plot", type=Path, default=Path("gait_sweep.png"))
    args = ap.parse_args()

    real = None
    if args.real:
        rec, real = load_real(args.real)
        coef, freqs = rec["cycle_coef"], list(rec["sweep_freqs"])
        seconds, amp = float(rec["sweep_seconds"]), float(rec["sweep_amp"])
    else:
        coef, freqs, seconds, amp = fit_cycle(args.cycle_from), args.freqs, args.seconds, args.amp

    sim = sim_sweep(coef, freqs, seconds, amp)
    if sim["fell"] is not None:
        print(f"sim fell at {sim['fell']:.1f} s")
    results = {"sim": analyse(sim, freqs, seconds, args.analyse_seconds)}
    if real is not None:
        results["robot"] = analyse(real, freqs, seconds, args.analyse_seconds)

    print(f"{'Hz':>5s} " + "  ".join(f"{name:>8s} roll deg  gyro x  accel y" for name in results))
    for i, f in enumerate(freqs):
        print(f"{f:5.2f} " + "  ".join(
            f"{r[i, 1]:17.2f} {r[i, 2]:7.2f} {r[i, 3]:8.3f}" for r in results.values()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sim": "#2a78d6", "robot": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col, label in zip(axes, (1, 2, 3), ("roll amplitude (deg)", "gyro x amplitude (rad/s)",
                                                "lateral accel amplitude (g)")):
        ax.axvspan(*TRAIN_RANGE, color="#e4e3dd", lw=0, label="training gait_freq range")
        for name, r in results.items():
            ax.plot(r[:, 0], r[:, col], marker="o", lw=2, color=colors[name], label=name)
        ax.set_xlabel("gait frequency (Hz per leg)")
        ax.set_ylabel(label)
        ax.set_ylim(0, None)
        ax.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Gait-cycle sweep, played open loop (amp x{amp:g}): torso rocking at each step frequency",
                 x=0.01, ha="left")
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=120, facecolor="white")
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()

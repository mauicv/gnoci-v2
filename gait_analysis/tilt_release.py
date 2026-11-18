"""Tilt-release tests on every foot surface, robot vs sim.

Robot: servos holding home (no policy), the robot tipped sideways by hand onto
one foot and let go. Roll after release is the gyro integrated from the angle at
release (filter roll at the last still moment + gyro integral up to release).
Sim: main's model, same joint pose held by the actuators, rotated about the
lower foot's outer bottom edge to the same roll, released from rest.

  uv run python gait_analysis/tilt_release.py   -> gait_analysis/tilt_release.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from floor_analysis import JOINTS, SCENE, SIM_DT

FILES = [("pads", "logs/floor/cat-fire-potential/raw-accel-tests/tilt-test.npz"),
         ("bare", "logs/floor/cat-fire-potential/raw-accel-tests/tilt-test-nopads.npz"),
         ("rubber", "logs/floor/walk-time-hold/-tilt-test-rubber.np.npz"),
         ("rubber", "logs/floor/walk-time-hold/-tilt-test-rubber.npz")]
COLORS = {"sim": "#2a78d6", "rubber": "#eb6834", "bare": "#1baf7a", "pads": "#eda100"}
LABELS = {"pads": "robot, foam pads", "bare": "robot, bare feet", "rubber": "robot, 1 mm rubber",
          "sim": "sim (main's model)"}
MUTED, GRID = "#6b6a64", "#e4e3dd"
DUR, MIN_DEG = 3.0, 8.0


def robot_events():
    out = []
    for surf, path in FILES:
        d = np.load(path)
        t = d["t"] - d["t"][0]
        o, g = d["obs"], d["gravity"]
        gx = o[:, 0]
        froll = np.arctan2(-g[:, 1], -g[:, 2])
        n = int(DUR / 0.02)
        p, last = 15, -1000
        while p < len(gx) - n:
            big = abs(gx[p]) > 0.4 and abs(gx[p]) >= abs(gx[p - 1]) and abs(gx[p]) >= abs(gx[p + 1])
            if not big or p - last < 100:
                p += 1
                continue
            s = np.sign(gx[p])
            i = p
            while i > 15 and s * gx[i] > 0.03:  # back to where the release swing starts (at rest)
                i -= 1
            k = i  # last still moment before the hand tipped it
            while k > 10 and np.abs(gx[k - 10:k]).max() > 0.05:
                k -= 1
            r0 = froll[k - 5:k].mean() + np.trapezoid(gx[k:i + 1], dx=0.02)
            if abs(np.degrees(r0)) >= MIN_DEG and np.sign(r0) != s:
                seg = slice(i, i + n)
                roll = r0 + np.concatenate([[0], np.cumsum(0.5 * (gx[seg][1:] + gx[seg][:-1]) * 0.02)])
                out.append({"surface": surf, "file": Path(path).name, "t0": t[i], "roll0": r0,
                            "roll": roll, "q": o[i, 9:15]})
                last = p
            p += 1
    return out


def sim_release(roll0, q, dur=DUR):
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    act = [m.actuator(j).id for j in JOINTS]
    d.qpos[qadr] = q
    d.ctrl[act] = q
    mujoco.mj_forward(m, d)
    # Negative roll = left side down: pivot on the left foot's outer edge (+y), else the right's (-y).
    foot = "left_foot_base" if roll0 < 0 else "right_foot_base"
    gid = m.geom(foot).id
    R = d.geom_xmat[gid].reshape(3, 3)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[gid]
    world = d.geom_xpos[gid] + corners @ R.T
    y = world[:, 1].max() if roll0 < 0 else world[:, 1].min()
    pivot = np.array([0.0, y, world[:, 2].min()])
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([1.0, 0, 0]), roll0)
    Rr = np.zeros(9)
    mujoco.mju_quat2Mat(Rr, quat)
    d.qpos[:3] = pivot + Rr.reshape(3, 3) @ (d.qpos[:3] - pivot)
    d.qpos[3:7] = quat
    d.qpos[2] += 0.0005
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    imu = m.site("imu").id
    roll = []
    for _ in range(int(dur / 0.02)):
        gv = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        roll.append(np.arctan2(-gv[1], -gv[2]))
        for _ in range(int(round(0.02 / SIM_DT))):
            mujoco.mj_step(m, d)
    return np.array(roll)


def swing_peaks(roll, min_deg=0.8):
    """|roll| at the release and at each later turning point above min_deg, in degrees."""
    r = np.degrees(roll)
    ext = [k for k in range(1, len(r) - 1) if (r[k] - r[k - 1]) * (r[k + 1] - r[k]) < 0]
    peaks, sign = [abs(r[0])], np.sign(r[0])
    for k in ext:
        if np.sign(r[k]) == -sign and abs(r[k]) > min_deg:
            peaks.append(abs(r[k]))
            sign = -sign
    return np.array(peaks)


def decay(peaks):
    """Geometric-mean amplitude ratio per half-swing, from the first overshoot on."""
    p = peaks[1:]
    if len(p) >= 2:
        return float(np.exp(np.mean(np.diff(np.log(p)))))
    return np.nan


def style(ax):
    ax.grid(color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    ev = robot_events()
    for e in ev:
        e["sim"] = sim_release(e["roll0"], e["q"])
        e["pk"], e["sim_pk"] = swing_peaks(e["roll"]), swing_peaks(e["sim"])
        print(f"{e['surface']:7s} {e['file']:28s} t {e['t0']:5.1f} from {np.degrees(e['roll0']):+6.1f} deg | "
              f"robot overshoot {e['pk'][1] / e['pk'][0] if len(e['pk']) > 1 else 0:4.2f} decay {decay(e['pk']):.2f} "
              f"swings {len(e['pk']) - 1} | sim overshoot {e['sim_pk'][1] / e['sim_pk'][0] if len(e['sim_pk']) > 1 else 0:4.2f} "
              f"decay {decay(e['sim_pk']):.2f} swings {len(e['sim_pk']) - 1}")

    order = ["pads", "bare", "rubber"]
    ev.sort(key=lambda e: (order.index(e["surface"]), e["file"], e["t0"]))
    ncol = 4
    nrow = 1 + int(np.ceil(len(ev) / ncol))
    fig = plt.figure(figsize=(15, 3.3 * nrow))
    gs = fig.add_gridspec(nrow, ncol)

    # Summary 1: swing amplitude relative to the release angle, per half-swing.
    ax = fig.add_subplot(gs[0, :2])
    style(ax)
    for e in ev:
        ax.plot(np.arange(len(e["pk"])), e["pk"] / e["pk"][0], color=COLORS[e["surface"]], lw=1.6, marker="o", ms=4)
        ax.plot(np.arange(len(e["sim_pk"])), e["sim_pk"] / e["sim_pk"][0], color=COLORS["sim"], lw=1.2,
                marker="o", ms=4, alpha=.6)
    ax.set_yscale("log")
    ax.set_ylim(0.04, 1.1)
    ax.set_xlabel("half-swing (0 = release)")
    ax.set_ylabel("|roll peak| / release angle")
    ax.set_title("Swing amplitude after release (each line one release; peaks above 0.8 deg)", loc="left", fontsize=10)
    for s in ["pads", "bare", "rubber", "sim"]:
        ax.plot([], [], color=COLORS[s], lw=2, marker="o", ms=4, label=LABELS[s])
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # Summary 2: decay ratio per half-swing, per release.
    ax = fig.add_subplot(gs[0, 2:])
    style(ax)
    cats = ["pads", "bare", "rubber", "sim"]
    rng = np.random.default_rng(0)
    for xi, c in enumerate(cats):
        vals = [decay(e["sim_pk"]) for e in ev] if c == "sim" else [decay(e["pk"]) for e in ev if e["surface"] == c]
        n_rel = len(vals)
        vals = np.array([v for v in vals if v == v])
        ax.scatter(xi + rng.uniform(-.12, .12, len(vals)), vals, color=COLORS[c], s=40, zorder=3,
                   edgecolor="white", lw=1)
        if len(vals):
            med = np.median(vals)
            ax.plot([xi - .25, xi + .25], [med, med], color="#1f1f1e", lw=1.5)
            ax.annotate(f"{med:.2f}", (xi + .28, med), va="center", fontsize=8, color="#1f1f1e")
        if len(vals) < n_rel:
            ax.annotate(f"{n_rel - len(vals)} stopped\nafter 1 swing", (xi, 0.13), ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(cats)), [LABELS[c] for c in cats])
    ax.set_xlim(-.5, len(cats) - .3)
    ax.set_ylim(0.1, 1.35)
    ax.set_ylabel("amplitude ratio per half-swing")
    ax.set_title("Decay per half-swing (lower = settles faster; bar = median)", loc="left", fontsize=10)

    # One panel per release.
    for k, e in enumerate(ev):
        ax = fig.add_subplot(gs[1 + k // ncol, k % ncol])
        style(ax)
        ax.axhline(0, color=MUTED, lw=.8)
        t = np.arange(len(e["roll"])) * 0.02
        ax.plot(t, np.degrees(e["roll"]), color=COLORS[e["surface"]], lw=2, label=LABELS[e["surface"]])
        ax.plot(t[:len(e["sim"])], np.degrees(e["sim"]), color=COLORS["sim"], lw=1.6, label=LABELS["sim"])
        ax.set_title(f"{e['surface']}: {e['file'].lstrip('-').replace('.npz', '')} at {e['t0']:.1f} s, "
                     f"from {np.degrees(e['roll0']):+.0f} deg", loc="left", fontsize=9)
        ax.set_xlim(0, DUR)
        if k % ncol == 0:
            ax.set_ylabel("roll (deg, + = right side down)")
        if k >= len(ev) - ncol:
            ax.set_xlabel("time since release (s)")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right" if e["roll0"] < 0 else "lower right")
    fig.tight_layout()
    out = Path(__file__).with_name("tilt_release.png")
    fig.savefig(out, dpi=110, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()

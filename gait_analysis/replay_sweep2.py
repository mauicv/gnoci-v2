"""One-at-a-time sensitivity of the x0.3 open-loop replay, plus plots.

For each parameter, varies it alone around two bases (torso COM shift 0 and
+40 mm) and records the gait-frequency features. Then plots:
  replay_sensitivity.png  features vs each parameter, robot value as a band
  replay_timeseries.png   roll, pitch, gyro x: robot vs current model vs best

  uv run python replay_sweep2.py <scratchpad dir> <out dir>
"""
import itertools
import sys

sys.path.insert(0, sys.argv[1])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from floor_analysis import DT, JOINTS, SCENE, SIM_DT, angles, clock_fit
from replay_fit import RUNS, SKIP, load_run

BASE = mujoco.MjModel.from_xml_path(str(SCENE))
TORSO = BASE.body("body_main").id
BASE_IPOS = BASE.body_ipos[TORSO].copy()
FEET = [BASE.geom(n).id for n in ("left_foot_base", "right_foot_base")]
FLOOR = BASE.geom("floor").id
HIP_LAT = [BASE.actuator(n).id for n in ("left_hip_lat", "right_hip_lat")]

DEFAULTS = dict(dx=0.0, dy=0.0, dz=0.0, friction=1.0, kp=1.0, kp_hip_lat=1.0, solref=0.02, foot_width=34.0)
SWEEPS = {
    "dx": ([-0.02, 0.0, 0.02, 0.04, 0.06], "torso COM x shift (m)"),
    "dz": ([-0.03, 0.0, 0.03, 0.06], "torso COM z shift (m)"),
    "friction": ([0.3, 0.5, 0.75, 1.0], "foot/floor friction"),
    "kp": ([0.6, 0.8, 1.0, 1.2], "kp scale, all joints"),
    "kp_hip_lat": ([0.3, 0.5, 0.75, 1.0], "kp scale, hip_lat only"),
    "solref": ([0.01, 0.02, 0.04, 0.08], "contact solref timeconst (s)"),
    "foot_width": ([12.0, 20.0, 27.0, 34.0], "foot contact width (mm)"),
}


def replay(run, p):
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = SIM_DT
    m.body_ipos[TORSO] = BASE_IPOS + [p["dx"], p["dy"], p["dz"]]
    for g in FEET + [FLOOR]:
        m.geom_friction[g, 0] = p["friction"]
    for g in FEET:
        m.geom_solref[g, 0] = p["solref"]
        m.geom_size[g, 1] = p["foot_width"] / 2000  # local y = world y (half width)
    m.actuator_gainprm[:, 0] *= p["kp"]
    m.actuator_biasprm[:, 1] *= p["kp"]
    m.actuator_gainprm[HIP_LAT, 0] *= p["kp_hip_lat"]
    m.actuator_biasprm[HIP_LAT, 1] *= p["kp_hip_lat"]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    act = [m.actuator(j).id for j in JOINTS]
    d.qpos[qadr] = run["q"][0]
    d.ctrl[act] = run["q"][0]
    mujoco.mj_forward(m, d)
    for _ in range(250):
        mujoco.mj_step(m, d)
    imu = m.site("imu").id
    ga = m.sensor_adr[m.sensor("gyro").id]
    rp, gyro, fell = [], [], None
    for k, tgt in enumerate(run["target"]):
        g = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        if g[2] > -0.5:
            fell = k
            break
        rp.append(angles(g[None])[0])
        gyro.append(d.sensordata[ga:ga + 3].copy())
        d.ctrl[act] = tgt
        for _ in range(10):
            mujoco.mj_step(m, d)
    return np.array(rp), np.array(gyro), fell


def feats(rp, gyro, phase):
    sl = slice(SKIP, len(rp))
    ph = np.arctan2(phase[sl, 2], phase[sl, 0])
    x = np.c_[rp[sl], gyro[sl]]
    X = np.stack([np.ones_like(ph), np.cos(ph), np.sin(ph)], 1)
    c, *_ = np.linalg.lstsq(X, x, rcond=None)
    return {
        "roll_amp": np.degrees(np.hypot(c[1, 0], c[2, 0])),
        "roll_phase": np.degrees(np.arctan2(c[2, 0], c[1, 0])),
        "gyro_x_amp": np.hypot(c[1, 2], c[2, 2]),
        "pitch_mean": np.degrees(rp[sl, 1].mean()),
    }


def evaluate(runs, p):
    out = []
    for run in runs:
        rp, gyro, fell = replay(run, p)
        if len(rp) < SKIP + 50:
            out.append(None)
            continue
        out.append({**feats(rp, gyro, run["phase"][:len(rp)]), "fell": fell, "rp": rp, "gyro": gyro})
    return out


def mean_feat(res, k):
    v = [r[k] for r in res if r is not None]
    return np.mean(v) if v else np.nan


SIM_C = ["#2a78d6", "#1baf7a"]  # base dx 0, base dx +40
ROBOT, INK, MUTED, GRID = "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dd"


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    out = sys.argv[2]
    runs = [load_run(p) for p in RUNS]
    robot = [feats(r["rp"], r["gyro"], r["phase"]) for r in runs]
    rob = {k: (min(f[k] for f in robot), max(f[k] for f in robot)) for k in robot[0]}
    print("robot:", {k: tuple(round(x, 2) for x in v) for k, v in rob.items()})

    bases = {"COM shift 0 (main's model)": dict(DEFAULTS), "COM shift +40 mm": {**DEFAULTS, "dx": 0.04}}
    sens = {}
    for (bname, base), (pname, (vals, _)) in itertools.product(bases.items(), SWEEPS.items()):
        if pname == "dx" and bname != "COM shift 0 (main's model)":
            continue
        rows = []
        for v in vals:
            res = evaluate(runs, {**base, pname: v})
            rows.append({k: mean_feat(res, k) for k in ("roll_amp", "roll_phase", "gyro_x_amp", "pitch_mean")}
                        | {"falls": sum(r is None or r["fell"] is not None for r in res)})
            print(f"{bname:28s} {pname:11s} {v:7.3f}  " + " ".join(f"{k} {x:6.2f}" for k, x in rows[-1].items()), flush=True)
        sens[(bname, pname)] = rows

    # Small combined grid of the parameters that move roll the most.
    best = None
    for dx, kpl, sr, fw in itertools.product([0.02, 0.04], [0.3, 0.5, 1.0], [0.02, 0.04, 0.08], [12.0, 20.0, 34.0]):
        p = {**DEFAULTS, "dx": dx, "kp_hip_lat": kpl, "solref": sr, "foot_width": fw}
        res = evaluate(runs, p)
        if any(r is None or r["fell"] is not None for r in res):
            continue
        cost = sum(((mean_feat(res, k) - np.mean([f[k] for f in robot])) / s) ** 2
                   for k, s in (("roll_amp", 1.0), ("gyro_x_amp", 0.1), ("pitch_mean", 2.0)))
        if best is None or cost < best[0]:
            best = (cost, p, res)
    print("\nbest combined:", {k: v for k, v in best[1].items()}, "cost", round(best[0], 2),
          {k: round(mean_feat(best[2], k), 2) for k in ("roll_amp", "roll_phase", "gyro_x_amp", "pitch_mean")})

    # --- sensitivity plot ---
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK})
    feats_plot = [("roll_amp", "roll at gait clock (deg)"), ("gyro_x_amp", "gyro x at gait clock (rad/s)"),
                  ("pitch_mean", "mean pitch (deg, + nose down)")]
    fig, axes = plt.subplots(len(feats_plot), len(SWEEPS), figsize=(2.6 * len(SWEEPS), 7.2), sharey="row")
    for j, (pname, (vals, label)) in enumerate(SWEEPS.items()):
        for i, (fk, flabel) in enumerate(feats_plot):
            ax = axes[i, j]
            style(ax)
            ax.axhspan(*rob[fk], color=ROBOT, alpha=0.18, lw=0)
            ax.axhline(np.mean(rob[fk]), color=ROBOT, lw=1.5, label="robot (2 runs)" if (i, j) == (0, 0) else None)
            for c, bname in zip(SIM_C, bases):
                rows = sens.get((bname, pname))
                if rows is None:
                    continue
                y = [r[fk] for r in rows]
                ax.plot(vals, y, color=c, lw=2, marker="o", ms=5, label=f"sim replay, {bname}" if (i, j) == (0, 0) else None)
                for v, r, yy in zip(vals, rows, y):
                    if r["falls"]:
                        ax.plot(v, yy, "x", color=INK, ms=7)
            if j == 0:
                ax.set_ylabel(flabel)
            if i == len(feats_plot) - 1:
                ax.set_xlabel(label)
    for ax in axes[0]:
        ax.set_ylim(0, None)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ROBOT, lw=1.5, label="robot (2 runs; band = range)")]
    handles += [Line2D([], [], color=c, lw=2, marker="o", label=f"sim replay, base: {b}") for c, b in zip(SIM_C, bases)]
    handles += [Line2D([], [], color=INK, marker="x", ls="", label="sim fell")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("x0.3 open-loop replay: what moves the sim's rocking toward the robot's "
                 "(one parameter at a time; x = fell)", x=0.01, y=1.04, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/replay_sensitivity.png", dpi=120, facecolor="white", bbox_inches="tight")

    # --- time series ---
    cur = evaluate(runs, DEFAULTS)
    run, r0, rb = runs[0], cur[0], best[2][0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = np.arange(len(run["rp"])) * DT
    p = best[1]
    best_label = (f"sim replay, best: COM x +{1000 * p['dx']:.0f} mm, hip_lat kp x{p['kp_hip_lat']}, "
                  f"solref {p['solref']}, foot width {p['foot_width']:.0f} mm")
    for ax, (y_rob, y_cur, y_best, lab) in zip(axes, [
            (np.degrees(run["rp"][:, 0]), np.degrees(r0["rp"][:, 0]), np.degrees(rb["rp"][:, 0]), "roll (deg)"),
            (np.degrees(run["rp"][:, 1]), np.degrees(r0["rp"][:, 1]), np.degrees(rb["rp"][:, 1]), "pitch (deg, + nose down)"),
            (run["gyro"][:, 0], r0["gyro"][:, 0], rb["gyro"][:, 0], "gyro x (rad/s)")]):
        style(ax)
        ax.plot(t, y_rob, color=ROBOT, lw=2, label="robot (reconstructed true tilt; measured gyro)")
        ax.plot(t[:len(y_cur)], y_cur, color=SIM_C[0], lw=1.6, label="sim replay, main's model")
        ax.plot(t[:len(y_best)], y_best, color=SIM_C[1], lw=1.6, label=best_label)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel(lab)
    axes[0].set_xlim(1, 6)
    axes[0].legend(loc="upper left", ncol=1, frameon=False, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("cat-fire-potential x0.3, run r1-small: robot targets replayed open loop in the sim",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/replay_timeseries.png", dpi=120, facecolor="white")

    # --- roll over one gait cycle, averaged ---
    from floor_analysis import FK
    fk = FK()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    bins = np.linspace(-np.pi, np.pi, 25)
    centers = np.degrees(0.5 * (bins[1:] + bins[:-1]))
    for ax, run, res, name in zip(axes, runs, cur, ("r1-small", "r2-small")):
        style(ax)
        n = len(res["rp"])
        sl = slice(SKIP, n)
        ph = np.arctan2(run["phase"][sl, 2], run["phase"][sl, 0])
        idx = np.digitize(ph, bins) - 1
        fz = fk.feet_z(run["q"][sl])
        kin = -np.arcsin(np.clip((fz[:, 0] - fz[:, 1]) / 0.165, -1, 1))
        series = [
            (np.degrees(run["filt_rp"][sl, 0]), ROBOT, ":", "robot: filter roll (policy input)"),
            (np.degrees(run["rp"][sl, 0]), ROBOT, "-", "robot: true roll (reconstructed)"),
            (np.degrees(res["rp"][:, 0][sl]), SIM_C[0], "-", "sim replay, main's model"),
            (np.degrees(kin), MUTED, "--", "both feet flat: roll from the measured leg lengths"),
        ]
        for y, c, ls, lab in series:
            yb = np.array([y[idx == b].mean() - y.mean() for b in range(len(centers))])
            ax.plot(centers, yb, color=c, ls=ls, lw=2, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("gait phase, left leg (deg)")
        ax.set_title(f"cat-fire-potential x0.3, {name}", loc="left", fontsize=10)
    axes[0].set_ylabel("roll, cycle average minus mean (deg)\n+ = left side up")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("Roll over one gait cycle: the robot rocks opposite to the sim and to the leg geometry",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/replay_cycle.png", dpi=120, facecolor="white")
    print("saved plots")


if __name__ == "__main__":
    main()

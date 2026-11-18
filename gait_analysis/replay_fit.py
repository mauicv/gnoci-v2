"""Open-loop replay of the x0.3 floor runs: sweep torso COM, friction and kp.

Plays the robot's recorded targets into the free-floating sim (no policy, no
filter) and compares gait-frequency features of the sim's true tilt and gyro
with the robot's reconstructed tilt and gyro.

  uv run python replay_fit.py <scratchpad dir> <out dir>
"""
import itertools
import json
import sys

sys.path.insert(0, sys.argv[1])
import mujoco
import numpy as np

from floor_analysis import JOINTS, SCENE, SIM_DT, angles, clock_fit, reconstruct

RUNS = ["logs/floor/cat-fire-potential/r1-small.npz", "logs/floor/cat-fire-potential/r2-small.npz"]
SKIP = 50          # steps left out at the start (1 s)
END_TRIM = 75      # steps left out before the end (the robot's fall)
TRAJ_STEPS = 150   # steps used for the trajectory error (3 s after SKIP)

DX = [-0.04, -0.02, 0.0, 0.02, 0.04, 0.06]   # torso COM shift from the current model (m)
DY = [-0.01, 0.0, 0.01]
DZ = [-0.03, 0.0, 0.03]
FRICTION = [0.5, 1.0]
KP = [0.8, 1.0]


def load_run(path):
    d = np.load(path)
    o = d["obs"]
    n = len(o) - END_TRIM
    ref = reconstruct(angles(d["gravity"]), o[:, :3])
    return {"q": o[:n, 9:15], "gyro": o[:n, :3], "phase": o[:n, 27:31], "target": d["target"][:n],
            "rp": ref[:n], "filt_rp": angles(d["gravity"])[:n]}


BASE = mujoco.MjModel.from_xml_path(str(SCENE))
TORSO = BASE.body("body_main").id
BASE_IPOS = BASE.body_ipos[TORSO].copy()
FOOT_GEOMS = [BASE.geom(n).id for n in ("left_foot_base", "right_foot_base")]
FLOOR = BASE.geom("floor").id


def replay(run, dx=0.0, dy=0.0, dz=0.0, friction=1.0, kp=1.0):
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = SIM_DT
    m.body_ipos[TORSO] = BASE_IPOS + [dx, dy, dz]
    for g in FOOT_GEOMS + [FLOOR]:
        m.geom_friction[g, 0] = friction
    m.actuator_gainprm[:, 0] *= kp
    m.actuator_biasprm[:, 1] *= kp
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    act = [m.actuator(j).id for j in JOINTS]
    d.qpos[qadr] = run["q"][0]
    d.ctrl[act] = run["q"][0]
    mujoco.mj_forward(m, d)
    # Settle on the floor for 0.5 s holding the first pose.
    for _ in range(250):
        mujoco.mj_step(m, d)
    imu = m.site("imu").id
    ga = m.sensor_adr[m.sensor("gyro").id]
    rp, gyro, q = [], [], []
    fell = None
    for k, tgt in enumerate(run["target"]):
        g = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        if g[2] > -0.5:
            fell = k
            break
        rp.append(angles(g[None])[0])
        gyro.append(d.sensordata[ga:ga + 3].copy())
        q.append(d.qpos[qadr].copy())
        d.ctrl[act] = tgt
        for _ in range(10):
            mujoco.mj_step(m, d)
    return {"rp": np.array(rp), "gyro": np.array(gyro), "q": np.array(q), "fell": fell}


def features(rp, gyro, phase):
    sl = slice(SKIP, len(rp))
    a1, a2, _ = clock_fit(np.c_[rp[sl], gyro[sl]], phase[sl])
    return {
        "roll_amp": np.degrees(a1[0]),        # roll at the gait clock (deg)
        "pitch_amp": np.degrees(a1[1]),
        "pitch_mean": np.degrees(rp[sl, 1].mean()),
        "roll_mean": np.degrees(rp[sl, 0].mean()),
        "gyro_x_amp": a1[2],                  # rad/s at the clock
        "gyro_z_rms": np.sqrt(np.mean(gyro[sl, 2] ** 2)),
    }


# How each feature's error is scaled in the cost (rough measurement spread).
SCALE = {"roll_amp": 1.0, "pitch_amp": 1.0, "pitch_mean": 2.0, "gyro_x_amp": 0.1}


def main():
    out = sys.argv[2]
    runs = [load_run(p) for p in RUNS]
    robot = [features(r["rp"], r["gyro"], r["phase"]) for r in runs]
    print("robot:", [{k: round(v, 3) for k, v in f.items()} for f in robot])
    results = []
    grid = list(itertools.product(DX, DY, DZ, FRICTION, KP))
    for i, (dx, dy, dz, fr, kp) in enumerate(grid):
        per_run = []
        for run, rf in zip(runs, robot):
            s = replay(run, dx, dy, dz, fr, kp)
            n = len(s["rp"])
            if n < SKIP + 50:
                per_run.append({"fell": s["fell"], "cost": np.inf})
                continue
            f = features(s["rp"], s["gyro"], run["phase"][:n])
            tr = slice(SKIP, min(n, SKIP + TRAJ_STEPS))
            traj = np.degrees(np.sqrt(np.mean((s["rp"][tr] - run["rp"][tr]) ** 2, 0)))
            qerr = np.sqrt(np.mean((s["q"][SKIP:] - run["q"][SKIP:n]) ** 2))
            cost = sum(((f[k] - rf[k]) / SCALE[k]) ** 2 for k in SCALE) + (s["fell"] is not None) * 100
            per_run.append({**f, "fell": s["fell"], "traj_roll": traj[0], "traj_pitch": traj[1],
                            "q_rms": qerr, "cost": cost})
        results.append({"dx": dx, "dy": dy, "dz": dz, "friction": fr, "kp": kp, "runs": per_run,
                        "cost": float(np.mean([r["cost"] for r in per_run]))})
        if i % 20 == 0:
            print(f"{i}/{len(grid)}", flush=True)
    results.sort(key=lambda r: r["cost"])
    print("\nbest 12 (torso COM shift from current model, mm):")
    for r in results[:12]:
        f = r["runs"]
        print(f"dx {1000*r['dx']:+4.0f} dy {1000*r['dy']:+3.0f} dz {1000*r['dz']:+3.0f} fr {r['friction']} kp {r['kp']}"
              f"  cost {r['cost']:6.2f}  roll_amp {[round(x.get('roll_amp', np.nan), 2) for x in f]}"
              f" pitch_mean {[round(x.get('pitch_mean', np.nan), 1) for x in f]}"
              f" gyro_x_amp {[round(x.get('gyro_x_amp', np.nan), 2) for x in f]}"
              f" traj r/p {[(round(x.get('traj_roll', np.nan), 1), round(x.get('traj_pitch', np.nan), 1)) for x in f]}"
              f" q_rms {[round(x.get('q_rms', np.nan), 3) for x in f]} fell {[x['fell'] for x in f]}")
    base = next(r for r in results if r["dx"] == 0 and r["dy"] == 0 and r["dz"] == 0 and r["friction"] == 1.0 and r["kp"] == 1.0)
    print(f"\ncurrent model: cost {base['cost']:.2f}", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for x in base["runs"]])
    with open(f"{out}/replay_fit.json", "w") as fh:
        json.dump({"robot": robot, "results": results}, fh, default=float)


if __name__ == "__main__":
    main()

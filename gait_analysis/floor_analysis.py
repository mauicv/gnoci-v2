"""Analyse logs/floor recordings (safe-throne-deal, cat-fire-potential) vs the sim.

1. Filter tilt error: the robot's complementary filter (tau 0.163 s) mixes the
   accelerometer's own motion into the tilt. True tilt isn't recorded, so it's
   reconstructed offline: ref = integral(gyro) + zero-phase low-pass(filter -
   integral(gyro)), cutoff FC Hz. The same reconstruction is checked on sim
   rollouts against the true tilt.
2. Gait: clock-locked amplitude per joint, and foot lift from forward
   kinematics of the measured joint angles.
"""
import glob
import sys

import mujoco
import numpy as np
from scipy.signal import butter, filtfilt

from gnoci_v2.control.filter import ComplementaryFilter
from gnoci_v2.control.predict import PolicyRunner
from gnoci_v2.training.constants import SIM_PATH

SCENE = SIM_PATH / "desc" / "scene.xml"
DT, SIM_DT = 0.02, 0.002
JOINTS = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]
FC = 0.3
TAU = 0.98 * (1 / 300) / 0.02


def angles(g):
    roll = np.arctan2(-g[:, 1], -g[:, 2])
    pitch = np.arcsin(np.clip(g[:, 0], -1, 1))
    return np.stack([roll, pitch], 1)


def reconstruct(filt_rp, gyro):
    """Reference roll/pitch (rad) from filter angles and 50 Hz gyro samples."""
    w = gyro[:, :2]
    integ = np.concatenate([[np.zeros(2)], np.cumsum(0.5 * (w[1:] + w[:-1]) * DT, 0)])
    diff = filt_rp - integ
    if len(diff) > 150:
        b, a = butter(2, FC / 25)
        low = filtfilt(b, a, diff, axis=0, padlen=min(150, len(diff) - 1))
    else:  # Too short to low-pass: only remove the starting offset.
        low = np.repeat(diff[:5].mean(0, keepdims=True), len(diff), 0)
    return integ + low


def rms(x):
    return np.sqrt(np.mean(x ** 2, 0))


def clock_fit(x, phase_obs):
    """Amplitude of x at the clock (left leg phase) and its 2nd harmonic; var share explained."""
    ph = np.arctan2(phase_obs[:, 2], phase_obs[:, 0])
    X = np.stack([np.ones_like(ph), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)], 1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    a1 = np.hypot(coef[1], coef[2])
    a2 = np.hypot(coef[3], coef[4])
    r2 = 1 - ((x - X @ coef) ** 2).sum(0) / ((x - x.mean(0)) ** 2).sum(0)
    return a1, a2, r2


class FK:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS]
        self.sites = [self.m.site("l_foot").id, self.m.site("r_foot").id]

    def feet_z(self, q, rp=None):
        """Foot site z (m) relative to the torso, in the world frame if roll/pitch given."""
        out = []
        for k, qk in enumerate(q):
            self.d.qpos[:] = 0
            self.d.qpos[3] = 1
            if rp is not None:
                r, p = rp[k]
                quat = np.zeros(4)
                mujoco.mju_euler2Quat(quat, np.array([r, p, 0.0]), "XYZ")
                self.d.qpos[3:7] = quat
            self.d.qpos[self.qadr] = qk
            mujoco.mj_kinematics(self.m, self.d)
            out.append(self.d.site_xpos[self.sites, 2].copy())
        return np.array(out)


def lift_stats(fz):
    """Per-step swing lift: how far the higher foot is above the lower one."""
    dz = fz[:, 0] - fz[:, 1]
    return np.percentile(np.abs(dz), 95), np.abs(dz).mean()


def sim_rollout(policy, scale_mult, steps, seed, feed="filter"):
    """Closed loop like the robot: policy sees the filter (300 Hz, tau 0.163 s) gravity."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    rng = np.random.default_rng(seed)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    dadr = [m.jnt_dofadr[m.joint(j).id] for j in JOINTS]
    act = [m.actuator(j).id for j in JOINTS]
    d.qpos[qadr] = rng.uniform(-0.02, 0.02, 6)
    d.ctrl[act] = d.qpos[qadr]
    mujoco.mj_forward(m, d)
    imu = m.site("imu").id
    ga = m.sensor_adr[m.sensor("gyro").id]
    aa = m.sensor_adr[m.sensor("accelerometer").id]
    sid = [m.site("l_foot").id, m.site("r_foot").id]
    alpha = TAU / (TAU + SIM_DT)
    filt = ComplementaryFilter(alpha=alpha, dt=SIM_DT)
    filt.update(d.sensordata[aa:aa + 3], d.sensordata[ga:ga + 3])
    phase = np.array([0.0, np.pi])
    last = np.zeros(6)
    rec = {k: [] for k in ("true", "filt", "gyro", "q", "phase", "footz", "action")}
    fell = None
    for k in range(steps):
        gt = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        gf = filt.gravity
        if gt[2] > -0.5 or d.qpos[2] < 0.15:
            fell = k
            break
        gyro = d.sensordata[ga:ga + 3].copy()
        ph = np.concatenate([np.cos(phase), np.sin(phase)])
        obs = np.concatenate([gyro, gf if feed == "filter" else gt, np.zeros(3), d.qpos[qadr],
                              d.qvel[dadr], last, ph])
        a = policy.predict(obs)
        last = a
        for key, v in (("true", gt), ("filt", gf), ("gyro", gyro), ("q", d.qpos[qadr].copy()),
                       ("phase", ph), ("footz", d.site_xpos[sid, 2].copy()), ("action", a)):
            rec[key].append(v)
        phase = np.fmod(phase + 2 * np.pi * 1.375 * DT + np.pi, 2 * np.pi) - np.pi
        d.ctrl[act] = scale_mult * policy.action_scale * a
        for _ in range(10):
            mujoco.mj_step(m, d)
            filt.update(d.sensordata[aa:aa + 3], d.sensordata[ga:ga + 3])
    return {k: np.array(v) for k, v in rec.items()}, fell


def main():
    fk = FK()
    deg = np.degrees
    print("=== REAL ===")
    print(f"{'run':34s} {'T':>5s} {'filt tilt std r/p':>18s} {'est filt err rms r/p':>21s}"
          f" {'gyro rms x/y/z':>18s}")
    real = {}
    for f in sorted(glob.glob("logs/floor/*/r*.npz")):
        d = np.load(f)
        name = f.split("floor/")[1][:-4] + (" (x0.3)" if abs(d["target"][np.abs(d["action"]) > .05].ravel()[0] / d["action"][np.abs(d["action"]) > .05].ravel()[0] - 0.15) < 0.01 else "")
        o = d["obs"]
        rp = angles(d["gravity"])
        ref = reconstruct(rp, o[:, :3])
        # Leave out the last 0.3 s (the fall itself) and the first 0.5 s.
        sl = slice(25, max(26, len(o) - 15))
        err = rp[sl] - ref[sl]
        real[name] = (o, rp, ref, d)
        print(f"{name:34s} {d['t'][-1]:5.1f} {deg(rp[sl].std(0)[0]):8.2f} {deg(rp[sl].std(0)[1]):8.2f}"
              f" {deg(rms(err))[0]:10.2f} {deg(rms(err))[1]:9.2f}  "
              f"{' '.join(f'{v:5.2f}' for v in rms(o[sl, :3]))}")

    print("\n=== SIM (cat-fire-potential / safe-throne-deal, zero command, policy fed the filter) ===")
    print(f"{'case':34s} {'fell':>5s} {'true err rms r/p':>18s} {'est err rms r/p':>17s}"
          f" {'gyro rms x/y/z':>18s} {'true tilt std r/p':>18s}")
    sims = {}
    for model, mult in (("cat-fire-potential", 1.0), ("cat-fire-potential", 0.3), ("safe-throne-deal", 1.0)):
        pol = PolicyRunner(model)
        for seed in range(3):
            r, fell = sim_rollout(pol, mult, 500, seed)
            if len(r["true"]) < 60:
                print(f"{model} x{mult} seed {seed}: fell at {fell}")
                continue
            tr, fr = angles(r["true"]), angles(r["filt"])
            ref = reconstruct(fr, r["gyro"])
            sl = slice(25, len(tr) - 15)
            sims[(model, mult, seed)] = (r, fell)
            print(f"{model + f' x{mult} s{seed}':34s} {str(fell):>5s} {deg(rms(fr[sl] - tr[sl]))[0]:8.2f} {deg(rms(fr[sl] - tr[sl]))[1]:8.2f}"
                  f" {deg(rms(fr[sl] - ref[sl]))[0]:8.2f} {deg(rms(fr[sl] - ref[sl]))[1]:8.2f}  "
                  f"{' '.join(f'{v:5.2f}' for v in rms(r['gyro'][sl]))} "
                  f"{deg(tr[sl].std(0))[0]:8.2f} {deg(tr[sl].std(0))[1]:8.2f}")

    print("\n=== GAIT: clock fit (amplitude rad at 1.375 Hz | 2.75 Hz, R^2) ===")
    def gait_row(name, q, phase_obs, rp=None, footz_true=None):
        a1, a2, r2 = clock_fit(q, phase_obs)
        fz = fk.feet_z(q, rp)
        p95, mean = lift_stats(fz)
        s = f"{name:34s} " + " ".join(f"{x:.3f}|{y:.3f}({z:.2f})" for x, y, z in zip(a1, a2, r2))
        s += f"  FK lift p95 {1000 * p95:4.1f} mm mean {1000 * mean:4.1f}"
        if footz_true is not None:
            h = footz_true - footz_true.min(1, keepdims=True)
            s += f" | sim true foot clearance p95 {1000 * np.percentile(h.max(1), 95):4.1f} mm"
        print(s)
    print(f"{'':34s} " + "  ".join(f"{j:>15s}" for j in JOINTS))
    for name, (o, rp, ref, d) in real.items():
        sl = slice(25, max(26, len(o) - 15))
        gait_row(name, o[sl, 9:15], o[sl, 27:31], ref[sl])
    for (model, mult, seed), (r, fell) in sims.items():
        if seed:
            continue
        sl = slice(25, len(r["q"]) - 15)
        gait_row(f"SIM {model} x{mult}", r["q"][sl], r["phase"][sl], angles(r["true"])[sl], r["footz"][sl])


if __name__ == "__main__":
    main()

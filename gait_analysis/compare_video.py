"""Side-by-side video: sim (physics) vs a reconstruction of the real robot.

The reconstruction is kinematic, not simulated: a copy of the model posed each
step with the robot's measured joint angles and its reconstructed torso tilt
(roll/pitch from gyro + low-passed accelerometer, yaw from the integrated
gyro), lowered until its lowest foot touches the floor. Horizontal position
isn't measured, so it stays put.

  tilt   the tilt-release test: servos holding home, released from the
         robot's held angle; sim released from the same angle.
  walk   an x0.3 rollout: the robot's recorded targets replayed open loop
         in the sim.

  uv run python compare_video.py tilt --release 11.3
  uv run python compare_video.py walk --run logs/floor/cat-fire-potential/raw-accel-tests/r1-small.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import imageio_ffmpeg
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import butter, filtfilt

from floor_analysis import DT, JOINTS, SCENE, SIM_DT, angles

TILT_REC = "logs/floor/cat-fire-potential/raw-accel-tests/tilt-test.npz"


def true_tilt(rec):
    """(roll, pitch, yaw) per step: gyro integral, with roll/pitch's slow part from the accelerometer."""
    o = rec["obs"]
    gyro = o[:, :3]
    integ = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * DT, 0)])
    if "accel" in rec:
        a = rec["accel"]
        slow = np.stack([np.arctan2(a[:, 1], a[:, 2]), np.arctan2(-a[:, 0], np.hypot(a[:, 1], a[:, 2]))], 1)
    else:
        slow = angles(rec["gravity"])
    b, c = butter(2, 0.3 / 25)
    rp = integ[:, :2] + filtfilt(b, c, slow - integ[:, :2], axis=0)
    return np.c_[rp, integ[:, 2] - integ[0, 2]]


def release_index(rec, t_release):
    """First step of the release near t_release: where gyro x starts rising to its peak."""
    t, gx = rec["t"], rec["obs"][:, 0]
    p = int(np.argmin(np.abs(t - t_release - 0.3)))
    p = p - 15 + int(np.argmax(gx[p - 15:p + 16]))
    while p > 0 and gx[p] > 0.03:
        p -= 1
    return p


class Model:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.m.opt.timestep = SIM_DT
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS]
        self.act = [self.m.actuator(j).id for j in JOINTS]
        self.feet = [self.m.geom(n).id for n in ("left_foot_base", "right_foot_base")]
        self.imu = self.m.site("imu").id
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)

    def lowest_foot_z(self):
        z = []
        for g in self.feet:
            R = self.d.geom_xmat[g].reshape(3, 3)
            c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * self.m.geom_size[g]
            z.append((self.d.geom_xpos[g] + c @ R.T)[:, 2].min())
        return min(z)

    def roll(self):
        g = self.d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        return np.degrees(angles(g[None])[0, 0])

    def pose(self, q, rpy):
        """Kinematic pose: joints q, torso roll/pitch/yaw, lowest foot on the floor."""
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([rpy[2], rpy[1], rpy[0]]), "zyx")
        self.d.qpos[3:7] = quat
        self.d.qpos[self.qadr] = q
        self.d.qpos[2] = 0.3
        mujoco.mj_kinematics(self.m, self.d)
        self.d.qpos[2] -= self.lowest_foot_z()
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)  # also the subtree COM the tracking camera follows

    def tilt_about_left_edge(self, roll0, q):
        """Physics start for the tilt test: rotated about the left foot's outer bottom edge, at rest."""
        self.d.qpos[self.qadr] = q
        self.d.ctrl[self.act] = q
        mujoco.mj_forward(self.m, self.d)
        g = self.feet[0]
        R = self.d.geom_xmat[g].reshape(3, 3)
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * self.m.geom_size[g]
        w = self.d.geom_xpos[g] + c @ R.T
        pivot = np.array([0.0, w[:, 1].max(), w[:, 2].min()])
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([1.0, 0, 0]), roll0)
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, quat)
        self.d.qpos[:3] = pivot + Rr.reshape(3, 3) @ (self.d.qpos[:3] - pivot)
        self.d.qpos[3:7] = quat
        self.d.qpos[2] += 0.0005
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def step(self, target=None):
        if target is not None:
            self.d.ctrl[self.act] = target
        for _ in range(int(round(DT / SIM_DT))):
            mujoco.mj_step(self.m, self.d)


def camera(body, azimuth):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED if body is None else mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body
    cam.distance, cam.azimuth, cam.elevation = 0.8, azimuth, -8.0
    return cam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tilt", "walk"])
    ap.add_argument("--release", type=float, default=11.3, help="tilt: time of the release in the recording (s)")
    ap.add_argument("--run", default="logs/floor/cat-fire-potential/raw-accel-tests/r1-small.npz")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed")
    ap.add_argument("--azimuth", type=float, default=180.0, help="180 = from the front")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = TILT_REC if args.mode == "tilt" else args.run
    rec = dict(np.load(path))
    tilt = true_tilt(rec)
    q_real = rec["obs"][:, 9:15]
    sim, real = Model(), Model()

    if args.mode == "tilt":
        i0 = release_index(rec, args.release)
        steps = int((args.seconds or 3.0) / DT)
        sim.tilt_about_left_edge(tilt[i0, 0], q_real[i0])
        targets = [None] * steps
        title = f"tilt-release from {np.degrees(tilt[i0, 0]):.0f} deg, servos holding home"
        out = args.out or f"videos/compare_tilt_{args.release:g}s.mp4"
    else:
        i0 = 0
        steps = int((args.seconds or 6.0) / DT)
        sim.d.qpos[sim.qadr] = q_real[0]
        sim.d.ctrl[sim.act] = q_real[0]
        mujoco.mj_forward(sim.m, sim.d)
        for _ in range(250):
            mujoco.mj_step(sim.m, sim.d)
        targets = list(rec["target"])
        title = f"{Path(path).parent.name}/{Path(path).stem}: robot targets replayed open loop"
        out = args.out or f"videos/compare_walk_{Path(path).stem}.mp4"
    steps = min(steps, len(q_real) - i0)

    torso = sim.m.body("body_main").id
    renderer = mujoco.Renderer(sim.m, height=480, width=480)
    frames = []
    for k in range(steps):
        i = i0 + k
        real.pose(q_real[i], tilt[i] - [0, 0, tilt[i0, 2]])
        views = []
        for mdl in (sim, real):
            renderer.update_scene(mdl.d, camera=camera(torso, args.azimuth))
            views.append(renderer.render())
        img = Image.fromarray(np.concatenate(views, axis=1))
        dr = ImageDraw.Draw(img)
        dr.text((10, 8), title, fill="white")
        dr.text((10, 24), f"t {k * DT:4.2f} s   x{args.speed:g} speed   front view", fill="white")
        dr.text((10, 452), f"SIM (physics)   roll {sim.roll():+5.1f} deg", fill=(120, 180, 255))
        dr.text((490, 452), f"REAL (reconstructed pose)   roll {np.degrees(tilt[i, 0]):+5.1f} deg",
                fill=(255, 150, 90))
        frames.append(np.asarray(img))
        sim.step(targets[k] if k < len(targets) and targets[k] is not None else None)
        if k + i0 + 1 >= len(q_real):
            break

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(out, frames, fps=args.speed / DT)
    print(f"wrote {len(frames)} frames ({len(frames) * DT:.1f} s of motion) to {out}")


if __name__ == "__main__":
    main()

"""Video of a robot recording's joint targets replayed open loop in the sim.

No policy and no filter: the sim gets the robot's recorded targets each step.
Renders a front view (to see roll) beside the tracking camera, with each
foot's height above the floor and the sim's vs the robot's roll overlaid.
Also prints how often each foot is off the floor.

  uv run python replay_video.py
  uv run python replay_video.py --run logs/floor/cat-fire-potential/r2-small.npz --dx 0.02 --speed 0.5
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

from floor_analysis import DT, JOINTS, SCENE, SIM_DT, angles
from replay_fit import load_run

LIFT_MM = 2.0  # A foot counts as off the floor above this.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="logs/floor/cat-fire-potential/r1-small.npz")
    ap.add_argument("--dx", type=float, default=0.0, help="torso COM x shift (m)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = half speed)")
    ap.add_argument("--out", default="videos/replay.mp4")
    args = ap.parse_args()

    run = load_run(args.run)
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.opt.timestep = SIM_DT
    torso = m.body("body_main").id
    m.body_ipos[torso] += [args.dx, 0, 0]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    act = [m.actuator(j).id for j in JOINTS]
    d.qpos[qadr] = run["q"][0]
    d.ctrl[act] = run["q"][0]
    mujoco.mj_forward(m, d)
    for _ in range(250):  # settle holding the first pose
        mujoco.mj_step(m, d)

    imu = m.site("imu").id
    feet = [m.geom(n).id for n in ("left_foot_base", "right_foot_base")]

    def foot_heights_mm():
        """Lowest corner of each foot box above the floor."""
        out = []
        for g in feet:
            R = d.geom_xmat[g].reshape(3, 3)
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
            out.append(1000 * (d.geom_xpos[g][2] + (corners @ R.T)[:, 2].min()))
        return out

    front = mujoco.MjvCamera()
    front.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    front.trackbodyid = torso
    front.distance, front.azimuth, front.elevation = 0.75, 180.0, -8.0  # camera in front, looking back at the robot
    renderer = mujoco.Renderer(m, height=480, width=480)

    steps = min(len(run["target"]), int(args.seconds / DT))
    frames, heights = [], []
    for k in range(steps):
        g = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        if g[2] > -0.5:
            print(f"sim fell at step {k}")
            break
        sim_roll = np.degrees(angles(g[None])[0, 0])
        h = foot_heights_mm()
        heights.append(h)
        if True:
            views = []
            for cam in (front, "track"):
                renderer.update_scene(d, camera=cam)
                views.append(renderer.render())
            img = Image.fromarray(np.concatenate(views, axis=1))
            dr = ImageDraw.Draw(img)
            dr.text((10, 10), f"{Path(args.run).parent.name}/{Path(args.run).stem}  open-loop replay  "
                              f"t {k * DT:4.2f} s  (COM x {1000 * args.dx:+.0f} mm, x{args.speed:g} speed)", fill="white")
            dr.text((10, 28), "front view (robot's left on the right)", fill="white")
            for i, (name, hh) in enumerate(zip(("left foot", "right foot"), h)):
                col = (255, 210, 80) if hh > LIFT_MM else (200, 200, 200)
                dr.text((10, 440 + 16 * i - 16), f"{name}: {hh:5.1f} mm above floor", fill=col)
            dr.text((490, 424), f"roll  sim   {sim_roll:+5.1f} deg", fill=(120, 180, 255))
            dr.text((490, 440), f"roll  robot {np.degrees(run['rp'][k, 0]):+5.1f} deg (reconstructed)", fill=(255, 150, 90))
            frames.append(np.asarray(img))
        d.ctrl[act] = run["target"][k]
        for _ in range(10):
            mujoco.mj_step(m, d)

    h = np.array(heights)[50:]
    for i, name in enumerate(("left", "right")):
        print(f"{name} foot: off the floor (> {LIFT_MM} mm) {100 * (h[:, i] > LIFT_MM).mean():4.1f}% of steps, "
              f"max {h[:, i].max():.1f} mm, 95th pct {np.percentile(h[:, i], 95):.1f} mm")
    print(f"both feet down {100 * (h.max(1) <= LIFT_MM).mean():.1f}% of steps")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    mediapy.write_video(args.out, frames, fps=args.speed / DT)
    print(f"wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()

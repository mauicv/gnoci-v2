"""Show the real robot's pose live in the MuJoCo viewer.

The Pi streams each control step's observation, and this script writes the
joint angles and the torso tilt into qpos. No physics is simulated: it calls
mj_forward, which only computes positions, so the sim shows exactly what the
robot reports. Use it to check joint signs, joint limits and the IMU axes.

On the Pi (--torque-off lets you move the legs by hand):
  uv run --group control gnoci run --stream --torque-off
On the laptop:
  uv run python scripts/live_view.py --host <pi-ip>

The IMU gives no yaw (heading), so the sim always faces +x. The torso position
isn't measured either, so it's pinned at --height.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from gnoci_v2.networking.client import Client
from gnoci_v2.training.constants import SIM_PATH

SCENE = SIM_PATH / "desc" / "scene.xml"

# Slices of the policy observation (see Main.get_observation).
ACCEL = slice(3, 6)
JOINT_POS = slice(9, 15)
OBS_SIZE = 31

JOINT_NAMES = [
    "left_hip_lat", "left_hip", "left_knee",
    "right_hip_lat", "right_hip", "right_knee",
]

# Refresh the on-screen text this many times a second.
TEXT_HZ = 10


def gravity_to_quat(gravity):
  """Returns the torso quaternion (w, x, y, z) whose gravity reading is gravity.

  The sim's gravity obs is R.T @ [0, 0, -1]: world down in the IMU frame. So
  R.T takes world z to body up (-gravity). Of the rotations that do, this picks
  the smallest, which keeps yaw near zero.
  """
  up = -np.asarray(gravity, dtype=float)
  up /= np.linalg.norm(up)
  quat = np.zeros(4)
  mujoco.mju_quatZ2Vec(quat, up)  # R.T
  mujoco.mju_negQuat(quat, quat)  # R
  return quat


def obs_gravity(obs):
  """The unit gravity vector, from the observation's accelerometer.

  A still accelerometer reads the reaction to gravity (up), so gravity is its
  negated direction. While the robot moves this also includes its acceleration.
  """
  accel = np.asarray(obs[ACCEL], dtype=float)
  return -accel / np.linalg.norm(accel)


def sim_gravity(data, site_id):
  """The policy's gravity obs, computed from the sim's pose (joystick.py)."""
  return data.site_xmat[site_id].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


def roll_pitch(gravity):
  """Torso roll and pitch in degrees, from the gravity vector."""
  up = -np.asarray(gravity, dtype=float)
  roll = np.arctan2(up[1], up[2])
  pitch = np.arctan2(-up[0], np.hypot(up[1], up[2]))
  return np.degrees(roll), np.degrees(pitch)


def apply_obs(model, data, obs, height):
  """Writes the observation's pose into data and runs mj_forward."""
  data.qpos[0:3] = [0.0, 0.0, height]
  data.qpos[3:7] = gravity_to_quat(obs_gravity(obs))
  data.qpos[7:13] = obs[JOINT_POS]
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def status_text(model, obs, gravity_error, rate):
  """Left and right columns of the on-screen text."""
  roll, pitch = roll_pitch(obs_gravity(obs))
  gravity = np.array2string(obs_gravity(obs), precision=3, suppress_small=True)
  labels = [
      "rate",
      "gravity (from accel)",
      "gravity error",
      "roll (+ left side up)",
      "pitch (+ nose down)",
  ]
  values = [
      f"{rate:.1f} Hz",
      gravity,
      f"{gravity_error:.1e}",
      f"{roll:+.1f} deg",
      f"{pitch:+.1f} deg",
  ]
  # qpos[7 + i] is joint i. Flag any angle outside the sim's joint range.
  for i, name in enumerate(JOINT_NAMES):
    angle = obs[JOINT_POS][i]
    low, high = model.jnt_range[model.joint(name).id]
    flag = "  OUT OF RANGE" if not low <= angle <= high else ""
    labels.append(name)
    values.append(f"{angle:+.3f}  [{low:+.3f}, {high:+.3f}]{flag}")
  return "\n".join(labels), "\n".join(values)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--host", required=True, help="The Pi's address.")
  parser.add_argument("--port", type=int, default=5000)
  parser.add_argument(
      "--height", type=float, default=0.35,
      help="Torso height in metres. Home (feet on the floor) is 0.2723.",
  )
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(str(SCENE))
  data = mujoco.MjData(model)
  site_id = model.site("imu").id

  client = Client(args.host, args.port)
  client.connect()
  print(f"Connected to {args.host}:{args.port}")

  last_text = 0.0
  prev = time.perf_counter()
  rate = 0.0
  try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
      while viewer.is_running():
        # Blocks until the next step's observation, so this never falls behind.
        obs = client.receive()
        if obs is None:
          print("Stream closed")
          break
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (OBS_SIZE,):
          raise ValueError(f"Expected {OBS_SIZE} values, got shape {obs.shape}")

        now = time.perf_counter()
        rate = 0.9 * rate + 0.1 / max(now - prev, 1e-6)
        prev = now

        with viewer.lock():
          apply_obs(model, data, obs, args.height)
          # Should be ~1e-7: the sim shows exactly the tilt the robot reports.
          gravity_error = np.max(np.abs(
              sim_gravity(data, site_id) - obs_gravity(obs)
          ))
        if now - last_text > 1 / TEXT_HZ:
          left, right = status_text(model, obs, gravity_error, rate)
          viewer.set_texts((None, None, left, right))
          last_text = now
        viewer.sync()
  finally:
    client.close()


if __name__ == "__main__":
  main()

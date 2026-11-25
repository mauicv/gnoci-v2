"""Records a policy's stepping-in-place joint targets in the sim, for stepping_test.py.

Runs the policy (models/<model>/policy.onnx) on the free-floating training model
in CPU MuJoCo at zero command, with the gravity observation from the robot's
complementary filter, as on the robot. After --settle seconds it records
--seconds of joint targets, and averages them by gait phase into one gait
cycle. stepping_test.py replays that cycle open loop at other amplitudes and
frequencies, so no policy (and no IMU feedback) is involved in the test.

  uv run python scripts/record_stepping_targets.py --model differ-night-frame

Writes models/<model>/stepping_targets.npz (models/ goes to the robot with the
repo) and a plot next to it:
  targets (N, 6)     the raw recorded joint targets, sim joint order (rad)
  phase   (N,)       the left leg's gait phase at each step, in [-pi, pi)
  cycle   (BINS, 6)  the targets averaged over the cycles, by left-leg phase;
                     bin i covers phase -pi + 2*pi*i/BINS onwards
  mean    (6,)       the cycle's mean pose: amplitudes scale about this
  gait_freq          the frequency it was recorded at (Main's GAIT_FREQ)
plus model and action_scale.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from gnoci_v2.control.filter import ComplementaryFilter
from gnoci_v2.control.main import GAIT_FREQ
from gnoci_v2.control.predict import MODELS_DIR, PolicyRunner
from gnoci_v2.sim_mj.constants import SIM_PATH

SCENE = SIM_PATH / "desc" / "scene.xml"
CTRL_DT, SIM_DT = 0.02, 0.002
JOINTS = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]
# The robot's filter time constant (imu_hwi.FILTER_TAU).
FILTER_TAU = 1.0
BINS = 50


def run_policy(policy, steps):
  """Returns per-step (target, left phase, tilt in degrees) of a zero-command rollout."""
  model = mujoco.MjModel.from_xml_path(str(SCENE))
  model.opt.timestep = SIM_DT
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
  mujoco.mj_forward(model, data)
  qadr = [model.jnt_qposadr[model.joint(j).id] for j in JOINTS]
  dadr = [model.jnt_dofadr[model.joint(j).id] for j in JOINTS]
  act = [model.actuator(j).id for j in JOINTS]
  imu = model.site("imu").id
  gyro_adr = model.sensor_adr[model.sensor("gyro").id]
  acc_adr = model.sensor_adr[model.sensor("accelerometer").id]
  filt = ComplementaryFilter(alpha=FILTER_TAU / (FILTER_TAU + SIM_DT), dt=SIM_DT)

  def sensors():
    return (data.sensordata[acc_adr:acc_adr + 3].copy(),
            data.sensordata[gyro_adr:gyro_adr + 3].copy())

  filt.update(*sensors())
  phase = np.array([0.0, np.pi])
  last_action = np.zeros(6)
  targets, phases, tilts = [], [], []
  for _ in range(steps):
    _, gyro = sensors()
    obs = np.concatenate([gyro, filt.gravity, np.zeros(3), data.qpos[qadr], data.qvel[dadr],
                          last_action, np.cos(phase), np.sin(phase)])
    action = np.asarray(policy.predict(obs), dtype=float)
    target = policy.action_scale * action
    targets.append(target)
    phases.append(phase[0])
    last_action = action
    phase = np.fmod(phase + 2 * np.pi * GAIT_FREQ * CTRL_DT + np.pi, 2 * np.pi) - np.pi
    data.ctrl[act] = target
    for _ in range(int(round(CTRL_DT / SIM_DT))):
      mujoco.mj_step(model, data)
      filt.update(*sensors())
    up = data.site_xmat[imu].reshape(3, 3)[2, 2]
    tilts.append(np.degrees(np.arccos(np.clip(up, -1, 1))))
  return np.array(targets), np.array(phases), np.array(tilts)


def phase_average(targets, phases, bins=BINS):
  """Averages targets into bins by phase; returns (bins, 6)."""
  index = np.floor((phases + np.pi) / (2 * np.pi) * bins).astype(int) % bins
  cycle = np.zeros((bins, targets.shape[1]))
  for b in range(bins):
    if not np.any(index == b):
      raise SystemExit(f"No samples in phase bin {b}: record for longer")
    cycle[b] = targets[index == b].mean(0)
  return cycle


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--model", default="differ-night-frame")
  p.add_argument("--settle", type=float, default=3.0, help="Seconds left out at the start.")
  p.add_argument("--seconds", type=float, default=10.0, help="Seconds recorded after that.")
  args = p.parse_args()

  policy = PolicyRunner(args.model)
  skip = int(round(args.settle / CTRL_DT))
  steps = skip + int(round(args.seconds / CTRL_DT))
  targets, phases, tilts = run_policy(policy, steps)
  if tilts.max() > 30:
    raise SystemExit(f"The policy tilted {tilts.max():.0f} degrees in the sim: not a stable gait")
  targets, phases = targets[skip:], phases[skip:]
  cycle = phase_average(targets, phases)

  # How repeatable the stepping is: the recorded targets against the averaged cycle.
  index = np.floor((phases + np.pi) / (2 * np.pi) * BINS).astype(int) % BINS
  residual = np.sqrt(((targets - cycle[index]) ** 2).mean(0))
  print(f"{args.model}: {args.seconds:g} s of zero-command stepping at {GAIT_FREQ} Hz,"
        f" max tilt {tilts[skip:].max():.1f} deg")
  print(" " * 22 + " ".join(f"{j:>13}" for j in JOINTS))
  print(f"{'cycle mean (rad)':<22}" + " ".join(f"{v:13.3f}" for v in cycle.mean(0)))
  print(f"{'cycle p2p (rad)':<22}" + " ".join(f"{v:13.3f}" for v in np.ptp(cycle, 0)))
  print(f"{'off-cycle rms (rad)':<22}" + " ".join(f"{v:13.3f}" for v in residual))

  out = MODELS_DIR / args.model / "stepping_targets.npz"
  np.savez(out, targets=targets, phase=phases, cycle=cycle, mean=cycle.mean(0),
           gait_freq=GAIT_FREQ, model=args.model, action_scale=policy.action_scale)
  print(f"Saved {out}")

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
  bin_phase = -np.pi + 2 * np.pi * (np.arange(BINS) + 0.5) / BINS
  for i, (ax, joint) in enumerate(zip(axes.T.flat, JOINTS)):
    ax.plot(phases, targets[:, i], ".", ms=2, alpha=0.4, label="recorded")
    ax.plot(bin_phase, cycle[:, i], "k", lw=2, label="averaged cycle")
    ax.set_title(joint)
    ax.grid(alpha=0.3)
  axes[0, 0].legend(fontsize=8)
  for ax in axes[-1]:
    ax.set_xlabel("left-leg gait phase (rad)")
  fig.suptitle(f"{args.model} stepping targets at zero command (rad)")
  fig.tight_layout()
  fig.savefig(out.with_suffix(".png"), dpi=100)
  print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
  main()

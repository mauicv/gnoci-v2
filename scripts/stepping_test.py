"""Plays a recorded stepping gait open loop, on the robot or in the sim, and records it.

For comparing the robot's IMU with the sim's: both get exactly the same joint
targets, with no policy in the loop, so any difference in gyro, accelerometer or
filter gravity comes from the body and the sensor, not from a policy reacting.

The gait is one averaged cycle of a policy stepping in place at zero command,
from scripts/record_stepping_targets.py (models/<MODEL>/stepping_targets.npz).
It's played at every frequency in FREQUENCIES and, for each, every amplitude in
AMPLITUDES (1.0 = the policy's own step size, scaled about the cycle's mean
pose). Each segment ramps up, holds, ramps down, then stands still for a rest.
Edit the lists below to choose what runs.

On the robot (puts it on the floor, no leash; stops if it tilts past FALL_TILT):
  uv run python scripts/stepping_test.py --out logs/stepping/robot-1.npz
In the sim, the same schedule:
  uv run python scripts/stepping_test.py --sim --out logs/stepping/sim.npz
Just print the schedule:
  uv run python scripts/stepping_test.py --dry-run

The .npz has the same keys as `gnoci run --record` (t, obs, action, target,
gravity, accel, voltage, ...), with action = target / action_scale, plus per
step: amplitude, frequency and segment (-1 while resting).
"""

import argparse
import time
from pathlib import Path

import numpy as np

# ----- What to run: edit these -----------------------------------------------
MODEL = "differ-night-frame"       # models/<MODEL>/stepping_targets.npz
AMPLITUDES = [0.3, 0.4, 0.5, 0.6]  # step size, x the policy's own (1.0)
FREQUENCIES = [1.5, 1.75]          # gait cycles per second (the policy uses 1.75)
HOLD_SECONDS = 8.0                 # at full amplitude, per segment
RAMP_SECONDS = 1.5                 # amplitude ramps up from 0, and back down, over this
REST_SECONDS = 2.0                 # standing still between segments
BASELINE_SECONDS = 3.0             # standing still before the first segment
FALL_TILT = 45.0                   # degrees (filter) for 0.1 s: stop, torque off, save
# -----------------------------------------------------------------------------

CONTROL_FREQ = 50
DT = 1 / CONTROL_FREQ
FALL_STEPS = 5
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_cycle(model):
  data = np.load(MODELS_DIR / model / "stepping_targets.npz")
  return data["cycle"], data["mean"], float(data["action_scale"])


def cycle_at(cycle, phase):
  """The cycle's targets at left-leg phase (rad), interpolated between bins."""
  bins = len(cycle)
  x = (np.mod(phase + np.pi, 2 * np.pi) / (2 * np.pi)) * bins - 0.5
  i0 = int(np.floor(x)) % bins
  w = x - np.floor(x)
  return (1 - w) * cycle[i0] + w * cycle[(i0 + 1) % bins]


def build_schedule():
  """Per control step: amplitude, frequency, segment (-1 at rest), and left phase."""
  amp, freq, seg = [], [], []

  def add(seconds, a, f, s):
    n = int(round(seconds / DT))
    for k in range(n):
      amp.append(a(k / n) if callable(a) else a)
      freq.append(f)
      seg.append(s)

  add(BASELINE_SECONDS, 0.0, 0.0, -1)
  s = 0
  for f in FREQUENCIES:
    for a in AMPLITUDES:
      add(RAMP_SECONDS, lambda x, a=a: a * x, f, s)
      add(HOLD_SECONDS, a, f, s)
      add(RAMP_SECONDS, lambda x, a=a: a * (1 - x), f, s)
      add(REST_SECONDS, 0.0, 0.0, -1)
      s += 1
  amp, freq, seg = np.array(amp), np.array(freq), np.array(seg)
  # The phase only advances while stepping, and each segment starts at phase 0.
  phase = np.zeros(len(amp))
  p = 0.0
  for k in range(len(amp)):
    if k and seg[k] != seg[k - 1]:
      p = 0.0
    phase[k] = np.fmod(p + np.pi, 2 * np.pi) - np.pi
    p += 2 * np.pi * freq[k] * DT
  return amp, freq, seg, phase


def schedule_targets(cycle, mean, amp, phase, clamp):
  """Joint targets for every step: mean + amplitude * (cycle - mean), clamped."""
  return np.array([clamp(mean + a * (cycle_at(cycle, ph) - mean)) for a, ph in zip(amp, phase)])


def observation(gyro, gravity, pos, vel, last_action, phase):
  """The 31-value layout of Main.get_observation, with a zero command."""
  phases = np.array([phase, phase + np.pi])
  return np.concatenate([gyro, gravity, np.zeros(3), pos, vel, last_action,
                         np.cos(phases), np.sin(phases)])


def tilt(gravity):
  return float(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))


class Log:
  """Collects the steps and saves them in the `gnoci run --record` format."""

  def __init__(self, path, action_scale, sim):
    from gnoci_v2.control.record import Recorder
    self.recorder = Recorder(path, model=MODEL, action_scale=action_scale,
                             control_freq=CONTROL_FREQ, gait_freq=np.nan, close_loop=True,
                             sim=sim)
    self.action_scale = action_scale
    self.extra = {"amplitude": [], "frequency": [], "segment": []}

  def add(self, t, obs, target, gravity, accel, voltage, amp, freq, seg):
    self.recorder.add(t, obs, target / self.action_scale, target, gravity, accel, voltage)
    for key, value in zip(self.extra, (amp, freq, seg)):
      self.extra[key].append(value)

  def save(self):
    self.recorder.meta.update({k: np.array(v) for k, v in self.extra.items()})
    path = self.recorder.save()
    if path:
      print(f"Saved {len(self.recorder.t)} steps to {path}")


def run_robot(targets, amp, freq, seg, phase, log, start_delay):
  from gnoci_v2.control.imu_hwi import IMUHWI
  from gnoci_v2.control.loop import Loop
  from gnoci_v2.control.servo_hwi import ServoHWI

  servo = ServoHWI()
  imu = IMUHWI(dt=1 / 300)
  latest = {}

  def read_imu():
    gyro, accel, gravity = imu.read_full()
    latest["imu"] = (gyro, accel, gravity, time.perf_counter())

  imu_loop = Loop(func=read_imu, hz=300)
  try:
    imu_loop.start()
    servo.init_servos()  # Moves slowly to home.
    servo.move_to(targets[0], speed=0.3)
    end = time.monotonic() + start_delay
    while (left := end - time.monotonic()) > 0:
      print(f"Starting in {int(np.ceil(left))} s")
      time.sleep(min(1.0, left))
    print(f"Running {len(targets) * DT:.0f} s")

    next_time = time.perf_counter()
    last_action = targets[0] / log.action_scale
    tilted = 0
    for k, target in enumerate(targets):
      start = time.perf_counter()
      if imu_loop.failed or "imu" not in latest:
        raise RuntimeError("The IMU loop isn't running")
      gyro, accel, gravity, stamp = latest["imu"]
      if start - stamp > 0.05:
        raise RuntimeError("The IMU reading is stale")
      pos = servo.get_positions()
      vel = servo.get_velocities()
      obs = observation(gyro, gravity, pos, vel, last_action, phase[k])
      servo.set_positions(target)
      try:
        voltage = servo.get_voltages()
      except RuntimeError:
        voltage = np.full(6, np.nan)
      log.add(start, obs, target, gravity, accel, voltage, amp[k], freq[k], seg[k])
      last_action = target / log.action_scale
      if k % CONTROL_FREQ == 0 and (k == 0 or seg[k] != seg[k - CONTROL_FREQ]):
        print(f"{k * DT:6.1f} s: " + ("resting" if seg[k] < 0 else
                                       f"segment {seg[k]}: {freq[k]} Hz x{max(amp[k:k + 200]):g}"))
      tilted = tilted + 1 if tilt(gravity) > FALL_TILT else 0
      if tilted >= FALL_STEPS:
        print(f"Fell: tilted {tilt(gravity):.0f} degrees at {k * DT:.1f} s")
        break
      next_time += DT
      now = time.perf_counter()
      if next_time <= now:
        next_time = now + DT
      time.sleep(next_time - now)
  finally:
    servo.disable()
    imu_loop.stop()
    log.save()
    if servo.read_retries:
      print(f"Servo reads tried again: {servo.read_retries}")


def run_sim(targets, amp, freq, seg, phase, log):
  import mujoco
  from gnoci_v2.control.filter import ComplementaryFilter
  from gnoci_v2.sim_mj.constants import SIM_PATH

  sim_dt = 0.002
  n_sub = int(round(DT / sim_dt))
  joints = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]
  model = mujoco.MjModel.from_xml_path(str(SIM_PATH / "desc" / "scene.xml"))
  model.opt.timestep = sim_dt
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
  mujoco.mj_forward(model, data)
  qadr = [model.jnt_qposadr[model.joint(j).id] for j in joints]
  dadr = [model.jnt_dofadr[model.joint(j).id] for j in joints]
  act = [model.actuator(j).id for j in joints]
  gyro_adr = model.sensor_adr[model.sensor("gyro").id]
  acc_adr = model.sensor_adr[model.sensor("accelerometer").id]
  # The robot's filter (imu_hwi.FILTER_TAU), run every physics step.
  tau = 1.0
  filt = ComplementaryFilter(alpha=tau / (tau + sim_dt), dt=sim_dt)

  def sensors():
    return (data.sensordata[acc_adr:acc_adr + 3] / 9.81,
            data.sensordata[gyro_adr:gyro_adr + 3].copy())

  def physics(target):
    data.ctrl[act] = target
    for _ in range(n_sub):
      mujoco.mj_step(model, data)
      filt.update(*sensors())

  filt.update(*sensors())
  # As on the robot: stand at home, move to the start pose at 0.3 rad/s, settle.
  for _ in range(CONTROL_FREQ):
    physics(np.zeros(6))
  moves = max(1, int(np.ceil(np.max(np.abs(targets[0])) / (0.3 * DT))))
  for i in range(1, moves + 1):
    physics(targets[0] * i / moves)
  for _ in range(2 * CONTROL_FREQ):
    physics(targets[0])

  last_action = targets[0] / log.action_scale
  tilted = 0
  for k, target in enumerate(targets):
    accel, gyro = sensors()
    gravity = filt.gravity
    obs = observation(gyro, gravity, data.qpos[qadr].copy(), data.qvel[dadr].copy(),
                      last_action, phase[k])
    log.add(k * DT, obs, target, gravity, accel, np.full(6, np.nan), amp[k], freq[k], seg[k])
    last_action = target / log.action_scale
    tilted = tilted + 1 if tilt(gravity) > FALL_TILT else 0
    if tilted >= FALL_STEPS:
      print(f"Fell: tilted {tilt(gravity):.0f} degrees at {k * DT:.1f} s")
      break
    physics(target)
  log.save()


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--sim", action="store_true", help="Run in the sim instead of on the robot.")
  p.add_argument("--out", type=Path, default=None,
                 help="Default: logs/stepping/<robot|sim>-<date-time>.npz")
  p.add_argument("--start-delay", type=float, default=5.0,
                 help="Robot only: seconds standing at the start pose before it begins.")
  p.add_argument("--dry-run", action="store_true", help="Print the schedule and stop.")
  args = p.parse_args()

  from gnoci_v2.control.servo_hwi import ServoHWI

  def clamp(x):
    return np.clip(x, ServoHWI.LOWER_LIMITS + ServoHWI.LIMIT_MARGIN,
                   ServoHWI.UPPER_LIMITS - ServoHWI.LIMIT_MARGIN)

  cycle, mean, action_scale = load_cycle(MODEL)
  amp, freq, seg, phase = build_schedule()
  targets = schedule_targets(cycle, mean, amp, phase, clamp)

  print(f"{MODEL}: {seg.max() + 1} segments, {len(amp) * DT:.0f} s in all")
  for s in range(seg.max() + 1):
    m = seg == s
    p2p = np.ptp(targets[m], 0)
    print(f"  segment {s}: {freq[m][0]} Hz, x{amp[m].max():g}, target p2p (rad)"
          f" {np.round(p2p, 3).tolist()}")
  if args.dry_run:
    return

  kind = "sim" if args.sim else "robot"
  out = args.out or Path("logs/stepping") / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.npz"
  log = Log(out, action_scale, sim=args.sim)
  if args.sim:
    run_sim(targets, amp, freq, seg, phase, log)
  else:
    run_robot(targets, amp, freq, seg, phase, log, args.start_delay)


if __name__ == "__main__":
  main()

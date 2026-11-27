"""Slowly leans the robot forward (or back) over its feet until it starts to tip, and records it.

Both legs' targets move together along a fixed direction, `lean` rad at a time:
  forwards   FORWARD_DIRECTION (knees bend): the torso pitches nose-down and
             its centre of mass moves towards the toes (in the sim, 90% of the
             way from heel to toe at lean 0.4)
  backwards  BACKWARD_DIRECTION (hip back, knee forward): the torso stays level
             and slides back over the heels
Once the centre of mass passes the toes (or heels) the feet start to rotate off
the floor. The lean at which that happens shows where the centre of mass really
is: scripts/lean_compare.py finds the torso centre-of-mass shift that makes the
sim tip at the same lean.

There's no ankle, so the feet's pitch is the torso's pitch (the filter) plus
hip + knee (the measured joints), the same on the robot and in the sim; it
stays put while the feet are flat. The script stops leaning once that has
changed by more than STOP_TILT, or the pitch rate passes STOP_RATE, or the lean
reaches MAX_LEAN, then moves straight back to home. Be ready to catch it: it's
meant to start tipping.

On the robot (on the floor, no leash, flat and level surface):
  uv run python scripts/lean_test.py --forwards --out logs/lean/forwards-1.npz
  uv run python scripts/lean_test.py --backwards --out logs/lean/backwards-1.npz
In the sim, with the torso's centre of mass shifted forward by 20 mm:
  uv run python scripts/lean_test.py --forwards --sim --com-x 20 --out logs/lean/sim-fwd-20.npz

The .npz has the `gnoci run --record` keys (t, obs, action, target, gravity,
accel, voltage), with action = target, plus per step `lean` (rad along the
direction), `foot_pitch` (deg, change from the start) and `phase` (0 settle,
1 leaning, 2 returning).
"""

import argparse
import time
from pathlib import Path

import numpy as np

# ----- Edit these -------------------------------------------------------------
# Joint targets per unit of lean, sim joint order (hip_lat, hip, knee per leg).
FORWARD_DIRECTION = np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0])
BACKWARD_DIRECTION = np.array([0.0, -1.0, 1.0, 0.0, -1.0, 1.0])
LEAN_RATE = 0.01       # rad/s along the direction (slow: the filter lags about tau x rate)
MAX_LEAN = 1.6         # rad: stop here even if it hasn't tipped (backwards, the hip
                       # clamps at about 0.8 but the knee keeps going to about 2.2)
STOP_TILT = 15.0       # degrees of foot pitch change from the start: stop and go back
                       # (the rounded soles and flex make it drift several degrees
                       # before it really tips, so this is a backstop for STOP_RATE)
STOP_RATE = 0.6        # rad/s of pitch rate (gyro y): stop and go back
SETTLE_SECONDS = 3.0   # standing at home before leaning, also the baseline
RETURN_SPEED = 0.5     # rad/s back to home after stopping
# -----------------------------------------------------------------------------

CONTROL_FREQ = 50
DT = 1 / CONTROL_FREQ
ACTION_SCALE = 1.0  # No policy: targets are recorded as actions unscaled.
JOINTS = ["left_hip_lat", "left_hip", "left_knee", "right_hip_lat", "right_hip", "right_knee"]


def pitch_deg(gravity):
  return float(np.degrees(np.arcsin(np.clip(gravity[0], -1.0, 1.0))))


def foot_pitch_deg(gravity, pos):
  """The feet's pitch (deg, + toes down): torso pitch + hip + knee, averaged over the legs."""
  pos = np.asarray(pos)
  return pitch_deg(gravity) + float(np.degrees((pos[1] + pos[2] + pos[4] + pos[5]) / 2))


def observation(gyro, gravity, pos, vel, last_target):
  """Main.get_observation's layout, with zero command and phase."""
  return np.concatenate([gyro, gravity, np.zeros(3), pos, vel, last_target,
                         np.ones(2), np.zeros(2)])


class Lean:
  """The lean schedule as a state machine: call step() once per control step."""

  def __init__(self, direction, clamp):
    self.direction = direction
    self.clamp = clamp
    self.lean = 0.0
    self.phase = 0
    self.k = 0
    self.start = None
    self.foot = 0.0  # Foot pitch change from the start, deg.
    self.reason = None

  def target(self):
    return self.clamp(self.lean * self.direction)

  def step(self, gravity, gyro, pos):
    """Advances one step given the latest IMU and joint angles; returns False when finished."""
    self.k += 1
    foot = foot_pitch_deg(gravity, pos)
    if self.phase == 0:
      if self.k * DT >= SETTLE_SECONDS:
        self.start = foot
        self.phase = 1
    elif self.phase == 1:
      self.foot = foot - self.start
      if abs(self.foot) > STOP_TILT:
        self.reason = f"foot pitch changed {self.foot:+.1f} deg"
      elif abs(gyro[1]) > STOP_RATE:
        self.reason = f"pitch rate {gyro[1]:+.2f} rad/s"
      elif abs(self.lean) >= MAX_LEAN:
        self.reason = "reached MAX_LEAN"
      if self.reason:
        print(f"Stopped at lean {self.lean:+.3f} rad: {self.reason}")
        self.phase = 2
      else:
        self.lean += LEAN_RATE * DT
    else:
      self.lean = max(0.0, self.lean - RETURN_SPEED * DT)
      if self.lean == 0.0:
        return False
    return True


class Log:
  def __init__(self, path, sim, way, direction, com_x):
    from gnoci_v2.control.record import Recorder
    self.recorder = Recorder(path, model="lean_test", action_scale=ACTION_SCALE,
                             control_freq=CONTROL_FREQ, gait_freq=np.nan, close_loop=True,
                             sim=sim, way=way, com_x_mm=com_x,
                             lean_direction=direction, lean_rate=LEAN_RATE)
    self.lean, self.phase, self.foot = [], [], []

  def add(self, t, obs, target, gravity, accel, voltage, schedule):
    self.recorder.add(t, obs, target, target, gravity, accel, voltage)
    self.lean.append(schedule.lean)
    self.phase.append(schedule.phase)
    self.foot.append(schedule.foot)

  def save(self):
    self.recorder.meta.update(lean=np.array(self.lean), phase=np.array(self.phase),
                              foot_pitch=np.array(self.foot))
    path = self.recorder.save()
    if path:
      print(f"Saved {len(self.lean)} steps to {path}")
    return path


def run_robot(schedule, log):
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
    print("Settling, then leaning. Be ready to catch it.")
    next_time = time.perf_counter()
    running = True
    while running:
      start = time.perf_counter()
      if imu_loop.failed or "imu" not in latest or start - latest["imu"][3] > 0.05:
        raise RuntimeError("The IMU loop isn't running")
      gyro, accel, gravity, _ = latest["imu"]
      target = schedule.target()
      pos = servo.get_positions()
      obs = observation(gyro, gravity, pos, servo.get_velocities(), target)
      servo.set_positions(target)
      try:
        voltage = servo.get_voltages()
      except RuntimeError:
        voltage = np.full(6, np.nan)
      log.add(start, obs, target, gravity, accel, voltage, schedule)
      if schedule.phase == 1 and schedule.k % CONTROL_FREQ == 0:
        print(f"lean {schedule.lean:.3f} rad, torso pitch {pitch_deg(gravity):+.1f} deg,"
              f" foot pitch change {schedule.foot:+.1f} deg")
      running = schedule.step(gravity, gyro, pos)
      next_time += DT
      now = time.perf_counter()
      if next_time <= now:
        next_time = now + DT
      time.sleep(next_time - now)
    time.sleep(1.0)  # Hold home a moment before torque goes off.
  finally:
    servo.disable()
    imu_loop.stop()
    log.save()


def sim_model(com_x_mm):
  """The training model, with the torso's centre of mass shifted forward by com_x_mm."""
  import mujoco
  from gnoci_v2.sim_mj.constants import SIM_PATH
  model = mujoco.MjModel.from_xml_path(str(SIM_PATH / "desc" / "scene.xml"))
  model.opt.timestep = 0.002
  model.body_ipos[model.body("body_main").id, 0] += com_x_mm / 1000
  return model


def run_sim(schedule, log, com_x_mm):
  import mujoco
  from gnoci_v2.control.filter import ComplementaryFilter
  from gnoci_v2.control.imu_hwi import FILTER_TAU

  model = sim_model(com_x_mm)
  sim_dt = model.opt.timestep
  n_sub = int(round(DT / sim_dt))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
  mujoco.mj_forward(model, data)
  qadr = [model.jnt_qposadr[model.joint(j).id] for j in JOINTS]
  dadr = [model.jnt_dofadr[model.joint(j).id] for j in JOINTS]
  act = [model.actuator(j).id for j in JOINTS]
  gyro_adr = model.sensor_adr[model.sensor("gyro").id]
  acc_adr = model.sensor_adr[model.sensor("accelerometer").id]
  filt = ComplementaryFilter(alpha=FILTER_TAU / (FILTER_TAU + sim_dt), dt=sim_dt)

  def sensors():
    return (data.sensordata[acc_adr:acc_adr + 3] / 9.81,
            data.sensordata[gyro_adr:gyro_adr + 3].copy())

  filt.update(*sensors())
  for _ in range(CONTROL_FREQ * 2):  # Like init_servos: the filter settles at home.
    data.ctrl[act] = 0.0
    for _ in range(n_sub):
      mujoco.mj_step(model, data)
      filt.update(*sensors())
  running = True
  k = 0
  while running:
    accel, gyro = sensors()
    gravity = filt.gravity
    target = schedule.target()
    pos = data.qpos[qadr].copy()
    obs = observation(gyro, gravity, pos, data.qvel[dadr].copy(), target)
    log.add(k * DT, obs, target, gravity, accel, np.full(6, np.nan), schedule)
    running = schedule.step(gravity, gyro, pos)
    data.ctrl[act] = target
    for _ in range(n_sub):
      mujoco.mj_step(model, data)
      filt.update(*sensors())
    k += 1
  return log.save()


def run(way, sim, com_x_mm=0.0, out=None):
  """way: "forwards" or "backwards". Returns the saved path."""
  from gnoci_v2.control.servo_hwi import ServoHWI

  def clamp(x):
    return np.clip(x, ServoHWI.LOWER_LIMITS + ServoHWI.LIMIT_MARGIN,
                   ServoHWI.UPPER_LIMITS - ServoHWI.LIMIT_MARGIN)

  direction = FORWARD_DIRECTION if way == "forwards" else BACKWARD_DIRECTION
  if out is None:
    kind = f"sim-{way}-com{com_x_mm:+g}" if sim else f"robot-{way}"
    out = Path("logs/lean") / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.npz"
  schedule = Lean(direction, clamp)
  log = Log(out, sim, way, direction, com_x_mm if sim else np.nan)
  if sim:
    return run_sim(schedule, log, com_x_mm)
  return run_robot(schedule, log)


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  way = p.add_mutually_exclusive_group(required=True)
  way.add_argument("--forwards", action="store_true")
  way.add_argument("--backwards", action="store_true")
  p.add_argument("--sim", action="store_true", help="Run in the sim instead of on the robot.")
  p.add_argument("--com-x", type=float, default=0.0,
                 help="Sim only: shift the torso's centre of mass forward by this many mm.")
  p.add_argument("--out", type=Path, default=None,
                 help="Default: logs/lean/<robot|sim>-<direction>-...-<date-time>.npz")
  args = p.parse_args()
  run("forwards" if args.forwards else "backwards", args.sim, args.com_x, args.out)


if __name__ == "__main__":
  main()

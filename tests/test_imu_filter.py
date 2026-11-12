"""The env's IMU filter must match the robot's ComplementaryFilter (control/filter.py)."""

from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np

from gnoci_v2.control.filter import ComplementaryFilter
from gnoci_v2.training import joystick

GRAVITY = slice(3, 6)


def _fake_data(env, accel, gyro):
  """Stands in for mjx.Data: only sensordata is read by the filter."""
  m = env.mj_model
  sensordata = np.zeros(m.nsensordata)
  for name, value in (("accelerometer", accel), ("gyro", gyro)):
    adr = m.sensor_adr[m.sensor(name).id]
    sensordata[adr:adr + 3] = value
  return SimpleNamespace(sensordata=jp.array(sensordata))


def test_filter_update_matches_robot_filter():
  env = joystick.GnociV2Joystick()
  rng = np.random.default_rng(0)
  robot = ComplementaryFilter(alpha=env._filter_alpha, dt=env._config.sim_dt)
  robot.roll, robot.pitch = 0.1, -0.05
  roll_pitch = jp.array([0.1, -0.05])
  for _ in range(200):
    # Accel in m/s^2 like the sim's sensor (the filter only uses its direction).
    accel = 9.81 * (np.array([0, 0, 1.0]) + rng.normal(0, 0.3, 3))
    gyro = rng.normal(0, 1.0, 3)
    robot.update(accel, gyro)
    roll_pitch = env._filter_update(roll_pitch, _fake_data(env, accel, gyro))
  np.testing.assert_allclose(roll_pitch, [robot.roll, robot.pitch], atol=1e-5)
  np.testing.assert_allclose(env._filter_gravity(roll_pitch), robot.gravity, atol=1e-5)


def test_gravity_obs_is_filtered():
  config = joystick.default_config()
  config.noise_config.level = 0.0
  config.latency_config.action_delay_range = [0, 0]
  config.latency_config.imu_delay_range = [0, 0]
  config.push_config.enable = False
  env = joystick.GnociV2Joystick(config=config)
  state = env.reset(jax.random.PRNGKey(0))
  # Starts settled on the true tilt.
  np.testing.assert_allclose(
      state.obs["state"][GRAVITY], env._true_gravity(state.data), atol=1e-5
  )
  step = jax.jit(env.step)
  for _ in range(10):
    state = step(state, jp.zeros(env.action_size))
  # The obs reports the filter, the critic the truth; they differ once moving.
  np.testing.assert_allclose(
      state.obs["state"][GRAVITY], env._filter_gravity(state.info["imu_filter"]),
      atol=1e-5,
  )
  assert not np.allclose(state.obs["state"][GRAVITY], env._true_gravity(state.data))
  # The filter still tracks the true tilt: well within 10 degrees.
  assert np.abs(state.obs["state"][GRAVITY] - env._true_gravity(state.data)).max() < 0.17

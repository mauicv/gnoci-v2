import jax
import jax.numpy as jp


def test_joystick_imports_and_inits():
  from gnoci_v2.training.joystick import GnociV2Joystick

  env = GnociV2Joystick()

  assert env.action_size == 6


def test_joystick_reset():
  from gnoci_v2.training.joystick import GnociV2Joystick

  env = GnociV2Joystick()

  state = env.reset(jax.random.PRNGKey(0))

  assert state.data.qpos.shape == (13,)
  assert state.data.qvel.shape == (12,)
  assert state.data.ctrl.shape == (6,)
  assert state.info["last_contact"].shape == (2,)
  assert state.info["last_contact"].sum() == 0


def test_joystick_step():
  from gnoci_v2.training.joystick import GnociV2Joystick

  env = GnociV2Joystick()

  state = env.reset(jax.random.PRNGKey(0))

  action = jp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  next_state = env.step(state, action)

  assert next_state.data.qpos.shape == (13,)
  assert next_state.data.qvel.shape == (12,)
  assert next_state.data.ctrl.shape == (6,)
  assert next_state.info["last_contact"].shape == (2,)
  assert next_state.info["last_contact"].dtype == jp.bool_

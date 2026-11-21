"""The ONNX export of a policy gives the same actions as the flax policy.

Uses a randomly initialised policy and normalizer built with the same networks as
training, so the tests don't depend on a checkpoint being on disk.
"""

import dataclasses

from brax.training.acme import specs
from brax.training.acme import running_statistics
import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort
import pytest

from gnoci_v2.training.convert import export_onnx, make_deterministic_policy, make_ppo_network

# Policy observation (see joystick.py's _get_obs) and privileged critic state.
OBS_SHAPES = {"state": (31,), "privileged_state": (75,)}


@pytest.fixture(scope="module")
def random_checkpoint():
    """(normalizer, policy) shaped like load_checkpoint's, with random values.

    The normalizer gets a non-trivial mean and std, so normalization is
    exercised too.
    """
    rng = np.random.default_rng(0)
    spec = {k: specs.Array(shape, jp.dtype('float32')) for k, shape in OBS_SHAPES.items()}
    state = running_statistics.init_state(spec)
    state = state.replace(
        mean={k: jp.asarray(rng.normal(size=s), jp.float32) for k, s in OBS_SHAPES.items()},
        std={k: jp.asarray(rng.uniform(0.1, 2.0, size=s), jp.float32) for k, s in OBS_SHAPES.items()},
    )
    normalizer = {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}
    policy = make_ppo_network(OBS_SHAPES).policy_network.init(jax.random.PRNGKey(0))
    return jax.tree.map(np.asarray, normalizer), jax.tree.map(np.asarray, policy)


def test_jax2onnx_converstion(tmpdir, random_checkpoint):
    normalizer, policy = random_checkpoint
    path = export_onnx(make_deterministic_policy(normalizer, policy), tmpdir)
    session = ort.InferenceSession(path)
    (inp,), (out,) = session.get_inputs(), session.get_outputs()
    assert inp.shape[1:] == [OBS_SHAPES["state"][0]]
    assert out.shape[1:] == [6]


@pytest.mark.parametrize("obs_scale", [1.0, 20.0])
def test_onnx_matches_flax_policy(tmpdir, random_checkpoint, obs_scale):
    """The saved ONNX model gives the same actions as the flax policy."""
    normalizer, policy = random_checkpoint
    deterministic_policy = make_deterministic_policy(normalizer, policy)
    session = ort.InferenceSession(export_onnx(deterministic_policy, tmpdir))

    # Observations around the normalizer's distribution, or far outside it.
    mean, std = normalizer["mean"]["state"], normalizer["std"]["state"]
    noise = np.random.default_rng(1).normal(size=(256, OBS_SHAPES["state"][0]))
    obs = (mean + obs_scale * std * noise).astype(np.float32)

    # Compare on CPU: JAX on the GPU differs from an exact float64 result by up
    # to 2.5e-3, while CPU JAX and onnxruntime both stay within about 1e-5.
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(deterministic_policy(obs))
    actual = session.run(None, {session.get_inputs()[0].name: obs})[0]

    # A random policy's actions must vary, or the comparison proves nothing.
    assert expected.std() > 0.05
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

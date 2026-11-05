from gnoci_v2.training.runner import GnociV2Runner
import argparse
from argparse import Namespace
import functools
from pathlib import Path
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
import jax
import numpy as np
import onnxruntime as ort
import pytest
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
from jax2onnx import to_onnx
from gnoci_v2.training.convert import load_checkpoint, make_deterministic_policy, export_onnx

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "2026_11_04_010649_160563200"
)

def test_jax2onnx_converstion(tmpdir):
    normalizer, policy = load_checkpoint(CHECKPOINT_PATH)

    print("Observation normalizer:")
    print(jax.tree.map(lambda x: x.shape, normalizer))
    print("Policy layers:")
    for name, layer in policy["params"].items():
        print(f"  {name}: Dense {layer['kernel'].shape[0]} -> {layer['kernel'].shape[1]}")
    num_params = sum(x.size for x in jax.tree.leaves(policy))
    print(f"Policy parameters: {num_params}")

    export_onnx(make_deterministic_policy(normalizer, policy), tmpdir)


@pytest.mark.parametrize("obs_scale", [1.0, 20.0])
def test_onnx_matches_flax_policy(tmpdir, obs_scale):
    """The saved ONNX model gives the same actions as the flax policy."""
    normalizer, policy = load_checkpoint(CHECKPOINT_PATH)
    deterministic_policy = make_deterministic_policy(normalizer, policy)
    session = ort.InferenceSession(export_onnx(deterministic_policy, tmpdir))

    # Observations around the training distribution, or far outside it.
    mean, std = normalizer["mean"]["state"], normalizer["std"]["state"]
    noise = np.random.default_rng(0).normal(size=(256, 31))
    obs = (mean + obs_scale * std * noise).astype(np.float32)

    # Compare on CPU: JAX on the GPU differs from an exact float64 result by up
    # to 2.5e-3, while CPU JAX and onnxruntime both stay within about 1e-5.
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(deterministic_policy(obs))
    actual = session.run(None, {session.get_inputs()[0].name: obs})[0]

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

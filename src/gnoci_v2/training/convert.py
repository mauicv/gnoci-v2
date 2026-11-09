from gnoci_v2.training.runner import GnociV2Runner
import argparse
from argparse import Namespace
import functools
import json
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

from gnoci_v2.training.constants import ENV_CONFIG_FILE


def load_checkpoint(path):
    """Returns (normalizer, policy) as numpy, so they aren't tied to the GPU."""
    # Saved by BaseRunner.policy_params_fn: [normalizer, policy, value].
    normalizer, policy, _ = ocp.PyTreeCheckpointer().restore(path)
    return jax.tree.map(np.asarray, normalizer), jax.tree.map(np.asarray, policy)


def make_deterministic_policy(normalizer, policy):
    """Rebuilds the networks as training did; the checkpoint only has weights."""
    ppo_config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )  # TODO
    ppo_network = ppo_networks.make_ppo_networks(
        jax.tree.map(lambda x: x.shape, normalizer["mean"]),
        action_size=6,
        preprocess_observations_fn=running_statistics.normalize,
        **ppo_config.network_factory,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_policy(
        (running_statistics.RunningStatisticsState(**normalizer), policy),
        deterministic=True,
    )

    def deterministic_policy(obs):
        """Maps raw observations to actions. The key is unused when deterministic."""
        action, _ = inference_fn({"state": obs}, jax.random.PRNGKey(0))
        return action

    return deterministic_policy


def checkpoint_action_scale(path):
    """The action_scale the checkpoint was trained with, or None if it wasn't recorded.

    Checkpoints saved before the env config was recorded don't have it.
    """
    config_path = Path(path) / ENV_CONFIG_FILE
    if not config_path.is_file():
        return None
    return float(json.loads(config_path.read_text())["action_scale"])


def write_model_config(model_dir, action_scale):
    """Writes models/<name>/config.json: what the robot needs to run the policy."""
    path = Path(model_dir) / "config.json"
    path.write_text(json.dumps({"action_scale": action_scale}, indent=2) + "\n")
    return path


def export_onnx(deterministic_policy, tmpdir):
    path = f"{tmpdir}/policy.onnx"
    to_onnx(
        deterministic_policy,
        [("B", 31)],
        return_mode="file",
        output_path=path,
    )
    return path
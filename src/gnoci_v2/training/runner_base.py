"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter
import imageio_ffmpeg
import mediapy

import os
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

# from playground.common.export_onnx import export_onnx


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        # Control steps per checkpoint video; 0 disables videos.
        self.video_steps = getattr(args, "video_steps", 500)
        self._rollout_fns = None
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        if "eval/episode_reward" in metrics:
            print("-----------")
            print(
                f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
            )
            print("-----------")
        else:
            # Training-episode metrics from log_training_metrics (keys are "episode/...").
            print(
                f'STEP: {num_steps} train reward: {metrics.get("episode/sum_reward", float("nan")):.3f}'
                f' length: {metrics.get("episode/length", float("nan")):.1f}'
                f' sps: {metrics.get("episode/sps", float("nan")):.0f}'
            )

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        if self.video_steps > 0:
            self.render_rollout(make_policy, params, f"{path}.mp4")
        # onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        # export_onnx(
        #     params,
        #     self.action_size,
        #     self.ppo_params,
        #     self.obs_size,  # may not work
        #     output_path=onnx_export_path
        # )

    def render_rollout(self, make_policy, params, path, num_steps=None, seed=0):
        """Roll out the policy deterministically in eval_env and save an mp4."""
        num_steps = num_steps or self.video_steps
        if self._rollout_fns is None:
            self._rollout_fns = (
                jax.jit(self.eval_env.reset),
                jax.jit(self.eval_env.step),
                # Params are an argument so this compiles once, not per checkpoint.
                jax.jit(lambda p, obs, key: make_policy(p, deterministic=True)(obs, key)),
            )
        reset_fn, step_fn, policy_fn = self._rollout_fns

        rng, reset_rng = jax.random.split(jax.random.PRNGKey(seed))
        state = reset_fn(reset_rng)
        trajectory = [state]
        for _ in range(num_steps):
            rng, act_rng = jax.random.split(rng)
            action, _ = policy_fn(params, state.obs, act_rng)
            state = step_fn(state, action)
            trajectory.append(state)
            if state.done:
                break

        frames = self.eval_env.render(trajectory, height=480, width=640, camera="track")
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        mediapy.write_video(path, frames, fps=1.0 / self.eval_env.dt)
        print(
            f"Saved rollout video ({len(trajectory) - 1} steps,"
            f" command {trajectory[0].info['command']}): {path}"
        )

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            log_training_metrics=True,
            training_metrics_steps=2048,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
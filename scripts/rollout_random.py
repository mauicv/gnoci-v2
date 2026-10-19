"""Roll out the joystick env with random actions and save a video.

Usage:
  uv run python scripts/rollout_random.py --steps 250 --out videos/random.mp4
"""

import argparse
import pathlib

import imageio_ffmpeg
import jax
import mediapy

from gnoci_v2.training.joystick import GnociV2Joystick


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=250, help="Control steps to roll out.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("videos/random.mp4"))
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--camera", default="track")
  args = parser.parse_args()

  env = GnociV2Joystick()
  reset = jax.jit(env.reset)
  step = jax.jit(env.step)

  rng = jax.random.PRNGKey(args.seed)
  rng, reset_rng = jax.random.split(rng)
  state = reset(reset_rng)
  trajectory = [state]
  for i in range(args.steps):
    rng, act_rng = jax.random.split(rng)
    action = jax.random.uniform(act_rng, (env.action_size,), minval=-1.0, maxval=1.0)
    state = step(state, action)
    trajectory.append(state)
    if state.done:
      print(f"Episode terminated at step {i + 1}.")
      break

  frames = env.render(trajectory, height=args.height, width=args.width, camera=args.camera)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  mediapy.write_video(args.out, frames, fps=1.0 / env.dt)
  print(f"Wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
  main()

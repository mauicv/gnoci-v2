"""Export a checkpoint to models/<name>/: policy.onnx, and config.json for the robot.

config.json holds the action_scale the policy was trained with, read from the
checkpoint's env_config.json. Checkpoints saved before that file existed need
--action-scale.

  uv run python scripts/convert_policy_jax2onnx.py --name dimension-rain-cape
  uv run python scripts/convert_policy_jax2onnx.py --name old-run --action-scale 0.5
"""

import argparse
from pathlib import Path

from gnoci_v2.training.convert import (
    checkpoint_action_scale,
    export_onnx,
    load_checkpoint,
    make_deterministic_policy,
    write_model_config,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_NAME = "dimension-rain-cape"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default=POLICY_NAME,
                        help="Checkpoint dir in checkpoints/, and the model name in models/.")
    parser.add_argument("--action-scale", type=float, default=None,
                        help="Only for checkpoints without env_config.json.")
    args = parser.parse_args()

    checkpoint_path = ROOT / "checkpoints" / args.name
    recorded = checkpoint_action_scale(checkpoint_path)
    if recorded is not None and args.action_scale is not None and recorded != args.action_scale:
        raise SystemExit(f"--action-scale {args.action_scale} doesn't match the checkpoint's {recorded}")
    action_scale = recorded if recorded is not None else args.action_scale
    if action_scale is None:
        raise SystemExit(f"{checkpoint_path} has no env_config.json: pass the action_scale it was"
                         " trained with as --action-scale")

    model_dir = ROOT / "models" / args.name
    model_dir.mkdir(parents=True, exist_ok=True)
    normalizer, policy = load_checkpoint(checkpoint_path)
    deterministic_policy = make_deterministic_policy(normalizer, policy)
    export_onnx(deterministic_policy, model_dir)
    config_path = write_model_config(model_dir, action_scale)
    print(f"Exported {model_dir / 'policy.onnx'}, action_scale {action_scale} in {config_path}")

import numpy as np
import onnxruntime as ort
import pytest
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
from jax2onnx import to_onnx
from gnoci_v2.training.convert import load_checkpoint, make_deterministic_policy, export_onnx
from pathlib import Path

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "2026_11_04_010649_160563200"
)

if __name__ == "__main__":
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    normalizer, policy = load_checkpoint(CHECKPOINT_PATH)
    deterministic_policy = make_deterministic_policy(normalizer, policy)
    export_onnx(deterministic_policy, f"{models_dir}/2026_11_04_010649_160563200")
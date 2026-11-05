from pathlib import Path

import numpy as np
import onnxruntime as ort

# The repo's models/ folder. Each model is models/<experiment name>/policy.onnx.
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
DEFAULT_MODEL = "serial-pollen-weep"


def available_models(models_dir=MODELS_DIR):
    return sorted(path.parent.name for path in Path(models_dir).glob("*/policy.onnx"))


def model_path(name, models_dir=MODELS_DIR):
    path = Path(models_dir) / name / "policy.onnx"
    if not path.is_file():
        raise FileNotFoundError(
            f"No model named {name!r} ({path} doesn't exist)."
            f" Available: {available_models(models_dir) or 'none'}"
        )
    return path


class PolicyRunner:
    def __init__(self, name=DEFAULT_MODEL, models_dir=MODELS_DIR):
        self.name = name
        self.path = model_path(name, models_dir)
        self.load_model(self.path)
        # Run once, so the first real step isn't slowed by onnxruntime's setup.
        self.predict(np.zeros(self.obs_dim))

    def load_model(self, path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), opts)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        # The model takes a batch, shaped (B, obs_dim).
        self.obs_dim = model_input.shape[1]
        self.action_dim = self.session.get_outputs()[0].shape[1]

    def predict(self, obs):
        """Returns the actions (in [-1, 1]) for a single observation."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected obs of shape ({self.obs_dim},), got {obs.shape}")
        result = self.session.run(None, {self.input_name: obs[None]})
        return result[0][0]

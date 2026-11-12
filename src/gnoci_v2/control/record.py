"""Records each control step of Main, for comparing the robot with the sim.

Saved as an .npz that scripts/hang_rollout_sim.py reads (--real):
  t       (N,)     seconds since the first step, when the observation was read
  obs     (N, 31)  the policy input: gyro, gravity, command, joint_pos,
                   joint_vel, last_action, phase (Main.get_observation)
  action  (N, 6)   the policy output, in [-1, 1]
  target  (N, 6)   the joint targets: action_scale * action, clamped to the
                   servo limits. Recorded even without --close-loop, when
                   nothing is sent.
  gravity (N, 3)   the complementary filter's gravity estimate ([0, 0, -1]
                   upright), the same as obs[:, 3:6]
  accel   (N, 3)   the accelerometer in g, offsets removed, from the same
                   reading as the gyro and gravity (missing in older recordings)
plus model, action_scale, control_freq, gait_freq and close_loop.
"""

from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, path, **meta):
        self.path = Path(path)
        self.meta = meta
        self.t, self.obs, self.action, self.target, self.gravity, self.accel = [], [], [], [], [], []
        self._t0 = None

    def add(self, t, obs, action, target, gravity, accel):
        if self._t0 is None:
            self._t0 = t
        self.t.append(t - self._t0)
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.action.append(np.asarray(action, dtype=np.float32))
        self.target.append(np.asarray(target, dtype=np.float32))
        self.gravity.append(np.asarray(gravity, dtype=np.float32))
        self.accel.append(np.asarray(accel, dtype=np.float32))

    def save(self):
        """Writes the steps so far. Returns the path, or None if there were none."""
        if not self.t:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            t=np.array(self.t),
            obs=np.stack(self.obs),
            action=np.stack(self.action),
            target=np.stack(self.target),
            gravity=np.stack(self.gravity),
            accel=np.stack(self.accel),
            **self.meta,
        )
        return self.path

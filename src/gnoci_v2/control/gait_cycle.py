"""One clean gait cycle of joint targets, taken from a policy recording.

The recorded targets are averaged over the gait phase (a Fourier series in the
left leg's phase), so the cycle keeps the policy's real leg motion but not its
step-to-step corrections or noise. `gnoci sweep` plays it open loop at other
frequencies to measure the robot's rocking response; scripts/gait_sweep.py does
the same in the sim.
"""

import numpy as np

HARMONICS = 3
# Steps left out of the fit: the start (settling) and the end (often a fall).
SKIP_START, SKIP_END = 50, 75


def _basis(phase, harmonics=HARMONICS):
    cols = [np.ones_like(phase)]
    for h in range(1, harmonics + 1):
        cols += [np.cos(h * phase), np.sin(h * phase)]
    return np.stack(cols, axis=-1)


def fit_cycle(paths, harmonics=HARMONICS):
    """Fourier coefficients (2 * harmonics + 1, 6) of the targets over the left leg's phase.

    paths: recordings from `gnoci run --record`. The phase is obs[:, 27:31]
    ([cos_l, cos_r, sin_l, sin_r]).
    """
    phases, targets = [], []
    for path in paths:
        rec = np.load(path)
        n = len(rec["t"])
        sl = slice(SKIP_START, max(SKIP_START + 1, n - SKIP_END))
        obs = rec["obs"][sl]
        phases.append(np.arctan2(obs[:, 29], obs[:, 27]))
        targets.append(rec["target"][sl])
    phase, target = np.concatenate(phases), np.concatenate(targets)
    coef, *_ = np.linalg.lstsq(_basis(phase, harmonics), target, rcond=None)
    return coef


def cycle_targets(coef, phase, amp=1.0):
    """Joint targets at the given left-leg phase(s), scaled about the cycle's mean pose by amp."""
    harmonics = (len(coef) - 1) // 2
    basis = _basis(np.asarray(phase, dtype=float), harmonics)
    mean = coef[0]
    return mean + amp * (basis @ coef - mean)


def schedule(freqs, seconds):
    """[(start time, freq)] for a sweep that holds each frequency for `seconds`."""
    return [(i * seconds, f) for i, f in enumerate(freqs)]

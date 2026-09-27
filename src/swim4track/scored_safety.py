"""Pure safety/status contract used by the scored Stonefish node."""
from __future__ import annotations

import numpy as np

SAFETY_NONFINITE = 1
SAFETY_SURFACE = 2
SAFETY_BOTTOM = 4
SAFETY_HORIZONTAL_BOUND = 8
SAFETY_DIVERGENCE = 16
SAFETY_STALE_ODOMETRY = 32
SAFETY_MULTIPLE_PUBLISHERS = 64
SAFETY_NO_SUBSCRIBER = 128
SAFETY_INVALID_OUTPUT = 256

PHASE_DISABLED = 0
PHASE_ACQUISITION = 1
PHASE_EVALUATION = 2
PHASE_COMPLETED = 3
PHASE_ABORTED = 4


def kinematic_safety_mask(
    state, *, position_error_m, surface_z_m=0.0, bottom_z_m=2.85,
    horizontal_bound_m=3.0, divergence_m=2.0,
):
    """Return a bit mask based only on observable kinematics/workspace bounds."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (12,) or not np.all(np.isfinite(state)):
        return SAFETY_NONFINITE
    mask = 0
    if state[2] < surface_z_m:
        mask |= SAFETY_SURFACE
    if state[2] > bottom_z_m:
        mask |= SAFETY_BOTTOM
    if np.max(np.abs(state[:2])) > horizontal_bound_m:
        mask |= SAFETY_HORIZONTAL_BOUND
    if not np.isfinite(position_error_m) or position_error_m > divergence_m:
        mask |= SAFETY_DIVERGENCE
    return mask

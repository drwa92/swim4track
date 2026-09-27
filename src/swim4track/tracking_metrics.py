"""Task-level metrics for continuous Swim4Track trajectory evaluation."""
from __future__ import annotations

import numpy as np


TRACKING_WINDOW_START_FRACTION = 0.20
POSITION_TOLERANCE_M = 0.30
ATTITUDE_TOLERANCE_RAD = np.deg2rad(10.0)
LINEAR_VELOCITY_TOLERANCE_MPS = 0.20
REQUIRED_TIME_IN_TOLERANCE = 0.90


def continuous_tracking_metrics(position, attitude, linear_velocity):
    """Evaluate sustained tracking after the declared initial transient.

    The legacy terminal-success metric remains available in the evaluator.
    This metric is added because a single endpoint is not representative of a
    continuously moving periodic reference.
    """
    position = np.asarray(position, dtype=np.float64)
    attitude = np.asarray(attitude, dtype=np.float64)
    linear_velocity = np.asarray(linear_velocity, dtype=np.float64)
    if not (position.ndim == attitude.ndim == linear_velocity.ndim == 1):
        raise ValueError("tracking metric inputs must be one-dimensional")
    if not (len(position) == len(attitude) == len(linear_velocity)) or len(position) == 0:
        raise ValueError("tracking metric inputs must be nonempty and equal length")
    if not all(np.all(np.isfinite(v)) for v in (position, attitude, linear_velocity)):
        raise ValueError("tracking metric inputs must be finite")

    start = min(int(np.ceil(TRACKING_WINDOW_START_FRACTION * len(position))), len(position) - 1)
    p = position[start:]
    a = attitude[start:]
    v = linear_velocity[start:]
    within = (
        (p < POSITION_TOLERANCE_M)
        & (a < ATTITUDE_TOLERANCE_RAD)
        & (v < LINEAR_VELOCITY_TOLERANCE_MPS)
    )
    fraction = float(np.mean(within))
    return {
        "tracking_window_start_step": start,
        "tracking_position_p90_m": float(np.percentile(p, 90)),
        "tracking_attitude_p90_deg": float(np.rad2deg(np.percentile(a, 90))),
        "tracking_linear_velocity_p90_mps": float(np.percentile(v, 90)),
        "time_in_tolerance_fraction": fraction,
        "continuous_success": bool(fraction >= REQUIRED_TIME_IN_TOLERANCE),
    }

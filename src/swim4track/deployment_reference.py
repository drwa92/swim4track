"""Smooth, deterministic reference staging for deployment trials."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .trajectory_reference import ReferenceState


def quintic_progress(elapsed_s: float, duration_s: float) -> tuple[float, float]:
    """Return minimum-jerk progress and its time derivative."""
    duration_s = float(duration_s)
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    x = float(np.clip(float(elapsed_s) / duration_s, 0.0, 1.0))
    progress = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    if 0.0 < x < 1.0:
        rate = (30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4) / duration_s
    else:
        rate = 0.0
    return progress, rate


def smooth_acquisition_reference(
    initial_eta,
    target: ReferenceState,
    elapsed_s: float,
    duration_s: float,
) -> ReferenceState:
    """Interpolate from the measured initial pose to a stationary path start."""
    initial_eta = np.asarray(initial_eta, dtype=np.float64)
    if initial_eta.shape != (6,) or not np.all(np.isfinite(initial_eta)):
        raise ValueError("initial_eta must contain six finite values")
    progress, rate = quintic_progress(elapsed_s, duration_s)
    initial_position = initial_eta[:3]
    position_delta = target.position_ned - initial_position
    position = initial_position + progress * position_delta
    linear_velocity = rate * position_delta

    initial_rotation = Rotation.from_euler("xyz", initial_eta[3:])
    target_rotation = Rotation.from_matrix(target.rotation_body_to_ned)
    relative_rotvec = (initial_rotation.inv() * target_rotation).as_rotvec()
    rotation = initial_rotation * Rotation.from_rotvec(progress * relative_rotvec)
    # For R=R0 Exp(s*[delta]x), spatial angular velocity is R0*delta*s_dot.
    angular_velocity = initial_rotation.apply(relative_rotvec) * rate
    return ReferenceState(
        position,
        rotation.as_matrix(),
        linear_velocity,
        angular_velocity,
    )


def ramped_trajectory_clock(
    elapsed_s: float,
    *,
    trajectory_time_scale: float,
    ramp_duration_s: float,
) -> tuple[float, float]:
    """Return internal trajectory time and its rate with a C1 speed ramp.

    The speed factor follows ``3*x**2 - 2*x**3`` during the ramp. Its integral
    defines trajectory time, so both reference position and velocity are
    continuous where acquisition ends and where full path speed begins.
    """
    elapsed_s = max(float(elapsed_s), 0.0)
    scale = float(trajectory_time_scale)
    ramp = float(ramp_duration_s)
    if scale <= 0.0:
        raise ValueError("trajectory_time_scale must be positive")
    if ramp < 0.0:
        raise ValueError("ramp_duration_s must be nonnegative")
    if ramp == 0.0:
        return elapsed_s / scale, 1.0 / scale
    if elapsed_s < ramp:
        x = elapsed_s / ramp
        speed_factor = 3.0 * x**2 - 2.0 * x**3
        trajectory_time = ramp * (x**3 - 0.5 * x**4) / scale
        return trajectory_time, speed_factor / scale
    return (elapsed_s - 0.5 * ramp) / scale, 1.0 / scale


def smooth_command_scale(elapsed_s: float, duration_s: float) -> float:
    """C2 command soft-start factor in [0, 1]."""
    if duration_s < 0.0:
        raise ValueError("duration_s must be nonnegative")
    if duration_s == 0.0:
        return 1.0
    return quintic_progress(elapsed_s, duration_s)[0]

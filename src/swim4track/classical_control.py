"""Frozen PID baseline and its damped thrust allocator.

Numerical PID and allocation implementation retained from the experiment.
This release does not expose NMPC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from .plant import T200ActuatorBank, ValidatedBlueROV2Heavy, t200_force
from .tracking import tracking_errors


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class AllocationResult:
    action: np.ndarray
    force_n: np.ndarray
    achieved_wrench: np.ndarray
    requested_wrench: np.ndarray
    feasibility_scale: float


class DampedThrusterAllocator:
    """Minimum-norm BlueROV2 allocator with common feasibility scaling."""

    def __init__(self, allocation_matrix=None, damping=1e-6, force_limit_n=None):
        matrix = (
            ValidatedBlueROV2Heavy().p.allocation_matrix
            if allocation_matrix is None else np.asarray(allocation_matrix, dtype=np.float64)
        )
        if matrix.shape != (6, 8) or np.linalg.matrix_rank(matrix) != 6:
            raise ValueError("allocation_matrix must be a full-row-rank 6x8 matrix")
        if damping < 0.0:
            raise ValueError("damping must be nonnegative")
        self.matrix = matrix.copy()
        self.damping = float(damping)
        self.force_limit_n = float(
            T200ActuatorBank.applied_force_limit if force_limit_n is None else force_limit_n
        )
        if self.force_limit_n <= 0.0:
            raise ValueError("force_limit_n must be positive")
        gram = self.matrix @ self.matrix.T + self.damping * np.eye(6)
        self.inverse = self.matrix.T @ np.linalg.inv(gram)
        self.wrench_scale = np.sum(np.abs(self.matrix), axis=1) * self.force_limit_n
        command_grid = np.linspace(-T200ActuatorBank.command_limit,
                                   T200ActuatorBank.command_limit, 20001)
        force_grid = t200_force(command_grid)
        if not np.all(np.diff(force_grid) > 0.0):
            raise RuntimeError("T200 command-force map is not monotone on the valid interval")
        self._command_grid = command_grid
        self._force_grid = force_grid

    def allocate(self, wrench):
        wrench = _vector(wrench, 6, "wrench")
        desired_force = self.inverse @ wrench
        peak = float(np.max(np.abs(desired_force)))
        scale = 1.0 if peak <= self.force_limit_n else self.force_limit_n / peak
        force = scale * desired_force
        command = np.interp(force, self._force_grid, self._command_grid)
        action = command / T200ActuatorBank.command_limit
        action = np.clip(action, -1.0, 1.0)
        realized_force = t200_force(T200ActuatorBank.command_limit * action)
        realized_force = np.clip(realized_force, -self.force_limit_n, self.force_limit_n)
        return AllocationResult(
            action=action,
            force_n=realized_force,
            achieved_wrench=self.matrix @ realized_force,
            requested_wrench=wrench.copy(),
            feasibility_scale=float(scale),
        )


@dataclass(frozen=True)
class PIDGains:
    kp_pose: np.ndarray = field(default_factory=lambda: np.array([38., 38., 48., 8., 8., 12.]))
    kd_twist: np.ndarray = field(default_factory=lambda: np.array([28., 28., 35., 5., 5., 7.]))
    ki_pose: np.ndarray = field(default_factory=lambda: np.array([1.8, 1.8, 2.5, .25, .25, .35]))
    integral_limit: np.ndarray = field(default_factory=lambda: np.array([.8, .8, .6, .35, .35, .45]))
    antiwindup_gain: float = 0.25
    feedforward: bool = True

    def __post_init__(self):
        for name in ("kp_pose", "kd_twist", "ki_pose", "integral_limit"):
            value = _vector(getattr(self, name), 6, name)
            if np.any(value < 0.0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.antiwindup_gain < 0.0:
            raise ValueError("antiwindup_gain must be nonnegative")


class PIDWrenchController:
    """Six-axis body-wrench PID with model feedforward and anti-windup."""

    method = "pid"

    def __init__(self, gains=None, allocator=None, model=None, control_dt=.1):
        self.gains = gains or PIDGains()
        self.allocator = allocator or DampedThrusterAllocator()
        self.model = model or ValidatedBlueROV2Heavy()
        self.control_dt = float(control_dt)
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self.reset()

    def reset(self):
        self.integral = np.zeros(6)
        self.previous_reference_twist = None
        self.filtered_reference_acceleration = np.zeros(6)
        self.last_diagnostics = {}

    def _errors(self, state, reference):
        state = _vector(state, 12, "state")
        errors = tracking_errors(state[:6], state[6:], reference)
        rotation = errors["current_rotation"]
        pose_error = np.r_[
            rotation.T @ (-errors["position_ned"]),
            errors["attitude_body"],
        ]
        twist_error = np.r_[
            rotation.T @ (-errors["linear_velocity_ned"]),
            rotation.T @ (-errors["angular_velocity_ned"]),
        ]
        reference_twist = np.r_[
            rotation.T @ reference.linear_velocity_ned,
            rotation.T @ reference.angular_velocity_ned,
        ]
        return pose_error, twist_error, reference_twist

    def _feedforward(self, state, reference_twist):
        if not self.gains.feedforward:
            return np.zeros(6)
        if self.previous_reference_twist is None:
            acceleration = np.zeros(6)
        else:
            raw_acceleration = (
                reference_twist - self.previous_reference_twist
            ) / self.control_dt
            self.filtered_reference_acceleration = (
                0.8 * self.filtered_reference_acceleration + 0.2 * raw_acceleration
            )
            acceleration = self.filtered_reference_acceleration
        acceleration = np.clip(acceleration, -2.0, 2.0)
        return (
            self.model.mass @ acceleration
            + self.model.coriolis(reference_twist) @ reference_twist
            + self.model.damping(reference_twist) @ reference_twist
            + self.model.restoring(state[:6])
        )

    def act(self, state, reference, trajectory=None, time_s=None):
        started = perf_counter()
        pose_error, twist_error, reference_twist = self._errors(state, reference)
        self.integral = np.clip(
            self.integral + self.control_dt * pose_error,
            -self.gains.integral_limit,
            self.gains.integral_limit,
        )
        feedback = (
            self.gains.kp_pose * pose_error
            + self.gains.kd_twist * twist_error
            + self.gains.ki_pose * self.integral
        )
        requested = feedback + self._feedforward(state, reference_twist)
        allocation = self.allocator.allocate(requested)
        # Back-calculation is expressed in the same body-wrench coordinates as
        # the integrator; axes without integral action are left unchanged.
        enabled = self.gains.ki_pose > 1e-12
        correction = np.zeros(6)
        correction[enabled] = (
            allocation.achieved_wrench[enabled] - requested[enabled]
        ) / self.gains.ki_pose[enabled]
        self.integral = np.clip(
            self.integral + self.control_dt * self.gains.antiwindup_gain * correction,
            -self.gains.integral_limit,
            self.gains.integral_limit,
        )
        self.previous_reference_twist = reference_twist.copy()
        self.last_diagnostics = {
            "solve_time_ms": 1000.0 * (perf_counter() - started),
            "requested_wrench": requested.copy(),
            "achieved_wrench": allocation.achieved_wrench.copy(),
            "feasibility_scale": allocation.feasibility_scale,
            "optimizer_success": True,
        }
        return allocation.action.copy()

"""ROS-independent, guarded runtime for frozen Stonefish demonstrations.

Reference time follows the caller's monotonic clock. The controller and
actuator update at 10 Hz. This refactored demo runtime is not the historical
paper campaign scheduler, which advanced its reference by callback count.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np

from .actuator_dynamics import FirstOrderRateLimitedActuator
from .deployment import map_policy_action_to_stonefish
from .deployment_reference import (
    ramped_trajectory_clock, smooth_acquisition_reference, smooth_command_scale,
)
from .scored_safety import kinematic_safety_mask
from .tracking import tracking_errors, trajectory_observation
from .trajectory_reference import ReferenceState, evaluation_trajectories


@dataclass(frozen=True)
class RuntimeConfig:
    controller_kind: str = "learned"
    trajectory_name: str = "hold_level"
    control_dt_s: float = 0.1
    acquisition_s: float = 30.0
    evaluation_s: float = 60.0
    path_speed_ramp_s: float = 5.0
    command_soft_start_s: float = 2.0
    reference_z_offset_m: float = -2.5
    trajectory_time_scale: float = 2.0
    odometry_timeout_s: float = 0.5
    max_control_gap_s: float = 0.5
    surface_z_m: float = 0.0
    bottom_z_m: float = 2.85
    horizontal_bound_m: float = 3.0
    divergence_m: float = 2.0

    def __post_init__(self):
        if self.controller_kind not in {"learned", "pid"}:
            raise ValueError("controller_kind must be learned or pid")
        values = [v for v in self.__dict__.values() if isinstance(v, (float, int))]
        if not np.all(np.isfinite(values)):
            raise ValueError("runtime parameters must be finite")
        # The frozen PID gains and interface filter use this sample interval.
        if not np.isclose(self.control_dt_s, 0.1):
            raise ValueError("the frozen deployment contract requires control_dt_s=0.1")
        for key in ("evaluation_s", "trajectory_time_scale", "odometry_timeout_s",
                    "max_control_gap_s", "horizontal_bound_m", "divergence_m"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("acquisition_s", "path_speed_ramp_s", "command_soft_start_s"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.surface_z_m >= self.bottom_z_m:
            raise ValueError("surface_z_m must be less than bottom_z_m (NED depth)")


@dataclass
class StepResult:
    command: np.ndarray | None
    status: dict[str, Any]
    action: np.ndarray | None = None
    desired_command: np.ndarray | None = None
    reference: ReferenceState | None = None
    compute_ms: float | None = None


class TrackingRuntime:
    """Inject an actor with ``predict(obs)`` or a PID with ``act(state, ref)``.

    Disabled runtimes produce no command. A stop/abort after active operation
    returns exactly one immediate zero command; stopping does not slew through
    a nonzero command. Re-enabling always requires an explicit start request.
    """

    def __init__(self, controller, config: RuntimeConfig | None = None):
        self.controller = controller
        self.config = config or RuntimeConfig()
        available = {t.name: t for t in evaluation_trajectories()}
        if self.config.trajectory_name not in available:
            raise ValueError(f"unknown trajectory: {self.config.trajectory_name}")
        self.trajectory = available[self.config.trajectory_name]
        self.actuator = FirstOrderRateLimitedActuator(
            size=8, control_dt_s=0.1, time_constant_s=0.25,
            rate_limit_per_s=2.0, command_limit=1.0,
        )
        self.previous_action = np.zeros(8)
        self.state: np.ndarray | None = None
        self.received_at: float | None = None
        self.started_at: float | None = None
        self.last_step_at: float | None = None
        self.initial_eta: np.ndarray | None = None
        self.active = False
        self.phase = "disabled"
        self.reason = "awaiting explicit start"
        self.elapsed_s = 0.0
        self.run_id: str | None = None

    def update_odometry(self, state, received_at: float) -> bool:
        value = np.asarray(state, dtype=float)
        if value.shape != (12,) or not np.all(np.isfinite(value)) or not np.isfinite(received_at):
            return False
        self.state = value.copy()
        self.received_at = float(received_at)
        return True

    def odometry_fresh(self, now: float) -> bool:
        return bool(self.state is not None and self.received_at is not None
                    and np.isfinite(now)
                    and 0.0 <= now - self.received_at <= self.config.odometry_timeout_s)

    def start(self, now: float) -> tuple[bool, str]:
        if self.active:
            return False, "already active; disable before starting a new run"
        if not self.odometry_fresh(now):
            return False, "start refused: missing or stale valid odometry"
        mask = kinematic_safety_mask(
            self.state, position_error_m=0.0,
            surface_z_m=self.config.surface_z_m, bottom_z_m=self.config.bottom_z_m,
            horizontal_bound_m=self.config.horizontal_bound_m,
            divergence_m=self.config.divergence_m,
        )
        if mask:
            return False, f"start refused: workspace safety mask {mask}"
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        self.actuator.reset()
        self.previous_action.fill(0.0)
        self.initial_eta = self.state[:6].copy()
        self.started_at = float(now)
        self.last_step_at = float(now)
        self.elapsed_s = 0.0
        self.run_id = uuid4().hex
        self.phase = "acquisition" if self.config.acquisition_s > 0 else "evaluation"
        self.reason = ""
        self.active = True
        return True, "started from the current measured pose"

    def status(self, **extra) -> dict[str, Any]:
        return dict(schema="swim4track_demo_status_v1", phase=self.phase,
                    run_id=self.run_id, active=self.active, reason=self.reason, elapsed_s=self.elapsed_s,
                    evaluation_elapsed_s=max(0.0, self.elapsed_s-self.config.acquisition_s),
                    trajectory=self.config.trajectory_name,
                    controller=self.config.controller_kind, **extra)

    def _finish(self, phase: str, reason: str) -> StepResult:
        command = np.zeros(8) if self.active else None
        self.active = False
        self.phase, self.reason = phase, reason
        self.actuator.reset()
        self.previous_action.fill(0.0)
        return StepResult(command, self.status())

    def stop(self, reason: str = "disabled by request") -> StepResult:
        return self._finish("disabled", reason)

    def abort(self, reason: str) -> StepResult:
        return self._finish("aborted", reason)

    def scheduled_reference(self, elapsed_s: float) -> ReferenceState:
        cfg = self.config
        offset = np.array([0., 0., cfg.reference_z_offset_m])
        first = self.trajectory.sample(0.0)
        target = ReferenceState(first.position_ned+offset, first.rotation_body_to_ned,
                                first.linear_velocity_ned, first.angular_velocity_ned)
        if elapsed_s < cfg.acquisition_s:
            return smooth_acquisition_reference(self.initial_eta, target, elapsed_s, cfg.acquisition_s)
        trajectory_time, rate = ramped_trajectory_clock(
            elapsed_s-cfg.acquisition_s, trajectory_time_scale=cfg.trajectory_time_scale,
            ramp_duration_s=cfg.path_speed_ramp_s,
        )
        nominal = self.trajectory.sample(trajectory_time)
        return ReferenceState(nominal.position_ned+offset, nominal.rotation_body_to_ned,
                              nominal.linear_velocity_ned*rate, nominal.angular_velocity_ned*rate)

    def step(self, now: float) -> StepResult:
        if not self.active:
            return StepResult(None, self.status())
        if not np.isfinite(now) or now < self.last_step_at:
            return self.abort("invalid or backward monotonic clock")
        self.elapsed_s = float(now-self.started_at)
        if now-self.last_step_at > self.config.max_control_gap_s:
            return self.abort("control callback gap exceeded limit")
        if not self.odometry_fresh(now):
            return self.abort("odometry missing or stale")
        self.last_step_at = float(now)
        if self.elapsed_s >= self.config.acquisition_s+self.config.evaluation_s:
            return self._finish("completed", "scheduled tracking interval completed")
        self.phase = "acquisition" if self.elapsed_s < self.config.acquisition_s else "evaluation"
        reference = self.scheduled_reference(self.elapsed_s)
        errors = tracking_errors(self.state[:6], self.state[6:], reference)
        position_error = float(np.linalg.norm(errors["position_ned"]))
        mask = kinematic_safety_mask(
            self.state, position_error_m=position_error,
            surface_z_m=self.config.surface_z_m, bottom_z_m=self.config.bottom_z_m,
            horizontal_bound_m=self.config.horizontal_bound_m, divergence_m=self.config.divergence_m,
        )
        if mask:
            return self.abort(f"kinematic safety mask {mask}")
        began = perf_counter()
        try:
            if self.config.controller_kind == "learned":
                observation = trajectory_observation(
                    self.state[:6], self.state[6:], reference, self.previous_action,
                    representation="swim4track",
                )
                action = self.controller.predict(observation)
            else:
                action = self.controller.act(self.state, reference)
            action = np.asarray(action, dtype=float)
            if action.shape != (8,) or not np.all(np.isfinite(action)):
                raise ValueError("controller must produce eight finite values")
            action = np.clip(action, -1.0, 1.0)
            desired = map_policy_action_to_stonefish(
                action, command_signs=-np.ones(8), training_command_limit=0.60,
                stonefish_command_limit=1.0, actuator_mapping="force_equivalent",
            )
            desired *= smooth_command_scale(self.elapsed_s, self.config.command_soft_start_s)
            command = self.actuator.step(desired)
        except Exception as exc:
            return self.abort(f"controller/interface error: {type(exc).__name__}: {exc}")
        compute_ms = (perf_counter()-began)*1000.0
        # Memory is the clipped, raw actor action, before calibration/filtering.
        self.previous_action = action.copy()
        return StepResult(
            command=command, action=action, desired_command=desired, reference=reference,
            compute_ms=compute_ms,
            status=self.status(position_error_m=position_error,
                               attitude_error_rad=float(np.linalg.norm(errors["attitude_body"])),
                               odometry_age_s=float(now-self.received_at),
                               deadline_miss=bool(compute_ms >= self.config.control_dt_s*1000.0)),
        )

"""Deterministic actuator-bandwidth interface for deployment."""
from __future__ import annotations

import math

import numpy as np


class FirstOrderRateLimitedActuator:
    """Apply first-order lag and a hard componentwise command-rate bound."""

    def __init__(
        self,
        *,
        size: int,
        control_dt_s: float,
        time_constant_s: float,
        rate_limit_per_s: float,
        command_limit: float,
    ):
        self.size = int(size)
        self.control_dt_s = float(control_dt_s)
        self.time_constant_s = float(time_constant_s)
        self.rate_limit_per_s = float(rate_limit_per_s)
        self.command_limit = float(command_limit)
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive")
        if self.rate_limit_per_s <= 0.0:
            raise ValueError("rate_limit_per_s must be positive")
        if self.command_limit <= 0.0:
            raise ValueError("command_limit must be positive")
        self.alpha = 1.0 - math.exp(
            -self.control_dt_s / self.time_constant_s
        )
        self.maximum_step = self.rate_limit_per_s * self.control_dt_s
        self.state = np.zeros(self.size, dtype=np.float64)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, desired_command) -> np.ndarray:
        desired = np.asarray(desired_command, dtype=np.float64)
        if desired.shape != (self.size,) or not np.all(np.isfinite(desired)):
            raise ValueError(
                f"desired_command must contain {self.size} finite values"
            )
        desired = np.clip(
            desired, -self.command_limit, self.command_limit
        )
        unconstrained_delta = self.alpha * (desired - self.state)
        delta = np.clip(
            unconstrained_delta, -self.maximum_step, self.maximum_step
        )
        self.state = np.clip(
            self.state + delta, -self.command_limit, self.command_limit
        )
        return self.state.copy()

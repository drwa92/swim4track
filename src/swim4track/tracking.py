"""Trajectory-conditioned state and nominal tracking task for Swim4Track v0.6."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .core import POSITION_SCALE, TWIST_SCALE, ReefCore
from .plant import rotation_body_to_world
from .trajectory_reference import (
    ReferenceState,
    ReferenceTrajectory,
    training_trajectory,
)

TRACKING_REWARD_WEIGHTS = np.array(
    [-4.0, -3.0, -1.8, -0.8, -0.8, -0.4, -0.4, -0.3],
    dtype=np.float64,
)
REPRESENTATIONS = ("conventional", "swim4track")


def tracking_errors(eta, nu, reference: ReferenceState):
    """Return physical tracking errors without normalization or clipping."""
    eta = np.asarray(eta, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    if eta.shape != (6,) or nu.shape != (6,):
        raise ValueError("eta and nu must each contain six values")
    current_rotation = rotation_body_to_world(eta[3:])
    desired_rotation = reference.rotation_body_to_ned
    position_ned = eta[:3] - reference.position_ned
    attitude_body = Rotation.from_matrix(
        current_rotation.T @ desired_rotation
    ).as_rotvec()
    linear_velocity_ned = current_rotation @ nu[:3] - reference.linear_velocity_ned
    angular_velocity_ned = current_rotation @ nu[3:] - reference.angular_velocity_ned
    relative_rpy = Rotation.from_matrix(
        current_rotation.T @ desired_rotation
    ).as_euler("xyz")
    return {
        "position_ned": position_ned,
        "attitude_body": attitude_body,
        "relative_rpy": relative_rpy,
        "linear_velocity_ned": linear_velocity_ned,
        "angular_velocity_ned": angular_velocity_ned,
        "current_rotation": current_rotation,
    }


def trajectory_observation(
    eta,
    nu,
    reference: ReferenceState,
    previous_action,
    representation="swim4track",
):
    """Build a matched 23-value policy state for either comparison arm.

    ``conventional`` retains inertial-coordinate errors and absolute RPY.
    ``swim4track`` expresses translation and twist mismatch in the current
    vehicle axes and replaces absolute RPY with orientation relative to the
    inertial vertical. Both arms receive the same tracking variables and have
    the same dimensionality; their three-value orientation context is the
    deliberately controlled representation difference.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    previous_action = np.asarray(previous_action, dtype=np.float64)
    if previous_action.shape != (8,):
        raise ValueError("previous_action must contain eight values")
    errors = tracking_errors(eta, nu, reference)
    rotation = errors["current_rotation"]
    if representation == "conventional":
        position = errors["position_ned"] / POSITION_SCALE
        attitude_context = np.asarray(eta[3:], dtype=np.float64) / np.pi
        linear = errors["linear_velocity_ned"] / TWIST_SCALE[:3]
        angular = errors["angular_velocity_ned"] / TWIST_SCALE[3:]
    else:
        position = (rotation.T @ errors["position_ned"]) / POSITION_SCALE
        attitude_context = rotation.T @ np.array([0.0, 0.0, 1.0])
        linear = (rotation.T @ errors["linear_velocity_ned"]) / TWIST_SCALE[:3]
        angular = (rotation.T @ errors["angular_velocity_ned"]) / TWIST_SCALE[3:]
    state = np.r_[
        position,
        errors["attitude_body"] / np.pi,
        attitude_context,
        linear,
        angular,
        previous_action,
    ]
    return np.clip(state, -1.0, 1.0).astype(np.float32)


def tracking_reward(eta, nu, reference, action, previous_action):
    """Matched physical reward used by both v0.6 comparison policies."""
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    errors = tracking_errors(eta, nu, reference)
    # Horizontal norms make the task reward independent of arbitrary global
    # heading while retaining a separate depth weight, as required in NED.
    terms = np.array([
        np.linalg.norm(errors["position_ned"][:2]),
        np.abs(errors["position_ned"][2]),
        np.linalg.norm(errors["attitude_body"]),
        np.linalg.norm(errors["linear_velocity_ned"][:2]),
        np.abs(errors["linear_velocity_ned"][2]),
        np.linalg.norm(errors["angular_velocity_ned"]),
        np.abs(action - previous_action).sum(),
        np.abs(action).sum(),
    ])
    return float(TRACKING_REWARD_WEIGHTS @ terms), terms, errors


class Swim4TrackCore(ReefCore):
    """Nominal continuous position-attitude tracking task.

    This class intentionally reuses the frozen plant, actuator, time-step and
    safety contracts.  Domain randomization and exogenous disturbance are off
    in v0.6 so that the representation experiment remains identifiable.
    """

    def __init__(
        self,
        representation="swim4track",
        control_dt=.1,
        physics_dt=.02,
        max_steps=400,
        command_limit=.60,
    ):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        super().__init__(
            control_dt=control_dt,
            physics_dt=physics_dt,
            max_steps=max_steps,
            command_limit=command_limit,
            domain_randomization=False,
            task="transfer36",
        )
        self.representation = representation
        self.trajectory = None
        self.reference = None
        self.time = 0.0

    def reset(self, seed=None, state=None, trajectory: ReferenceTrajectory | None = None):
        if seed is not None:
            self.seed(seed)
        self.actuators.reset()
        self.steps = 0
        self.time = 0.0
        self.previous_action.fill(0.0)
        self.trajectory = trajectory if trajectory is not None else training_trajectory(self.rng)
        if not isinstance(self.trajectory, ReferenceTrajectory):
            raise TypeError("trajectory must be a ReferenceTrajectory")
        self.reference = self.trajectory.sample(0.0)
        if state is None:
            rotation_error = Rotation.from_euler("xyz", [
                self.rng.uniform(-np.deg2rad(3.0), np.deg2rad(3.0)),
                self.rng.uniform(-np.deg2rad(3.0), np.deg2rad(3.0)),
                self.rng.uniform(-np.deg2rad(30.0), np.deg2rad(30.0)),
            ])
            current_rotation = self.reference.rotation_body_to_ned * 1.0
            current_rotation = current_rotation @ rotation_error.as_matrix()
            position = self.reference.position_ned + self.rng.uniform(
                [-.35, -.35, -.2], [.35, .35, .2]
            )
            body_linear = current_rotation.T @ self.reference.linear_velocity_ned
            body_angular = current_rotation.T @ self.reference.angular_velocity_ned
            nu = np.r_[body_linear, body_angular] + self.rng.uniform(-.03, .03, 6)
            state = np.r_[
                position,
                Rotation.from_matrix(current_rotation).as_euler("xyz"),
                nu,
            ]
        self.state = np.asarray(state, dtype=np.float64).copy()
        if self.state.shape != (12,) or not np.all(np.isfinite(self.state)):
            raise ValueError("state must contain 12 finite values")
        self.goal = np.r_[self.reference.position_ned, self.reference.rpy]
        return trajectory_observation(
            self.state[:6], self.state[6:], self.reference,
            self.previous_action, self.representation,
        )

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain eight finite values")
        command = self.command_limit * action
        for _ in range(self.substeps):
            force = self.actuators.update(command, self.physics_dt, self.health)
            wrench = self.vehicle.p.allocation_matrix @ force
            self.state = self.vehicle.step(self.state, wrench, self.physics_dt)
        self.steps += 1
        self.time = self.steps * self.control_dt
        self.reference = self.trajectory.sample(self.time)
        self.goal = np.r_[self.reference.position_ned, self.reference.rpy]
        finite = bool(np.all(np.isfinite(self.state)))
        position_distance = (
            np.linalg.norm(self.state[:3] - self.reference.position_ned)
            if finite else np.inf
        )
        # Gymnasium requires the built-in bool type, not numpy.bool_.
        unsafe = bool((not finite) or position_distance > 25.0)
        if finite:
            reward, terms, errors = tracking_reward(
                self.state[:6], self.state[6:], self.reference,
                action, self.previous_action,
            )
            next_observation = trajectory_observation(
                self.state[:6], self.state[6:], self.reference,
                action, self.representation,
            )
        else:
            reward, terms, errors = -1000.0, np.full(8, np.inf), None
            next_observation = np.zeros(23, dtype=np.float32)
        if unsafe and finite:
            reward -= 1000.0
        self.previous_action = action.copy()
        truncated = self.steps >= self.max_steps
        info = {
            "reward_terms": terms,
            "command": command.copy(),
            "unsafe": unsafe,
            "time": self.time,
            "trajectory": self.trajectory.name,
            "reference_position": self.reference.position_ned.copy(),
            "reference_rpy": self.reference.rpy.copy(),
            "errors": errors,
        }
        return next_observation, reward, unsafe, truncated, info

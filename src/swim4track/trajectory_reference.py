"""Continuous, differentiable reference trajectories for Swim4Track.

All positions and linear velocities are expressed in the NED world frame.
Rotations map FRD body coordinates to NED, and angular velocity is expressed
in NED.  Keeping this contract explicit prevents silent frame mixing between
the Python plant and Stonefish deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


def _array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"expected finite array with shape {shape}, got {result.shape}")
    return result


@dataclass(frozen=True)
class ReferenceState:
    """One desired position-attitude state and its first derivatives."""

    position_ned: np.ndarray
    rotation_body_to_ned: np.ndarray
    linear_velocity_ned: np.ndarray
    angular_velocity_ned: np.ndarray

    def __post_init__(self):
        position = _array(self.position_ned, (3,))
        rotation = _array(self.rotation_body_to_ned, (3, 3))
        linear = _array(self.linear_velocity_ned, (3,))
        angular = _array(self.angular_velocity_ned, (3,))
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("rotation_body_to_ned must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation_body_to_ned must have determinant +1")
        object.__setattr__(self, "position_ned", position)
        object.__setattr__(self, "rotation_body_to_ned", rotation)
        object.__setattr__(self, "linear_velocity_ned", linear)
        object.__setattr__(self, "angular_velocity_ned", angular)

    @property
    def rpy(self):
        return Rotation.from_matrix(self.rotation_body_to_ned).as_euler("xyz")


def _world_angular_velocity(rpy, rpy_rate):
    """Convert xyz Euler rates into an angular-velocity vector in NED."""
    roll, pitch, _ = rpy
    roll_rate, pitch_rate, yaw_rate = rpy_rate
    body = np.array([
        roll_rate - yaw_rate * np.sin(pitch),
        pitch_rate * np.cos(roll) + yaw_rate * np.sin(roll) * np.cos(pitch),
        -pitch_rate * np.sin(roll) + yaw_rate * np.cos(roll) * np.cos(pitch),
    ])
    rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    return rotation @ body


def _quintic_progress(t, duration):
    duration = max(float(duration), 1e-9)
    tau = np.clip(float(t) / duration, 0.0, 1.0)
    progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    if 0.0 < tau < 1.0:
        rate = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration
    else:
        rate = 0.0
    return progress, rate


class ReferenceTrajectory:
    """Serializable analytic trajectory selected by ``kind`` and parameters."""

    def __init__(self, name: str, kind: str, parameters: Mapping[str, Any]):
        supported = {"hold", "quintic", "circle", "helix", "coupled", "figure8"}
        if kind not in supported:
            raise ValueError(f"unsupported trajectory kind {kind!r}; choose from {sorted(supported)}")
        self.name = str(name)
        self.kind = kind
        self.parameters = dict(parameters)

    def manifest(self):
        def serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            return value
        return {
            "name": self.name,
            "kind": self.kind,
            "parameters": {key: serializable(value) for key, value in self.parameters.items()},
        }

    def sample(self, t: float) -> ReferenceState:
        t = max(float(t), 0.0)
        method = getattr(self, f"_sample_{self.kind}")
        return method(t)

    def _from_euler(self, position, velocity, rpy, rpy_rate):
        rpy = np.asarray(rpy, dtype=np.float64)
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        return ReferenceState(
            position, rotation, velocity, _world_angular_velocity(rpy, rpy_rate)
        )

    def _sample_hold(self, _t):
        p = self.parameters
        return self._from_euler(
            p["position"], np.zeros(3), p["rpy"], np.zeros(3)
        )

    def _sample_quintic(self, t):
        p = self.parameters
        progress, progress_rate = _quintic_progress(t, p["duration"])
        p0, p1 = _array(p["p0"], (3,)), _array(p["p1"], (3,))
        r0 = Rotation.from_euler("xyz", _array(p["rpy0"], (3,)))
        r1 = Rotation.from_euler("xyz", _array(p["rpy1"], (3,)))
        delta = (r0.inv() * r1).as_rotvec()
        rotation = (r0 * Rotation.from_rotvec(progress * delta)).as_matrix()
        position = p0 + progress * (p1 - p0)
        velocity = progress_rate * (p1 - p0)
        # For R=R0 Exp(s*[delta]x), the spatial angular velocity is R0*delta*s_dot.
        angular_velocity = r0.apply(delta) * progress_rate
        return ReferenceState(position, rotation, velocity, angular_velocity)

    def _sample_circle(self, t):
        p = self.parameters
        omega = 2.0 * np.pi / float(p["period"])
        phase = omega * t + float(p.get("phase", 0.0))
        center = _array(p["center"], (3,))
        radius = float(p["radius"])
        position = center + np.array([radius*np.cos(phase), radius*np.sin(phase), 0.0])
        velocity = np.array([-radius*omega*np.sin(phase), radius*omega*np.cos(phase), 0.0])
        roll_amp, pitch_amp = _array(p.get("tilt_amplitude", [0.0, 0.0]), (2,))
        rpy = np.array([
            roll_amp*np.sin(phase),
            pitch_amp*np.cos(phase),
            phase + np.pi/2.0 + float(p.get("yaw_offset", 0.0)),
        ])
        rpy_rate = np.array([
            roll_amp*omega*np.cos(phase),
            -pitch_amp*omega*np.sin(phase),
            omega,
        ])
        return self._from_euler(position, velocity, rpy, rpy_rate)

    def _sample_helix(self, t):
        p = self.parameters
        progress, progress_rate = _quintic_progress(t, p["duration"])
        turns = float(p.get("turns", 1.0))
        phase = 2.0 * np.pi * turns * progress + float(p.get("phase", 0.0))
        phase_rate = 2.0 * np.pi * turns * progress_rate
        center = _array(p["center"], (3,))
        radius, depth_change = float(p["radius"]), float(p["depth_change"])
        position = center + np.array([
            radius*np.cos(phase), radius*np.sin(phase), depth_change*progress
        ])
        velocity = np.array([
            -radius*phase_rate*np.sin(phase),
            radius*phase_rate*np.cos(phase),
            depth_change*progress_rate,
        ])
        tilt = _array(p.get("tilt", [0.0, 0.0]), (2,))
        rpy = np.array([tilt[0]*np.sin(phase), tilt[1]*np.cos(phase), phase+np.pi/2.0])
        rpy_rate = np.array([
            tilt[0]*phase_rate*np.cos(phase),
            -tilt[1]*phase_rate*np.sin(phase),
            phase_rate,
        ])
        return self._from_euler(position, velocity, rpy, rpy_rate)

    def _sample_coupled(self, t):
        p = self.parameters
        omega = 2.0 * np.pi / float(p["period"])
        phase = omega*t + float(p.get("phase", 0.0))
        center = _array(p["center"], (3,))
        amp = _array(p["position_amplitude"], (3,))
        position = center + amp*np.array([np.sin(phase), np.sin(2*phase), np.cos(phase)])
        velocity = amp*omega*np.array([np.cos(phase), 2*np.cos(2*phase), -np.sin(phase)])
        att = _array(p["attitude_amplitude"], (3,))
        rpy = att*np.array([np.sin(phase), np.sin(2*phase), np.cos(phase)])
        rpy_rate = att*omega*np.array([np.cos(phase), 2*np.cos(2*phase), -np.sin(phase)])
        return self._from_euler(position, velocity, rpy, rpy_rate)

    def _sample_figure8(self, t):
        p = self.parameters
        omega = 2.0 * np.pi / float(p["period"])
        phase = omega*t + float(p.get("phase", 0.0))
        center = _array(p["center"], (3,))
        amp = _array(p["position_amplitude"], (3,))
        position = center + np.array([
            amp[0]*np.sin(phase), amp[1]*np.sin(2*phase), amp[2]*np.cos(phase)
        ])
        velocity = np.array([
            amp[0]*omega*np.cos(phase),
            2*amp[1]*omega*np.cos(2*phase),
            -amp[2]*omega*np.sin(phase),
        ])
        att = _array(p["attitude_amplitude"], (3,))
        rpy = np.array([
            att[0]*np.sin(2*phase), att[1]*np.cos(phase), att[2]*np.sin(phase)
        ])
        rpy_rate = np.array([
            2*att[0]*omega*np.cos(2*phase),
            -att[1]*omega*np.sin(phase),
            att[2]*omega*np.cos(phase),
        ])
        return self._from_euler(position, velocity, rpy, rpy_rate)


def training_trajectory(rng: np.random.Generator) -> ReferenceTrajectory:
    """Draw one nominal training task; held-out geometries are excluded."""
    family = rng.choice(["hold", "quintic", "circle", "coupled"], p=[.20, .35, .25, .20])
    center = np.array([0.0, 0.0, 4.0]) + rng.uniform([-.5, -.5, -.25], [.5, .5, .25])
    if family == "hold":
        return ReferenceTrajectory("train_hold", "hold", {
            "position": center,
            "rpy": [0.0, 0.0, rng.uniform(-np.pi, np.pi)],
        })
    if family == "quintic":
        return ReferenceTrajectory("train_quintic", "quintic", {
            "p0": center + rng.uniform([-1., -1., -.5], [1., 1., .5]),
            "p1": center + rng.uniform([-2., -2., -1.], [2., 2., 1.]),
            "rpy0": [0.0, 0.0, rng.uniform(-np.pi, np.pi)],
            "rpy1": [0.0, 0.0, rng.uniform(-np.pi, np.pi)],
            "duration": rng.uniform(15.0, 35.0),
        })
    if family == "circle":
        return ReferenceTrajectory("train_circle", "circle", {
            "center": center,
            "radius": rng.uniform(.5, 1.5),
            "period": rng.uniform(20.0, 38.0),
            "phase": rng.uniform(-np.pi, np.pi),
            "tilt_amplitude": [0.0, 0.0],
            "yaw_offset": rng.uniform(-.35, .35),
        })
    return ReferenceTrajectory("train_coupled", "coupled", {
        "center": center,
        "period": rng.uniform(22.0, 40.0),
        "phase": rng.uniform(-np.pi, np.pi),
        "position_amplitude": rng.uniform([.4, .3, .2], [1.4, 1.0, .7]),
        "attitude_amplitude": [0.0, 0.0, rng.uniform(.25, 1.0)],
    })


def aggressive_training_trajectory(rng: np.random.Generator) -> ReferenceTrajectory:
    """Draw a fast but feasible training trajectory without using held-out geometry."""
    center = np.array([0.0, 0.0, 4.0]) + rng.uniform([-.3, -.3, -.2], [.3, .3, .2])
    family = rng.choice(["circle", "coupled"], p=[0.35, 0.65])
    if family == "circle":
        return ReferenceTrajectory("train_aggressive_circle", "circle", {
            "center": center,
            "radius": rng.uniform(.8, 1.4),
            "period": rng.uniform(18.0, 25.0),
            "phase": rng.uniform(-np.pi, np.pi),
            "tilt_amplitude": [0.0, 0.0],
            "yaw_offset": rng.uniform(-.30, .30),
        })
    return ReferenceTrajectory("train_aggressive_coupled", "coupled", {
        "center": center,
        "period": rng.uniform(18.0, 26.0),
        "phase": rng.uniform(-np.pi, np.pi),
        "position_amplitude": rng.uniform([.7, .5, .25], [1.25, .9, .6]),
        "attitude_amplitude": [0.0, 0.0, rng.uniform(.55, .95)],
    })


def adaptation_trajectories():
    """Fixed training-only mini-bank for process-weight updates.

    Parameters intentionally differ from the final evaluation manifest.
    """
    d = np.deg2rad
    return [
        ReferenceTrajectory("adapt_hold", "hold", {
            "position": [.25, -.20, 4.1], "rpy": d([0, 0, 35]),
        }),
        ReferenceTrajectory("adapt_quintic", "quintic", {
            "p0": [-.8, .4, 3.7], "p1": [1.1, -.9, 4.4],
            "rpy0": d([0, 0, -55]), "rpy1": d([0, 0, 75]), "duration": 26.0,
        }),
        ReferenceTrajectory("adapt_circle", "circle", {
            "center": [.1, -.1, 4.0], "radius": .9, "period": 24.0,
            "phase": -.45, "tilt_amplitude": [0, 0], "yaw_offset": .15,
        }),
        ReferenceTrajectory("adapt_coupled", "coupled", {
            "center": [0, 0, 4.0], "period": 22.0, "phase": -.2,
            "position_amplitude": [.9, .6, .35],
            "attitude_amplitude": d([0, 0, 42]),
        }),
    ]


def evaluation_trajectories():
    """Fixed nominal suite containing both seen families and unseen geometries."""
    d = np.deg2rad
    return [
        ReferenceTrajectory("hold_level", "hold", {"position": [0, 0, 4], "rpy": [0, 0, 0]}),
        ReferenceTrajectory("hold_heading", "hold", {"position": [.5, -.5, 4.2], "rpy": d([0, 0, 70])}),
        ReferenceTrajectory("quintic_coupled", "quintic", {
            "p0": [-1, -.5, 3.5], "p1": [1.5, 1., 4.7],
            "rpy0": d([0, 0, -80]), "rpy1": d([0, 0, 100]), "duration": 30.0,
        }),
        ReferenceTrajectory("circle_level", "circle", {
            "center": [0, 0, 4], "radius": 1.2, "period": 30.,
            "phase": .2, "tilt_amplitude": [0, 0], "yaw_offset": 0.,
        }),
        ReferenceTrajectory("circle_attitude", "circle", {
            "center": [0, 0, 4], "radius": 1.0, "period": 27.,
            "phase": -.7, "tilt_amplitude": [0, 0], "yaw_offset": .25,
        }),
        ReferenceTrajectory("helix_unseen", "helix", {
            "center": [0, 0, 3.3], "radius": 1.0, "depth_change": 1.4,
            "duration": 36., "turns": 1.25, "phase": .4, "tilt": [0, 0],
        }),
        ReferenceTrajectory("figure8_unseen", "figure8", {
            "center": [0, 0, 4], "period": 32.,
            "position_amplitude": [1.4, .8, .5],
            "attitude_amplitude": d([0, 0, 55]), "phase": -.3,
        }),
        ReferenceTrajectory("coupled_fast", "coupled", {
            "center": [0, 0, 4], "period": 20., "phase": .5,
            "position_amplitude": [1.0, .7, .4],
            "attitude_amplitude": d([0, 0, 50]),
        }),
    ]

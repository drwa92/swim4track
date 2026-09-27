"""Validated nonlinear BlueROV2 Heavy plant used by the REEF baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import cont2discrete, tf2ss


def _v(x):
    return np.asarray(x, dtype=np.float64)


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def wrap_angles(x):
    return (np.asarray(x) + np.pi) % (2. * np.pi) - np.pi


def rotation_body_to_world(euler):
    roll, pitch, yaw = euler
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch),
        np.cos(yaw), np.sin(yaw))
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr]])


def euler_rate_matrix(euler):
    roll, pitch, _ = euler
    cr, sr, cp = np.cos(roll), np.sin(roll), np.cos(pitch)
    cp = np.copysign(max(abs(cp), 1e-6), cp if cp else 1.)
    tp = np.sin(pitch) / cp
    return np.array([[1., sr*tp, cr*tp], [0., cr, -sr],
                     [0., sr/cp, cr/cp]])


def kinematic_jacobian(eta):
    out = np.zeros((6, 6))
    out[:3, :3] = rotation_body_to_world(eta[3:])
    out[3:, 3:] = euler_rate_matrix(eta[3:])
    return out


@dataclass(frozen=True)
class VehicleParameters:
    gravity: float = 9.82
    water_density: float = 1000.
    mass: float = 13.5
    displaced_volume: float = .0134
    inertia_diag: np.ndarray = field(default_factory=lambda: _v([.26, .23, .37]))
    added_mass_diag: np.ndarray = field(default_factory=lambda: _v([6.36, 7.12, 18.68, .189, .135, .222]))
    linear_damping: np.ndarray = field(default_factory=lambda: _v([13.7, 0., 33., 0., .8, 0.]))
    quadratic_damping: np.ndarray = field(default_factory=lambda: _v([141., 217., 190., 1.19, .47, 1.5]))
    center_of_gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    center_of_buoyancy: np.ndarray = field(default_factory=lambda: _v([0., 0., -.01]))

    @property
    def mass_matrix(self):
        return np.diag(np.r_[np.full(3, self.mass), self.inertia_diag] + self.added_mass_diag)

    @property
    def allocation_matrix(self):
        p0 = _v([.156, .111, .085])
        d0 = _v([1/np.sqrt(2), -1/np.sqrt(2), 0.])
        alphas = [0., 5.05, 1.91, np.pi]
        betas = [0., np.pi/2, 3*np.pi/2, np.pi]
        pv = _v([.120, .218, 0.])
        gammas = [0., 4.15, 1.01, np.pi]
        def rz(a, v):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ v
        positions = np.vstack([rz(a, p0) for a in alphas] + [rz(g, pv) for g in gammas])
        directions = np.vstack([rz(b, d0) for b in betas] + [_v([0., 0., -1.]) for _ in gammas])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return np.vstack((directions.T, np.cross(positions, directions).T))


def t200_force(command):
    v = np.asarray(command)
    return -140.3*v**9 + 389.9*v**7 - 404.1*v**5 + 176.*v**3 + 8.9*v


def stonefish_force(command):
    """Audited Stonefish BlueROV2 static thruster law in newtons."""
    v = np.asarray(command)
    return 16.853640533657778 * v * np.abs(v)


class T200ActuatorBank:
    """Eight third-order T200 models; command is clipped to the validated 0.60."""
    command_limit = .60
    applied_force_limit = .60 * 30.4
    force_rate = 41.6337657351 * 30.4

    def __init__(self, stonefish_blend=0., force_scale=1.):
        if not 0. <= stonefish_blend <= 1.:
            raise ValueError("stonefish_blend must lie in [0, 1]")
        if force_scale <= 0.:
            raise ValueError("force_scale must be positive")
        self.stonefish_blend = float(stonefish_blend)
        self.force_scale = float(force_scale)
        a, b, c, d = tf2ss([6136., 108700.], [1., 89., 9258., 108700.])
        self.a, self.b, self.c, self.d = a, b.reshape(3, 1), c.reshape(1, 3), float(d[0, 0])
        self.states = np.zeros((8, 3)); self.last_force = np.zeros(8); self.cache = {}

    def reset(self):
        self.states.fill(0.); self.last_force.fill(0.)

    def update(self, command, dt, health=None):
        command = np.clip(np.asarray(command), -self.command_limit, self.command_limit)
        health = np.ones(8) if health is None else np.clip(np.asarray(health), 0., 1.)
        target = self.force_scale * (
            (1.-self.stonefish_blend)*t200_force(command)
            + self.stonefish_blend*stonefish_force(command)
        )
        key = round(float(dt), 10)
        if key not in self.cache:
            ad, bd, _, _, _ = cont2discrete((self.a, self.b, self.c, [[self.d]]), dt, method="zoh")
            self.cache[key] = ad, bd.reshape(3, 1)
        ad, bd = self.cache[key]
        self.states = self.states @ ad.T + target[:, None] @ bd.T
        dynamic = (self.states @ self.c.T).ravel() + self.d * target
        dynamic = self.last_force + np.clip(dynamic-self.last_force, -self.force_rate*dt, self.force_rate*dt)
        self.last_force = dynamic
        limit = health * self.applied_force_limit
        return np.clip(health * dynamic, -limit, limit)


class ValidatedBlueROV2Heavy:
    def __init__(self, parameters=None):
        self.p = parameters or VehicleParameters()
        self.mass = self.p.mass_matrix
        self.mass_inverse = np.linalg.inv(self.mass)

    def coriolis(self, nu):
        momentum = self.mass @ nu
        out = np.zeros((6, 6))
        out[:3, 3:] = -skew(momentum[:3]); out[3:, :3] = -skew(momentum[:3])
        out[3:, 3:] = -skew(momentum[3:])
        return out

    def damping(self, nu):
        return np.diag(self.p.linear_damping + self.p.quadratic_damping*np.abs(nu))

    def restoring(self, eta):
        r = rotation_body_to_world(eta[3:])
        weight = r.T @ _v([0., 0., self.p.mass*self.p.gravity])
        buoyancy = r.T @ _v([0., 0., -self.p.water_density*self.p.gravity*self.p.displaced_volume])
        force = weight + buoyancy
        moment = np.cross(self.p.center_of_gravity, weight) + np.cross(self.p.center_of_buoyancy, buoyancy)
        return -np.r_[force, moment]

    def derivative(self, state, wrench, current_velocity_ned=None):
        eta, nu = state[:6], state[6:]
        current_velocity_ned = (
            np.zeros(3, dtype=np.float64)
            if current_velocity_ned is None
            else _v(current_velocity_ned)
        )
        if current_velocity_ned.shape != (3,) or not np.all(np.isfinite(current_velocity_ned)):
            raise ValueError("current_velocity_ned must contain three finite values")
        current_body = rotation_body_to_world(eta[3:]).T @ current_velocity_ned
        relative_velocity = nu.copy()
        relative_velocity[:3] -= current_body
        eta_dot = kinematic_jacobian(eta) @ nu
        nu_dot = self.mass_inverse @ (
            wrench
            - self.coriolis(nu) @ nu
            - self.damping(relative_velocity) @ relative_velocity
            - self.restoring(eta)
        )
        return np.r_[eta_dot, nu_dot]

    def step(self, state, wrench, dt, current_velocity_ned=None):
        k1 = self.derivative(state, wrench, current_velocity_ned)
        k2 = self.derivative(state + .5*dt*k1, wrench, current_velocity_ned)
        k3 = self.derivative(state + .5*dt*k2, wrench, current_velocity_ned)
        k4 = self.derivative(state + dt*k3, wrench, current_velocity_ned)
        out = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.
        out[3:6] = wrap_angles(out[3:6])
        return out

"""Simulator-independent REEF task definition and reward contract."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .plant import ValidatedBlueROV2Heavy, T200ActuatorBank, VehicleParameters

POSITION_SCALE = np.array([6., 6., 3.])
TWIST_SCALE = np.array([1.5, 1.5, 1., 1., 1., 1.])
REWARD_WEIGHTS = np.array([-4., -4., -3., -1.8, -.4, -.3])
# The installed Stonefish Heavy uses the same T1...T8 order as the training
# plant, but a positive direct command generates the opposite force direction
# on every channel.  Keep this deployment transform explicit and testable.
STONEFISH_COMMAND_SIGNS = -np.ones(8, dtype=np.float64)


def attitude_error_vector(current_rpy, desired_rpy):
    """Log(R_current.T R_desired), matching Swim4Real's SO(3) error."""
    rc = Rotation.from_euler("xyz", current_rpy)
    rd = Rotation.from_euler("xyz", desired_rpy)
    return (rc.inv() * rd).as_rotvec()


def observation(eta, nu, goal, previous_action):
    pos_error = eta[:3] - goal[:3]
    rot_error = attitude_error_vector(eta[3:], goal[3:])
    return np.clip(np.r_[pos_error/POSITION_SCALE, rot_error/np.pi,
                         np.asarray(eta[3:])/np.pi, np.asarray(nu)/TWIST_SCALE,
                         previous_action], -1., 1.).astype(np.float32)


def reef_reward(eta, goal, action, previous_action):
    p = np.abs(np.asarray(eta[:3]) - np.asarray(goal[:3]))
    theta = np.linalg.norm(attitude_error_vector(eta[3:], goal[3:]))
    terms = np.array([p[0], p[1], p[2], theta,
                      np.abs(action-previous_action).sum(), np.abs(action).sum()])
    return float(REWARD_WEIGHTS @ terms), terms


def attitude_grid_36():
    """Declared 36-goal grid; the paper did not publish its exact angles."""
    return np.array([[r, p, y] for r in (0., np.pi/4)
                     for p in (-np.pi/4, 0., np.pi/4)
                     for y in (-np.pi, -2*np.pi/3, -np.pi/3, 0., np.pi/3, 2*np.pi/3)])


def transfer_attitude_grid_36():
    """Feasible common 36-goal grid for plant-to-Stonefish transfer.

    The audited Stonefish configuration cannot statically hold the baseline
    grid's +/-45 degree pitch commands at the project's 0.60 actuator limit.
    A symmetric +/-15 degree roll/pitch envelope preserves a 36-pose task while
    remaining inside the identified endpoint's attitude authority.
    """
    tilt = np.deg2rad((-15., 0., 15.))
    yaw = np.deg2rad((-180., -90., 0., 90.))
    return np.array([[r, p, y] for r in tilt for p in tilt for y in yaw])


@dataclass
class ReefCore:
    control_dt: float = .1
    physics_dt: float = .02
    max_steps: int = 400
    command_limit: float = .60
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    domain_randomization: bool = False
    task: str = "baseline36"

    def __post_init__(self):
        if self.task not in ("baseline36", "transfer36"):
            raise ValueError("task must be 'baseline36' or 'transfer36'")
        ratio = self.control_dt/self.physics_dt
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        self.substeps = int(round(ratio))
        self.vehicle = ValidatedBlueROV2Heavy()
        self.actuators = T200ActuatorBank()
        self.health = np.ones(8)
        self.domain = {"alpha": 0., "stonefish_blend": 0., "force_scale": 1.}
        self.state = np.zeros(12); self.goal = np.zeros(6)
        self.previous_action = np.zeros(8); self.steps = 0

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def configure_domain(self, alpha, jitter=True):
        """Configure one point on the validated-plant-to-Stonefish continuum.

        One interpolation coordinate keeps mass, buoyancy geometry, inertia, and
        actuator family mutually consistent. Small independent multipliers cover
        identification uncertainty without producing implausible combinations.
        """
        alpha = float(alpha)
        if not 0. <= alpha <= 1.:
            raise ValueError("domain alpha must lie in [0, 1]")
        nominal = VehicleParameters()
        sf_mass, sf_density, sf_volume = 11.189, 1031., .01092
        sf_inertia = np.array([.154, .217, .356])
        sf_cg = np.array([-.005477490357, -9.243486e-05, .077353626001])
        sf_cb = np.array([-.005, 0., 0.])
        lerp = lambda a, b: (1.-alpha)*np.asarray(a) + alpha*np.asarray(b)
        mass_jitter = self.rng.uniform(.97, 1.03) if jitter else 1.
        volume_jitter = self.rng.uniform(.985, 1.015) if jitter else 1.
        inertia_jitter = self.rng.uniform(.9, 1.1, 3) if jitter else np.ones(3)
        added_mass_jitter = self.rng.uniform(.85, 1.15, 6) if jitter else np.ones(6)
        linear_jitter = self.rng.uniform(.8, 1.2, 6) if jitter else np.ones(6)
        quadratic_jitter = self.rng.uniform(.8, 1.2, 6) if jitter else np.ones(6)
        mass = float(lerp(nominal.mass, sf_mass) * mass_jitter)
        density = float(lerp(nominal.water_density, sf_density))
        volume = float(lerp(nominal.displaced_volume, sf_volume) * volume_jitter)
        inertia = lerp(nominal.inertia_diag, sf_inertia) * inertia_jitter
        added_mass = nominal.added_mass_diag * added_mass_jitter
        linear_damping = nominal.linear_damping * linear_jitter
        quadratic_damping = nominal.quadratic_damping * quadratic_jitter
        parameters = VehicleParameters(
            gravity=float(lerp(nominal.gravity, 9.81)),
            water_density=density,
            mass=mass,
            displaced_volume=volume,
            inertia_diag=inertia,
            added_mass_diag=added_mass,
            linear_damping=linear_damping,
            quadratic_damping=quadratic_damping,
            center_of_gravity=lerp(nominal.center_of_gravity, sf_cg),
            center_of_buoyancy=lerp(nominal.center_of_buoyancy, sf_cb),
        )
        force_scale = float(self.rng.uniform(.9, 1.1)) if jitter else 1.
        self.vehicle = ValidatedBlueROV2Heavy(parameters)
        self.actuators = T200ActuatorBank(alpha, force_scale)
        self.domain = {
            "alpha": alpha,
            "mass": mass,
            "water_density": density,
            "displaced_volume": volume,
            "cg_z": float(parameters.center_of_gravity[2]),
            "cb_z": float(parameters.center_of_buoyancy[2]),
            "stonefish_blend": alpha,
            "force_scale": force_scale,
            "jitter": bool(jitter),
        }

    def randomize_domain(self):
        """Sample a structured, jittered plant-to-Stonefish domain."""
        self.configure_domain(self.rng.uniform(0., 1.), jitter=True)

    def reset(self, seed=None, state=None, goal=None):
        if seed is not None: self.seed(seed)
        if self.domain_randomization: self.randomize_domain()
        self.actuators.reset(); self.steps = 0; self.previous_action.fill(0.)
        if goal is None:
            grid = transfer_attitude_grid_36() if self.task == "transfer36" else attitude_grid_36()
            goal = np.r_[0., 0., 4., grid[self.rng.integers(36)]]
        self.goal = np.asarray(goal, dtype=float).copy()
        if state is None:
            if self.task == "transfer36":
                position = self.goal[:3] + self.rng.uniform([-2., -2., -1.5], [2., 2., 1.5])
                attitude = np.array([
                    self.rng.uniform(-np.deg2rad(2.), np.deg2rad(2.)),
                    self.rng.uniform(-np.deg2rad(2.), np.deg2rad(2.)),
                    self.goal[5] + self.rng.uniform(-np.pi/4, np.pi/4),
                ])
                eta = np.r_[position, attitude]
            else:
                eta = np.r_[self.rng.uniform([-6., -6., 1.], [6., 6., 7.]),
                            self.rng.uniform([-np.pi/4, -np.pi/4, -np.pi],
                                             [np.pi/4, np.pi/4, np.pi])]
            state = np.r_[eta, np.zeros(6)]
        self.state = np.asarray(state, dtype=float).copy()
        return observation(self.state[:6], self.state[6:], self.goal, self.previous_action)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1., 1.)
        command = self.command_limit * action
        for _ in range(self.substeps):
            force = self.actuators.update(command, self.physics_dt, self.health)
            self.state = self.vehicle.step(self.state, self.vehicle.p.allocation_matrix @ force, self.physics_dt)
        finite = bool(np.all(np.isfinite(self.state)))
        unsafe = (not finite) or np.linalg.norm(self.state[:3]-self.goal[:3]) > 50.
        if finite:
            reward, terms = reef_reward(self.state[:6], self.goal, action, self.previous_action)
            next_observation = observation(self.state[:6], self.state[6:], self.goal, action)
        else:
            reward, terms = -1000., np.full(6, np.inf)
            next_observation = np.zeros(23, dtype=np.float32)
        self.previous_action = action.copy(); self.steps += 1
        truncated = self.steps >= self.max_steps
        if unsafe and finite: reward -= 1000.
        info = {"reward_terms": terms, "command": command.copy(), "goal": self.goal.copy(), "unsafe": unsafe}
        return next_observation, reward, unsafe, truncated, info

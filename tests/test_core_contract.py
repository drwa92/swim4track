"""Scientific interface regression tests using only NumPy, SciPy and unittest."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from swim4track.actuator_dynamics import FirstOrderRateLimitedActuator
from swim4track.core import STONEFISH_COMMAND_SIGNS
from swim4track.deployment import map_policy_action_to_stonefish, STONEFISH_FORCE_COEFFICIENT_N
from swim4track.deployment_reference import ramped_trajectory_clock, smooth_acquisition_reference
from swim4track.pid import load_frozen_pid, protocol_hash, PROTOCOL_SHA256
from swim4track.plant import ValidatedBlueROV2Heavy, T200ActuatorBank, t200_force
from swim4track.tracking import Swim4TrackCore, trajectory_observation
from swim4track.trajectory_reference import ReferenceState, evaluation_trajectories


class PlantContractTests(unittest.TestCase):
    def test_mass_coriolis_and_allocation(self):
        plant = ValidatedBlueROV2Heavy()
        self.assertTrue(np.all(np.linalg.eigvalsh(plant.mass) > 0))
        self.assertEqual(np.linalg.matrix_rank(plant.p.allocation_matrix), 6)
        velocity = np.array([.3, -.2, .1, .04, .03, -.1])
        coriolis = plant.coriolis(velocity)
        assert_allclose(coriolis + coriolis.T, 0, atol=1e-14)
        self.assertAlmostEqual(float(velocity @ coriolis @ velocity), 0, places=14)

    def test_restoring_compensation_and_failed_thruster(self):
        plant = ValidatedBlueROV2Heavy()
        state = np.array([0., 0., 4., .08, -.06, .2, 0, 0, 0, 0, 0, 0])
        assert_allclose(plant.derivative(state, plant.restoring(state[:6])), 0, atol=1e-13)
        actuator = T200ActuatorBank()
        health = np.ones(8)
        health[2] = 0
        for _ in range(25):
            force = actuator.update(np.full(8, .4), .02, health)
        self.assertEqual(force[2], 0)
        self.assertTrue(np.all(np.delete(force, 2) > 0))
        self.assertLessEqual(float(np.max(np.abs(force))), actuator.applied_force_limit)

    def test_seeded_core_is_deterministic_and_rejects_invalid_action(self):
        first, second = Swim4TrackCore(), Swim4TrackCore()
        assert_allclose(first.reset(seed=1729), second.reset(seed=1729), atol=0, rtol=0)
        for _ in range(10):
            a, b = first.step(np.zeros(8)), second.step(np.zeros(8))
            assert_allclose(first.state, second.state, atol=0, rtol=0)
            assert_allclose(a[0], b[0], atol=0, rtol=0)
            self.assertEqual(a[1], b[1])
        with self.assertRaises(ValueError):
            first.step(np.full(8, np.nan))
        with self.assertRaises(ValueError):
            first.step(np.zeros(7))


class ObservationAndReferenceTests(unittest.TestCase):
    def test_observation_is_invariant_to_horizontal_frame_change(self):
        eta = np.array([.8, -.4, 4.2, .08, -.06, .7])
        nu = np.array([.2, -.1, .03, .05, -.04, .12])
        reference = evaluation_trajectories()[3].sample(7.25)
        action = np.linspace(-.3, .3, 8)
        observed = trajectory_observation(eta, nu, reference, action)
        q = Rotation.from_euler("z", 1.1).as_matrix()
        shift = np.array([2., -3., 0.])
        rotated_eta = np.r_[q @ eta[:3] + shift,
                            Rotation.from_matrix(q @ Rotation.from_euler("xyz", eta[3:]).as_matrix()).as_euler("xyz")]
        rotated_reference = ReferenceState(
            q @ reference.position_ned + shift,
            q @ reference.rotation_body_to_ned,
            q @ reference.linear_velocity_ned,
            q @ reference.angular_velocity_ned,
        )
        transformed = trajectory_observation(rotated_eta, nu, rotated_reference, action)
        self.assertEqual(observed.shape, (23,))
        self.assertEqual(observed.dtype, np.float32)
        assert_allclose(observed, transformed, atol=2e-7)
        assert_allclose(observed[-8:], action, atol=2e-8)

    def test_reference_velocities_are_derivatives(self):
        t, h = 7.25, 1e-5
        for path in evaluation_trajectories():
            with self.subTest(path=path.name):
                before, reference, after = path.sample(t-h), path.sample(t), path.sample(t+h)
                assert_allclose((after.position_ned-before.position_ned)/(2*h), reference.linear_velocity_ned, atol=2e-8)
                spatial_increment = Rotation.from_matrix(after.rotation_body_to_ned @ before.rotation_body_to_ned.T).as_rotvec()
                assert_allclose(spatial_increment/(2*h), reference.angular_velocity_ned, atol=2e-8)

    def test_acquisition_endpoints_and_ramped_clock(self):
        target = evaluation_trajectories()[3].sample(0)
        initial = np.array([.1, -.2, 1.1, 0., 0., -.4])
        before = smooth_acquisition_reference(initial, target, 0, 15)
        after = smooth_acquisition_reference(initial, target, 15, 15)
        assert_allclose(before.position_ned, initial[:3])
        assert_allclose(after.position_ned, target.position_ned)
        assert_allclose(after.rotation_body_to_ned, target.rotation_body_to_ned, atol=1e-14)
        assert_allclose(before.linear_velocity_ned, 0)
        assert_allclose(after.linear_velocity_ned, 0)
        assert_allclose(ramped_trajectory_clock(5, trajectory_time_scale=2, ramp_duration_s=5), [1.25, .5])
        h = 1e-5
        left = ramped_trajectory_clock(5-h, trajectory_time_scale=2, ramp_duration_s=5)
        right = ramped_trajectory_clock(5+h, trajectory_time_scale=2, ramp_duration_s=5)
        self.assertAlmostEqual((right[0]-left[0])/(2*h), .5, places=7)


class DeploymentContractTests(unittest.TestCase):
    def test_force_equivalent_mapping_and_sign(self):
        action = np.linspace(-.7, .7, 8)
        command = map_policy_action_to_stonefish(action, command_signs=STONEFISH_COMMAND_SIGNS)
        realized_force = STONEFISH_COMMAND_SIGNS * STONEFISH_FORCE_COEFFICIENT_N * command * np.abs(command)
        assert_allclose(realized_force, t200_force(.6*action), atol=1e-12)
        with self.assertRaises(ValueError):
            map_policy_action_to_stonefish(action, command_signs=np.zeros(8))
        with self.assertRaises(ValueError):
            map_policy_action_to_stonefish(np.full(8, np.nan), command_signs=STONEFISH_COMMAND_SIGNS)

    def test_deployment_filter_respects_slew_and_bounds(self):
        filt = FirstOrderRateLimitedActuator(size=8, control_dt_s=.1, time_constant_s=.15,
                                            rate_limit_per_s=2., command_limit=1.)
        previous = np.zeros(8)
        for value in [1.]*15 + [-1.]*15:
            output = filt.step(np.full(8, value))
            self.assertLessEqual(float(np.max(np.abs(output-previous))), .2+1e-12)
            self.assertLessEqual(float(np.max(np.abs(output))), 1.)
            previous = output
        filt.reset()
        assert_allclose(filt.state, 0)


class FrozenPIDTests(unittest.TestCase):
    def test_frozen_pid_load_and_closed_loop(self):
        controller, metadata = load_frozen_pid()
        self.assertEqual(metadata["candidate_name"], "pid_kp_high")
        self.assertEqual(metadata["protocol_sha256"], PROTOCOL_SHA256)
        assert_allclose(controller.gains.kp_pose, 1.2*np.array([38.,38.,48.,8.,8.,12.]))
        core = Swim4TrackCore()
        core.reset(seed=17, trajectory=evaluation_trajectories()[0])
        for _ in range(30):
            action = controller.act(core.state, core.reference)
            self.assertTrue(np.all(np.isfinite(action)))
            self.assertLessEqual(float(np.max(np.abs(action))), 1.)
            _, _, terminated, _, _ = core.step(action)
            self.assertFalse(terminated)
        controller.reset()
        assert_allclose(controller.integral, 0)

    def test_protocol_hash_is_canonical_and_tamper_is_rejected(self):
        import swim4track.pid as pid_module
        source = Path(pid_module.__file__).parent / "config"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            for name in ["pid_frozen.json", "classical_benchmarks_v1.json"]:
                shutil.copyfile(source/name, target/name)
            protocol = target / "classical_benchmarks_v1.json"
            data = json.loads(protocol.read_text())
            protocol.write_text(json.dumps(data, sort_keys=True))
            self.assertEqual(protocol_hash(protocol), PROTOCOL_SHA256)
            load_frozen_pid(target)
            data["pid_candidates"][2]["kp_scale"] = 10
            protocol.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, "canonical SHA-256 mismatch"):
                load_frozen_pid(target)


if __name__ == "__main__":
    unittest.main()

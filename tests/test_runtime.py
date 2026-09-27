"""Substantive runtime safety/state tests; no ROS installation is required."""
import unittest

import numpy as np

from swim4track.deployment import map_policy_action_to_stonefish
from swim4track.runtime import RuntimeConfig, TrackingRuntime


class Actor:
    def __init__(self, action=None):
        self.action = np.full(8, 0.7) if action is None else np.asarray(action)
        self.observations = []

    def predict(self, observation):
        self.observations.append(observation.copy())
        return self.action.copy()


class RuntimeTests(unittest.TestCase):
    def make(self, action=None, **kwargs):
        actor = Actor(action)
        settings = dict(acquisition_s=0.0, command_soft_start_s=0.0, evaluation_s=1.0)
        settings.update(kwargs)
        runtime = TrackingRuntime(actor, RuntimeConfig(**settings))
        state = np.zeros(12)
        state[2] = 1.5
        runtime.update_odometry(state, 10.0)
        return runtime, actor, state

    def test_disabled_silent_and_explicit_fresh_start(self):
        runtime, actor, state = self.make()
        self.assertIsNone(runtime.step(10.1).command)
        self.assertFalse(runtime.start(10.6)[0])
        runtime.update_odometry(state, 10.6)
        self.assertTrue(runtime.start(10.6)[0])
        self.assertFalse(runtime.start(10.6)[0])
        self.assertEqual(len(actor.observations), 0)

    def test_raw_action_memory_and_common_deployment_map(self):
        runtime, actor, state = self.make()
        runtime.start(10.0)
        first = runtime.step(10.0)
        expected = map_policy_action_to_stonefish(actor.action, command_signs=-np.ones(8))
        np.testing.assert_allclose(first.desired_command, expected)
        self.assertLessEqual(np.max(np.abs(first.command)), 0.2+1e-12)
        self.assertTrue(np.all(first.command < 0))
        runtime.update_odometry(state, 10.1)
        second = runtime.step(10.1)
        np.testing.assert_allclose(actor.observations[1][-8:], actor.action, atol=1e-7)
        self.assertFalse(np.allclose(actor.observations[1][-8:], first.command))
        self.assertLessEqual(np.max(np.abs(second.command-first.command)), 0.2+1e-12)

    def test_stale_odometry_aborts_and_never_auto_resumes(self):
        runtime, actor, state = self.make()
        runtime.start(10.0)
        runtime.step(10.2)
        stale = runtime.step(10.51)
        self.assertEqual(stale.status["phase"], "aborted")
        self.assertIn("stale", stale.status["reason"])
        np.testing.assert_array_equal(stale.command, np.zeros(8))
        runtime.update_odometry(state, 10.6)
        self.assertIsNone(runtime.step(10.6).command)
        self.assertTrue(runtime.start(10.6)[0])

    def test_inference_exception_and_nan_output_stop(self):
        for action in ([float("nan")]*8, [0.0]*7):
            runtime, actor, state = self.make(action)
            runtime.start(10.0)
            result = runtime.step(10.1)
            self.assertEqual(result.status["phase"], "aborted")
            np.testing.assert_array_equal(result.command, np.zeros(8))
        runtime, actor, state = self.make()
        actor.predict = lambda _: (_ for _ in ()).throw(RuntimeError("test failure"))
        runtime.start(10.0)
        self.assertIn("test failure", runtime.step(10.1).status["reason"])

    def test_clock_gap_and_backward_clock_abort(self):
        runtime, _, state = self.make()
        runtime.start(10.0)
        runtime.update_odometry(state, 10.8)
        self.assertIn("callback gap", runtime.step(10.8).status["reason"])
        runtime, _, state = self.make()
        runtime.start(10.0)
        self.assertIn("backward", runtime.step(9.9).status["reason"])

    def test_completion_stops_output_once(self):
        runtime, _, state = self.make(evaluation_s=0.3)
        runtime.start(10.0)
        for now in (10.0, 10.1, 10.2):
            runtime.update_odometry(state, now)
            self.assertIsNotNone(runtime.step(now).action)
        runtime.update_odometry(state, 10.31)
        final = runtime.step(10.31)
        self.assertEqual(final.status["phase"], "completed")
        np.testing.assert_array_equal(final.command, np.zeros(8))
        self.assertIsNone(runtime.step(10.4).command)

    def test_restart_has_new_run_identity(self):
        runtime, _, state = self.make()
        self.assertIsNone(runtime.status()["run_id"])
        self.assertTrue(runtime.start(10.0)[0])
        first_id = runtime.status()["run_id"]
        self.assertTrue(first_id)
        self.assertEqual(runtime.stop().status["run_id"], first_id)
        runtime.update_odometry(state, 10.1)
        self.assertTrue(runtime.start(10.1)[0])
        self.assertNotEqual(runtime.status()["run_id"], first_id)
        self.assertEqual(runtime.status()["elapsed_s"], 0.0)

    def test_nonfinite_odometry_does_not_refresh_watchdog(self):
        runtime, _, state = self.make()
        state[0] = float("nan")
        self.assertFalse(runtime.update_odometry(state, 10.9))
        self.assertEqual(runtime.received_at, 10.0)

    def test_workspace_limits_and_configuration(self):
        runtime, _, state = self.make()
        state[2] = -0.1
        runtime.update_odometry(state, 10.0)
        self.assertFalse(runtime.start(10.0)[0])
        for change in ({"control_dt_s": 0.2}, {"evaluation_s": float("nan")},
                       {"acquisition_s": -1}, {"bottom_z_m": 0.0}):
            with self.assertRaises(ValueError):
                RuntimeConfig(**change)

    def test_acquisition_starts_at_measured_pose(self):
        runtime, _, state = self.make(acquisition_s=30.0)
        state[:3] = [0.2, -0.1, 1.0]
        runtime.update_odometry(state, 10.0)
        runtime.start(10.0)
        reference = runtime.scheduled_reference(0.0)
        np.testing.assert_allclose(reference.position_ned, state[:3])
        np.testing.assert_allclose(reference.linear_velocity_ned, np.zeros(3))
        np.testing.assert_allclose(runtime.scheduled_reference(30.0).position_ned, [0., 0., 1.5])

    def test_real_frozen_pid_uses_same_interface(self):
        from swim4track.pid import load_frozen_pid
        pid, metadata = load_frozen_pid()
        runtime = TrackingRuntime(pid, RuntimeConfig(controller_kind="pid", acquisition_s=0.0))
        state = np.zeros(12)
        state[2] = 1.5
        runtime.update_odometry(state, 1.0)
        self.assertTrue(runtime.start(1.0)[0])
        result = runtime.step(1.1)
        self.assertEqual(result.command.shape, (8,))
        self.assertTrue(np.all(np.isfinite(result.command)))


if __name__ == "__main__":
    unittest.main()

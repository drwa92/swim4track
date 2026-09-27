import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from swim4track.demo import run_demo, save_rollout


class DemoTest(unittest.TestCase):
    def test_pid_rollout_records_states_and_command_intervals(self):
        data, summary = run_demo(seconds=1.0)
        self.assertTrue(summary["completed"])
        self.assertEqual(data["state"].shape, (11, 12))
        self.assertEqual(data["raw_action"].shape, (10, 8))
        np.testing.assert_allclose(data["plant_command"], .6 * data["raw_action"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            save_rollout(output, data, summary)
            self.assertTrue(json.loads((output / "summary.json").read_text())["completed"])
            self.assertEqual(len((output / "commands.csv").read_text().splitlines()), 11)
            with self.assertRaises(FileExistsError):
                save_rollout(output, data, summary)

    def test_bad_duration_is_not_silently_rounded(self):
        for seconds in (0, -1, 1e-12, .15, float("nan")):
            with self.assertRaises(ValueError):
                run_demo(seconds=seconds)

    def test_model_options_cannot_silently_run_pid(self):
        for model_options in ({"model_path": "my_model.zip"}, {"model_sha256": "0"*64}):
            with self.assertRaisesRegex(ValueError, "controller.*learned"):
                run_demo(controller_kind="pid", seconds=1.0, **model_options)

    def test_nonfinite_failure_retains_outcome_and_raw_trace(self):
        def failed_step(core, action):
            core.state[:] = np.nan
            core.time = .1
            return np.zeros(23), -1000.0, True, False, {"command": .6 * action}
        with patch("swim4track.demo.Swim4TrackCore.step", failed_step):
            data, summary = run_demo(seconds=1.0)
        self.assertFalse(summary["completed"])
        self.assertTrue(summary["terminated_unsafe"])
        self.assertIsNone(summary["position_rmse_m"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "failed_run"
            save_rollout(output, data, summary)
            self.assertTrue(json.loads((output / "summary.json").read_text())["terminated_unsafe"])
            with np.load(output / "rollout.npz") as saved:
                self.assertTrue(np.isnan(saved["state"][-1]).all())

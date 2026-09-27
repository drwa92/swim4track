import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from swim4track.policy import FrozenPolicy, default_model_path, MODEL_SHA256, validate_observation, verify_checkpoint, main
from swim4track.deployment import file_sha256


class PolicyContractTest(unittest.TestCase):
    def test_bundled_model_is_the_frozen_checkpoint(self):
        self.assertEqual(file_sha256(default_model_path()), MODEL_SHA256)

    def test_corruption_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            path.write_bytes(b"not the model")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                FrozenPolicy(path)

    def test_empty_digest_is_not_treated_as_default(self):
        with self.assertRaisesRegex(ValueError, "64-character"):
            verify_checkpoint(expected_sha256="")

    def test_model_lookup_for_target_install(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            checkpoint = target / "share/swim4track/models" / default_model_path().name
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"path-only fixture")
            with patch("swim4track.policy.__file__", str(target / "swim4track/policy.py")), \
                 patch("swim4track.policy.sysconfig.get_path", return_value=str(target / "unrelated")):
                self.assertEqual(default_model_path(), checkpoint)

    def test_verification_cli_records_only_checks_actually_run(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([]), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["hash_verified"])
        self.assertFalse(result["inference_checked"])
        self.assertEqual(result["bytes"], 2879572)
        self.assertEqual(result["model_sha256"], MODEL_SHA256)

    def test_bad_observation_rejected(self):
        for bad in (np.zeros(24), np.full(23, np.nan), np.full(23, 2.0)):
            with self.assertRaises(ValueError):
                validate_observation(bad)

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "optional inference dependencies not installed")
    def test_released_actor_predicts_finite_bounded_action(self):
        policy = FrozenPolicy()
        action = policy.predict(np.zeros(23, dtype=np.float32))
        self.assertEqual(action.shape, (8,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(np.abs(action) <= 1.0))
        np.testing.assert_array_equal(action, policy.predict(np.zeros(23, dtype=np.float32)))

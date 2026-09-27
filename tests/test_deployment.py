"""Deployment bounds apply to both supported actuator mappings."""
import unittest
import numpy as np

from swim4track.deployment import map_policy_action_to_stonefish


class DeploymentLimitTest(unittest.TestCase):
    def test_custom_limit_applies_to_every_mapping_and_sign(self):
        action = np.array([-1., -.8, -.5, -.1, .1, .5, .8, 1.])
        signs = np.array([-1., 1., -1., 1., -1., 1., -1., 1.])
        for mapping in ("raw_command", "force_equivalent"):
            with self.subTest(mapping=mapping):
                command = map_policy_action_to_stonefish(
                    action, command_signs=signs, stonefish_command_limit=.2,
                    actuator_mapping=mapping,
                )
                self.assertLessEqual(float(np.max(np.abs(command))), .2)
                np.testing.assert_array_equal(np.sign(command), signs*np.sign(action))

    def test_default_raw_mapping_remains_training_command(self):
        action = np.linspace(-1, 1, 8)
        command = map_policy_action_to_stonefish(
            action, command_signs=np.ones(8), actuator_mapping="raw_command")
        np.testing.assert_array_equal(command, .6*action)

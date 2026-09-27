"""Recipe and logging contracts that do not require the optional ML runtime."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from swim4track.train import main, training_configuration
from swim4track.training_curves import read_training_scalars


class TrainingContractTests(unittest.TestCase):
    def test_recipe_matches_bundled_checkpoint_metadata(self):
        import zipfile
        from swim4track.policy import default_model_path
        with zipfile.ZipFile(default_model_path()) as archive:
            metadata = json.loads(archive.read("data"))
        recipe = training_configuration()
        self.assertEqual(recipe["seed"], metadata["seed"])
        self.assertEqual(recipe["total_timesteps"], metadata["num_timesteps"])
        for key in ("learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma", "tau"):
            self.assertEqual(recipe["optimizer"][key], metadata[key])
        kwargs = metadata["policy_kwargs"]
        self.assertEqual(recipe["actor_architecture"], kwargs["net_arch"]["pi"])
        self.assertEqual(recipe["critic_architecture"], kwargs["net_arch"]["qf"])
        self.assertEqual(recipe["n_critics"], kwargs["n_critics"])
        self.assertEqual(recipe["n_quantiles"], kwargs["n_quantiles"])

    def test_invalid_budget_and_worker_seeds_rejected(self):
        for parameters in ({"steps": 0}, {"envs": 0}, {"seed": -1},
                           {"steps": 101, "envs": 4}, {"seed": 2**32-1},
                           {"steps": 3.5}, {"seed": True}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                training_configuration(**parameters)

    def test_print_recipe_without_optional_imports(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main(["--print-config"])
        self.assertEqual(json.loads(stream.getvalue())["observation_size"], 23)

    def test_loss_samples_are_not_filled_or_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.csv"
            path.write_text("time/total_timesteps,rollout/ep_rew_mean,train/critic_loss\n"
                            "1600,-100,\n11200,-90,0.5\n")
            values = read_training_scalars(path)
            self.assertTrue(np.isnan(values["train/critic_loss"][0]))
            self.assertTrue(np.all(np.isnan(values["train/actor_loss"])))
            np.testing.assert_array_equal(values["time/total_timesteps"], [1600, 11200])
            path.write_text("time/total_timesteps,train/actor_loss\n0,\n")
            with self.assertRaisesRegex(ValueError, "no recorded"):
                read_training_scalars(path)


if __name__ == "__main__":
    unittest.main()

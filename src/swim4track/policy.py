"""Hash-verified, deterministic inference with the released TQC checkpoint."""
from __future__ import annotations

from pathlib import Path
from importlib.metadata import PackageNotFoundError, distribution
import site
import sysconfig
import numpy as np
from .deployment import file_sha256

MODEL_NAME = "tqc_nominal_seed17_final.zip"
MODEL_SHA256 = "304f281a7843d0c0b2e2016ad470a8c923524a4e236f536e07764a51d7967f2f"


def default_model_path():
    module = Path(__file__).resolve()
    candidates = [
        module.parents[2] / "models" / MODEL_NAME,
        Path(sysconfig.get_path("data")) / "share" / "swim4track" / "models" / MODEL_NAME,
        # pip --target places wheel data alongside the package itself.
        module.parents[1] / "share" / "swim4track" / "models" / MODEL_NAME,
    ]
    try:
        installed = distribution("swim4track")
    except PackageNotFoundError:
        installed = None
    if installed is not None:
        for entry in installed.files or ():
            if entry.name == MODEL_NAME and entry.parent.name == "models":
                candidates.append(Path(installed.locate_file(entry)))
    if site.USER_BASE:
        candidates.append(Path(site.USER_BASE) / "share" / "swim4track" / "models" / MODEL_NAME)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Bundled checkpoint was not found; provide --model explicitly.")


def verify_checkpoint(model_path=None, expected_sha256=None):
    """Verify artifact identity without importing ML libraries or unpickling it."""
    path = Path(model_path).expanduser().resolve() if model_path else default_model_path()
    expected = MODEL_SHA256 if expected_sha256 is None else str(expected_sha256).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    digest = file_sha256(path)
    if digest != expected:
        raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected}, got {digest}")
    return {"model_path": str(path.resolve()), "model_sha256": digest, "bytes": path.stat().st_size}


def validate_observation(observation):
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (23,) or not np.all(np.isfinite(obs)):
        raise ValueError("observation must contain 23 finite values")
    if np.any(np.abs(obs) > 1.0 + 1e-6):
        raise ValueError("observation must already follow the normalized [-1, 1] contract")
    return obs


class FrozenPolicy:
    """Only deserialize after verifying the trusted checkpoint's SHA-256.

    To evaluate your own model, pass its independently recorded SHA-256.
    This is not a trust mechanism for arbitrary downloaded model files.
    """
    def __init__(self, model_path=None, expected_sha256=None, device="cpu"):
        verified = verify_checkpoint(model_path, expected_sha256)
        try:
            from sb3_contrib import TQC
        except ImportError as exc:
            raise ImportError("Install inference dependencies: python -m pip install '.[inference]'") from exc
        self.model = TQC.load(verified["model_path"], device=device)
        if self.model.observation_space.shape != (23,) or self.model.action_space.shape != (8,):
            raise ValueError("checkpoint must have observation shape (23,) and action shape (8,)")
        for name, space in (("observation", self.model.observation_space), ("action", self.model.action_space)):
            if not np.allclose(space.low, -1.0) or not np.allclose(space.high, 1.0):
                raise ValueError(f"checkpoint {name} bounds must be [-1, 1]")
        self.model.policy.set_training_mode(False)
        self.metadata = {**verified,
                         "algorithm": "TQC", "deterministic": True, "device": str(self.model.device)}

    def reset(self):
        """The released actor is feedforward and has no hidden recurrent state."""

    def predict(self, observation):
        obs = validate_observation(observation)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise RuntimeError("policy returned an invalid eight-thruster action")
        if np.any(np.abs(action) > 1.0 + 1e-6):
            raise RuntimeError("policy action exceeded normalized command bounds")
        return np.clip(action, -1.0, 1.0)


def main(argv=None):
    """Inspect a trusted checkpoint; loading the actor is an explicit check."""
    import argparse
    import json
    from . import __version__

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-sha256", help="trusted digest for your own checkpoint")
    parser.add_argument("--load", action="store_true", help="also load the actor and test deterministic bounded inference")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = verify_checkpoint(args.model, args.model_sha256)
    result.update(schema="swim4track_model_check_v1", swim4track_version=__version__,
                  hash_verified=True, inference_checked=False)
    if args.load:
        policy = FrozenPolicy(args.model, args.model_sha256, args.device)
        observation = np.zeros(23, dtype=np.float32)
        first, second = policy.predict(observation), policy.predict(observation)
        if not np.array_equal(first, second):
            raise RuntimeError("repeated deterministic predictions differed")
        result.update(inference_checked=True, action_shape=list(first.shape), device=str(policy.model.device))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

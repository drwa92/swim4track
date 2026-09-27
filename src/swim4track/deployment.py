"""Dependency-light frozen action calibration and artifact hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from .plant import t200_force

ACTUATOR_MAPPINGS = {"raw_command", "force_equivalent"}
STONEFISH_FORCE_COEFFICIENT_N = 16.853640533657778


def map_policy_action_to_stonefish(
    action,
    *,
    command_signs,
    training_command_limit: float = 0.60,
    stonefish_command_limit: float = 1.0,
    actuator_mapping: str = "force_equivalent",
):
    """Map eight learned actions to the audited Stonefish command interface.

    ``raw_command`` preserves the legacy numerical-command mapping.
    ``force_equivalent`` first evaluates the T200 force used during training,
    then applies the analytic inverse of Stonefish's static quadratic thrust
    law.  This is a channel-wise actuator calibration, not thrust allocation.
    """
    action = np.asarray(action, dtype=np.float64)
    signs = np.asarray(command_signs, dtype=np.float64)
    if action.shape != (8,) or not np.all(np.isfinite(action)):
        raise ValueError("action must contain eight finite values")
    if signs.shape != (8,) or not np.all(np.isin(signs, [-1.0, 1.0])):
        raise ValueError("command_signs must contain eight values, each +1 or -1")
    if actuator_mapping not in ACTUATOR_MAPPINGS:
        raise ValueError(
            f"actuator_mapping must be one of {sorted(ACTUATOR_MAPPINGS)}"
        )
    if not 0.0 < training_command_limit <= 1.0:
        raise ValueError("training_command_limit must lie in (0, 1]")
    if not 0.0 < stonefish_command_limit <= 1.0:
        raise ValueError("stonefish_command_limit must lie in (0, 1]")

    action = np.clip(action, -1.0, 1.0)
    if actuator_mapping == "raw_command":
        # Honor the requested deployment limit in both calibration modes.
        return signs * np.clip(training_command_limit * action,
                               -stonefish_command_limit, stonefish_command_limit)

    desired_force = t200_force(training_command_limit * action)
    magnitude = np.sqrt(
        np.abs(desired_force) / STONEFISH_FORCE_COEFFICIENT_N
    )
    equivalent_command = np.sign(desired_force) * np.minimum(
        magnitude, stonefish_command_limit
    )
    return signs * equivalent_command


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one deployment artifact."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

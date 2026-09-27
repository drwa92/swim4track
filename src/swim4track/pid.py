"""Portable, hash-locked construction of the released PID baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .classical_control import DampedThrusterAllocator, PIDGains, PIDWrenchController
from .deployment import file_sha256

FROZEN_PID_SHA256 = "b76c7c93054e864aeea3bd4945ec3ea814c6ebe55d94dfc86ed2db6bb29a0ec6"
PROTOCOL_SHA256 = "afcae8710bc15aded1197f2971c6cba9df3bfa11f9259d193536779a52ffd189"


def protocol_hash(path):
    """Hash canonical JSON, exactly as in the original frozen protocol."""
    data = json.loads(Path(path).read_text())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_frozen_pid(config_dir=None):
    """Return the released PID and provenance after verifying frozen records.

    ``config_dir`` may point to a copied set of the release configuration files;
    it does not enable silently substituting retuned gains. Historical NMPC
    entries remain in the protocol solely to preserve its original hash.
    """
    directory = Path(config_dir) if config_dir is not None else Path(__file__).parent / "config"
    frozen_path = directory / "pid_frozen.json"
    protocol_path = directory / "classical_benchmarks_v1.json"
    if not frozen_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("frozen controller and protocol files are required")
    frozen_digest = file_sha256(frozen_path)
    if frozen_digest != FROZEN_PID_SHA256:
        raise RuntimeError(f"frozen-controller SHA-256 mismatch: {frozen_digest}")
    frozen = json.loads(frozen_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    canonical_digest = protocol_hash(protocol_path)
    if canonical_digest != PROTOCOL_SHA256:
        raise RuntimeError(f"classical-protocol canonical SHA-256 mismatch: {canonical_digest}")
    if frozen.get("controller") != "pid":
        raise RuntimeError("frozen controller kind does not match PID")
    if frozen.get("classical_protocol_id") != protocol.get("classical_protocol_id"):
        raise RuntimeError("frozen controller references a different protocol")
    if frozen.get("protocol_sha256") != canonical_digest:
        raise RuntimeError("frozen controller protocol digest is not current")
    if frozen.get("status") != "frozen_before_final_evaluation":
        raise RuntimeError("controller was not frozen before final evaluation")
    candidate = frozen.get("selected_candidate")
    index = int(frozen.get("selected_candidate_index", -1))
    candidates = protocol.get("pid_candidates", [])
    if index < 0 or index >= len(candidates) or candidates[index] != candidate:
        raise RuntimeError("selected candidate is inconsistent with the protocol")
    allocation = protocol["common_allocator"]
    allocator = DampedThrusterAllocator(
        damping=float(allocation["damping"]),
        force_limit_n=float(allocation["force_limit_n"]),
    )
    base = PIDGains()
    gains = PIDGains(
        kp_pose=base.kp_pose * float(candidate["kp_scale"]),
        kd_twist=base.kd_twist * float(candidate["kd_scale"]),
        ki_pose=base.ki_pose * float(candidate["ki_scale"]),
        integral_limit=base.integral_limit,
        antiwindup_gain=float(candidate["antiwindup_gain"]),
        feedforward=bool(candidate["feedforward"]),
    )
    return PIDWrenchController(gains=gains, allocator=allocator), {
        "frozen_sha256": frozen_digest,
        "protocol_sha256": canonical_digest,
        "candidate_name": candidate["name"],
    }

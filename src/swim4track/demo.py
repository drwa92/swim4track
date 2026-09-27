"""Run a recorded PID or frozen-policy rollout in the supplied Python plant."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np

from . import __version__
from .tracking import Swim4TrackCore
from .core import attitude_error_vector
from .trajectory_reference import evaluation_trajectories


def run_demo(controller_kind="pid", trajectory_name="circle_level", seconds=40.0,
             seed=20000, model_path=None, model_sha256=None):
    if not np.isfinite(seconds) or seconds < 0.1 or not np.isclose(seconds / 0.1, round(seconds / 0.1)):
        raise ValueError("seconds must be positive and an integer multiple of 0.1")
    paths = {path.name: path for path in evaluation_trajectories()}
    if trajectory_name not in paths:
        raise ValueError(f"unknown trajectory: {trajectory_name}")
    if controller_kind == "pid":
        if model_path is not None or model_sha256 is not None:
            raise ValueError("model options require controller_kind='learned' (--controller learned)")
        from .pid import load_frozen_pid
        controller, identity = load_frozen_pid()
    elif controller_kind == "learned":
        from .policy import FrozenPolicy
        controller = FrozenPolicy(model_path, model_sha256)
        identity = controller.metadata
    else:
        raise ValueError("controller must be pid or learned")
    steps = int(round(seconds / 0.1))
    core = Swim4TrackCore(max_steps=steps)
    obs = core.reset(seed=seed, trajectory=paths[trajectory_name])
    controller.reset()
    states = [core.state.copy()]
    references = [np.r_[core.reference.position_ned, core.reference.rpy]]
    actions, commands, rewards, compute_ms = [], [], [], []
    terminated = False
    for _ in range(steps):
        start = perf_counter()
        if controller_kind == "pid":
            action = controller.act(core.state, core.reference, core.trajectory, core.time)
        else:
            action = controller.predict(obs)
        compute_ms.append((perf_counter() - start) * 1000)
        obs, reward, terminated, truncated, info = core.step(action)
        states.append(core.state.copy())
        references.append(np.r_[core.reference.position_ned, core.reference.rpy])
        actions.append(np.asarray(action).copy())
        commands.append(info["command"])
        rewards.append(reward)
        if terminated or truncated:
            break
    data = {"time_s": np.arange(len(states)) * core.control_dt,
            "state": np.asarray(states), "reference_pose": np.asarray(references),
            "raw_action": np.asarray(actions), "plant_command": np.asarray(commands),
            "reward": np.asarray(rewards), "compute_ms": np.asarray(compute_ms)}
    position_error = np.linalg.norm(data["state"][:, :3] - data["reference_pose"][:, :3], axis=1)
    attitude_error_deg = np.rad2deg([
        np.linalg.norm(attitude_error_vector(s[3:6], r[3:6]))
        if np.all(np.isfinite(s[3:6])) else np.nan
        for s, r in zip(data["state"], data["reference_pose"])])
    data["position_error_m"] = position_error
    data["attitude_error_deg"] = attitude_error_deg
    # Exclude the initial sample from these per-step demonstration summaries.
    p, a = position_error[1:], attitude_error_deg[1:]
    finite_trace = bool(np.all(np.isfinite(data["state"])))
    summary = {
        "schema": "swim4track_python_demo_v1", "controller": controller_kind,
        "trajectory": trajectory_name, "seed": seed, "simulator": "supplied Python plant",
        "requested_duration_s": seconds, "observed_duration_s": float(core.time),
        "control_dt_s": core.control_dt, "physics_dt_s": core.physics_dt,
        "completed": not terminated and len(actions) == steps,
        "terminated_unsafe": bool(terminated), "recorded_steps": len(actions),
        "finite_state_trace": finite_trace,
        "position_rmse_m": float(np.sqrt(np.mean(p * p))) if finite_trace else None,
        "position_max_m": float(np.max(p)) if finite_trace else None,
        "attitude_rmse_deg": float(np.sqrt(np.mean(a * a))) if finite_trace else None,
        "maximum_abs_plant_command": float(np.max(np.abs(data["plant_command"]))),
        "local_controller_compute_p99_ms": float(np.percentile(compute_ms, 99)),
        "measurement_note": "Illustrative single rollout, full duration including acquisition; not a paper score or hard real-time benchmark.",
        "controller_identity": identity,
        "software": {"swim4track": __version__, "python": platform.python_version(), "numpy": np.__version__},
    }
    return data, summary


def save_rollout(output, data, summary):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "rollout.npz", **data)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    state_columns = ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad",
                     "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps"]
    # State and command tables are separate to make the zero-order-hold interval explicit.
    with (output / "states.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + state_columns + ["ref_" + c for c in state_columns[:6]] +
                        ["position_error_m", "attitude_error_deg"])
        writer.writerows(np.c_[data["time_s"], data["state"], data["reference_pose"],
                                data["position_error_m"], data["attitude_error_deg"]])
    with (output / "commands.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["interval_start_s", "interval_end_s"] + [f"raw_action_{i+1}" for i in range(8)] +
                        [f"plant_command_{i+1}" for i in range(8)] + ["reward", "compute_ms"])
        writer.writerows(np.c_[data["time_s"][:-1], data["time_s"][1:], data["raw_action"],
                               data["plant_command"], data["reward"], data["compute_ms"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("pid", "learned"), default="pid")
    parser.add_argument("--trajectory", choices=[p.name for p in evaluation_trajectories()], default="circle_level")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--playback", type=float, default=2.0, help="video playback speed relative to simulation")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; select a fresh directory")
    if not np.isfinite(args.playback) or args.playback <= 0:
        parser.error("playback must be positive and finite")
    data, summary = run_demo(args.controller, args.trajectory, args.seconds, args.seed, args.model, args.model_sha256)
    save_rollout(args.output, data, summary)
    if (args.plot or args.video) and summary["finite_state_trace"]:
        from .visualize import render_demo
        render_demo(args.output, video=args.video, playback=args.playback)
    elif args.plot or args.video:
        print("Nonfinite trajectory: raw failed-run evidence saved; visualization skipped.")
    print(json.dumps(summary, indent=2))
    print(f"Saved rollout: {args.output.resolve()}")
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

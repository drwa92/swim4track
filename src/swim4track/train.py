"""Train the nominal Swim4Track TQC recipe from scratch.

The default architecture, optimizer settings, environment and budget match the
released checkpoint recipe. A new run is not claimed to reproduce its weights
bit for bit across different devices, dependency versions or restart schedules.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import monotonic


def training_configuration(seed=17, steps=300_000, envs=4):
    """Dependency-light configuration record for the nominal recipe."""
    for name, value in (("seed", seed), ("steps", steps), ("envs", envs)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if steps <= 0 or envs <= 0:
        raise ValueError("steps and envs must be positive")
    if steps % envs:
        raise ValueError("steps must be divisible by envs for an exact transition budget")
    if seed < 0 or seed + envs > 2**32:
        raise ValueError("seed and all worker seeds must lie in [0, 2**32 - 1]")
    return {
        "method": "tqc_nominal",
        "representation": "swim4track",
        "training_initialization": "from_scratch",
        "domain_randomization": False,
        "ocean_disturbances": False,
        "seed": int(seed),
        "total_timesteps": int(steps),
        "parallel_environments": int(envs),
        "observation_size": 23,
        "action_size": 8,
        "control_dt_s": 0.1,
        "physics_dt_s": 0.02,
        "episode_steps": 400,
        "command_limit": 0.60,
        "actor_architecture": [128, 128],
        "critic_architecture": [256, 256],
        "n_critics": 2,
        "n_quantiles": 25,
        "top_quantiles_to_drop_per_net": 2,
        "optimizer": {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 1,
            "ent_coef": "auto",
            "use_sde": False,
        },
    }


def _env_factory(rank, seed, monitor_dir):
    def make():
        from .gym_env import Swim4TrackEnv
        from stable_baselines3.common.monitor import Monitor
        env = Swim4TrackEnv(representation="swim4track")
        env.reset(seed=seed + rank)
        return Monitor(env, filename=str(Path(monitor_dir) / f"worker_{rank:02d}"),
                       info_keywords=("trajectory", "unsafe"))
    return make


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true",
                        help="print the recipe and exit without ML dependencies or a training run")
    args = parser.parse_args(argv)
    try:
        config = training_configuration(args.seed, args.steps, args.envs)
    except ValueError as exc:
        parser.error(str(exc))
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    if args.output is None:
        parser.error("--output is required for training")
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output directory must be empty; use a fresh directory for a new training run")
    try:
        import gymnasium
        import numpy as np
        import stable_baselines3
        import sb3_contrib
        import torch
        from sb3_contrib import TQC
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        parser.error(f"Install training dependencies with pip install 'swim4track[train]': {exc}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    config["device"] = device
    config["checkpoint_interval_transitions"] = max(100_000 // args.envs, 1) * args.envs
    config["logging"] = {
        "episode_csv": "monitor/worker_*.monitor.csv",
        "training_csv": "logs/progress.csv",
        "tensorboard": "logs/",
        "log_interval_episodes": 4,
    }
    config["software"] = {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "monitor").mkdir()
    progress_path = output / "progress.json"
    progress_path.write_text(json.dumps({"complete": False, "status": "initializing",
                                         "target_timesteps": args.steps}, indent=2) + "\n")
    env = None
    logger = None
    model = None
    interrupted = False
    started = monotonic()
    try:
        env = SubprocVecEnv([_env_factory(rank, args.seed, output / "monitor")
                            for rank in range(args.envs)])
        logger = configure(str(output / "logs"), ["stdout", "csv", "tensorboard"])
        model = TQC(
            "MlpPolicy", env, **config["optimizer"],
            policy_kwargs={
                "net_arch": {"pi": config["actor_architecture"], "qf": config["critic_architecture"]},
                "n_critics": 2, "n_quantiles": 25,
            },
            top_quantiles_to_drop_per_net=2,
            seed=args.seed, verbose=1, device=device,
        )
        model.set_logger(logger)
        checkpoint = CheckpointCallback(
            save_freq=max(100_000 // args.envs, 1),
            save_path=str(output / "checkpoints"),
            name_prefix=f"tqc_nominal_seed{args.seed}",
        )
        try:
            model.learn(total_timesteps=args.steps, callback=checkpoint, progress_bar=False,
                        log_interval=config["logging"]["log_interval_episodes"])
        except KeyboardInterrupt:
            interrupted = True
        suffix = "interrupted" if interrupted else "final"
        model_path = output / f"tqc_nominal_seed{args.seed}_{suffix}.zip"
        model.save(model_path)
        from .deployment import file_sha256
        progress = {
            "complete": not interrupted and model.num_timesteps >= args.steps,
            "status": "interrupted" if interrupted else "completed",
            "completed_timesteps": int(model.num_timesteps),
            "target_timesteps": args.steps,
            "checkpoint": model_path.name,
            "sha256": file_sha256(model_path),
            "elapsed_wall_time_s": monotonic() - started,
            "note": "This trainer starts new runs; interrupted weights are not a replay-buffer resume archive.",
        }
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(progress, indent=2))
    except Exception as exc:
        progress_path.write_text(json.dumps({
            "complete": False, "status": "failed", "error": str(exc),
            "completed_timesteps": int(model.num_timesteps) if model is not None else 0,
            "target_timesteps": args.steps,
            "elapsed_wall_time_s": monotonic() - started,
        }, indent=2) + "\n")
        raise
    finally:
        if env is not None:
            env.close()
        if logger is not None:
            logger.close()
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()

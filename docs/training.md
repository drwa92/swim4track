# Train the nominal Swim4Track policy

The trainer starts a new TQC policy in the supplied Python plant. Its defaults
match the released checkpoint's nominal recipe: seed 17, four environments,
300,000 transitions, a 128–128 actor and 256–256 critics. The plant, observation
and reward remain unchanged. New runs are not expected to reproduce the
checkpoint weights bit for bit across devices or dependency versions.

## Inspect and run

From the repository root, inspect the recipe without installing PyTorch:

```bash
python -m swim4track.train --print-config
```

Install the training dependencies in your Python environment and use a fresh
output directory:

```bash
python -m pip install '.[train,viz]'
python -m swim4track.train --steps 300000 --envs 4 --seed 17 \
  --device cpu --output runs/train_seed17
```

Use `--device cuda` only with an available CUDA runtime; the command rejects an
unavailable GPU. The default `auto` selects CUDA when available, otherwise CPU.
The transition budget must be divisible by the environment count. It counts
transitions across all workers, not transitions per worker.

A short startup check can use `--steps 1600 --envs 4` and a separate output
directory. It tests environment stepping and checkpoint writing but performs
**no gradient updates**, because learning begins after 10,000 transitions.
It is not a trained controller. Full training is a separate, much longer run.

## Outputs and learning curves

| File or directory | Contents |
|---|---|
| `run_config.json` | Hyperparameters, environment contract, device and actual software versions |
| `monitor/worker_*.monitor.csv` | Per-episode return, length, wall time, trajectory and unsafe-termination flag |
| `logs/progress.csv` | Logged training-return averages and optimizer scalars |
| `logs/events.out.tfevents.*` | The same available scalars for TensorBoard |
| `checkpoints/` | Periodic policy checkpoints, approximately every 100,000 transitions |
| `tqc_nominal_seed17_final.zip` | Final policy when training finishes normally |
| `progress.json` | Completion/failure status, transition count, wall time, final model digest when saved |

Each worker is wrapped in Stable-Baselines3 `Monitor`; logging does not change
the physics or reward. CSV and TensorBoard use the documented
[SB3 logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html).
The normal logging interval is four completed episodes. Very short runs may
not emit all metrics, and optimizer losses are absent before learning starts.

```bash
tensorboard --logdir runs/train_seed17/logs
python -m swim4track.training_curves --run runs/train_seed17 \
  --output runs/train_seed17/figures
```

The figure command produces PNG/PDF diagnostics of recorded training return,
actor loss and critic loss; edit `training_curves.py` to change their style.
Missing series are marked as
unrecorded; the command does not reconstruct historical losses from weights.
The return curve uses SB3's mean over its recent-episode window (100 by
default), while optimizer losses are the values SB3 logged. No additional
smoothing is applied. Loss values are training diagnostics, not independent
evidence of tracking accuracy or robustness.

## Evaluate a new checkpoint

The released checkpoint remains untouched under `models/`. Obtain the SHA-256
of a new model and pass that digest explicitly:

```bash
sha256sum runs/train_seed17/tqc_nominal_seed17_final.zip
```

Then replace `YOUR_SHA256` with the printed digest:

```bash
python -m swim4track.demo --controller learned --trajectory helix_unseen \
  --model runs/train_seed17/tqc_nominal_seed17_final.zip \
  --model-sha256 YOUR_SHA256 --seconds 40 \
  --output runs/new_policy_helix --plot
```

Use independent evaluation seeds and report multiple training seeds for new
scientific comparisons. Keep held-out trajectories out of training and model
selection when making generalization claims. A single demo or decreasing
critic loss is not sufficient for those claims.

## Interruptions and scope

Ctrl+C during `learn()` saves an `_interrupted.zip` checkpoint and exits with
code 130; `progress.json` marks the run incomplete. A failed run retains logs
and reports the exception. The trainer refuses nonempty output directories.
It starts from scratch and does not provide exact replay-buffer resume: neither
periodic checkpoints nor interrupted weights contain a complete continuation
state for the training process.

The trainer is intentionally limited to the released nominal formulation.
It does not silently add domain randomization, actuator faults, different
rewards or a new controller architecture. See
[the model card](model_card.md) for training provenance and
[validation](VALIDATION.md) for what has actually been exercised in this release.

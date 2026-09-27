# Model card: S4T-TQC, nominal seed 17

## Identity and use

This is the frozen nominal **Swim4Track** trajectory-tracking checkpoint,
also named **S4T-TQC** to distinguish the framework from its variants. It
produces eight direct-thruster actions from the current vehicle state and
time-parameterized reference. It is supplied for simulation research and
reproduction of the controller interface.

| Field | Value |
|---|---|
| File | `models/tqc_nominal_seed17_final.zip` |
| Algorithm / loader | Truncated Quantile Critics / `sb3_contrib.TQC` |
| Training seed / transitions | 17 / 300,000 |
| Inference | Deterministic actor prediction |
| Observation / action | 23 / 8 values |
| Actor hidden layers | 128, 128 |
| Critic hidden layers | 256, 256 |
| Critics / quantiles per critic | 2 / 25 |
| Dropped upper quantiles per critic | 2 |
| Learning rate / discount / target update | 0.0003 / 0.99 / 0.005 |
| Replay capacity / batch / learning starts | 1,000,000 / 256 / 10,000 |
| Parallel environments / episode horizon | 4 / 400 control steps |
| Control / plant integration interval | 0.10 s / 0.02 s |

These values were read from the checkpoint's JSON members, rather than
inferred from old configuration defaults. The model digest in
[`model_manifest.json`](../models/model_manifest.json) matches the recorded
ablation and Stonefish robustness-v2 manifests. The release includes the
original file unchanged.

## Verify and run

From an installed checkout, verify the checkpoint without loading serialized
objects or importing PyTorch:

```bash
python -m swim4track.policy
```

After installing the inference dependencies, run the loader and deterministic
action check, then a recorded Python rollout:

```bash
python -m swim4track.policy --load
python -m swim4track.demo --controller learned --trajectory circle_level --seconds 40 --output runs/learned_circle
```

The first command's JSON distinguishes `hash_verified` from
`inference_checked`. Hash verification alone does not establish that the
checkpoint can run in a particular software environment. The short actor
check verifies loading, dimensions, bounds and repeated deterministic
prediction; the rollout tests the controller in the supplied plant.
Neither is a rerun of the paper's full evaluation.

## Input contract

The policy uses NED world coordinates and the corresponding FRD vehicle
frame. Let `R` map vehicle vectors to NED, and `R_d` be the reference
orientation. The implementation concatenates the following blocks, clips
all elements to `[-1,1]`, and returns `float32`:

| Python slice | Meaning | Normalization |
|---|---|---|
| `0:3` | `R.T @ (position - reference_position)` | Divide by `[6,6,3]` m |
| `3:6` | Rotation vector of `R.T @ R_d` | Divide by pi |
| `6:9` | `R.T @ [0,0,1]` | None; unit world-down direction |
| `9:12` | `R.T @ (R @ body_linear_velocity - reference_linear_velocity_NED)` | Divide by `[1.5,1.5,1]` m/s |
| `12:15` | `R.T @ (R @ body_angular_velocity - reference_angular_velocity_NED)` | Divide by `[1,1,1]` rad/s |
| `15:23` | Previous raw actor action | Eight normalized values |

The three orientation-context values are the world-down direction expressed
in vehicle axes, not a raw accelerometer reading. There are no camera,
sonar, DVL-message, future-reference-preview or fault-health inputs. The
ROS interface constructs state from simulated odometry. A different state
estimator requires an explicit frame and convention audit.

## Output and plant interfaces

Each actor output lies in `[-1,1]`. In the Python training plant, the command
is `0.60 * action`, passed through the source T200 static force relation
and actuator dynamics. The Stonefish deployment uses a separate fixed,
channel-wise force-equivalent mapping and command filtering. Actor action,
deployment command and fault-modified applied command must not be treated
as interchangeable signals. These normalized values are not microsecond
hardware PWM values.

The previous-action observation stores the previous **raw actor action**.
It must not be replaced with filtered or degraded deployment commands.
Thruster signs and ordering are properties of the audited simulator
configuration; copying them to another vehicle requires checking that
configuration.

## Training provenance

The nominal `tqc_nominal` branch of the original
`train_robust_policy_benchmark.py` used `Swim4TrackEnv`. The source
`tracking.py` constructs the observation and tracking reward; `plant.py`
provides the Python dynamics and actuator model. Deployment calibration
was defined in `deployment_contract.py` and is provided here in
[`deployment.py`](../src/swim4track/deployment.py). Historical YAML files for the
older pose baseline specify a different architecture and budget and are
not authoritative metadata for this checkpoint.

The cleaned [`train.py`](../src/swim4track/train.py) exposes the nominal
training recipe and writes new artifacts to the requested output directory.
It does not overwrite this checkpoint or claim bit-identical reproduction
across dependency versions and hardware. Only this nominal seed-17 model
is bundled; other comparison policies and the full research archives remain
separate.

The archive records Python 3.12.3, Stable-Baselines3 2.9.0, PyTorch
2.12.1+cu130, NumPy 2.5.2, Cloudpickle 3.1.2 and Gymnasium 1.3.0, with GPU
enabled during training. Its SB3-Contrib version is not recorded in
`system_info.txt`. The recorded Stonefish campaign used ROS Humble and
Python 3.10.12; its evidence does not include a complete pip lock.
Neither record is a claim that all such version combinations have been
tested for this cleaned repository.

## Evaluation scope and limits

The checkpoint was frozen for cross-simulator evaluation in ROS 2/Stonefish.
The paper's primary Stonefish evidence comprises a 60-trial nominal campaign
and a 270-trial actuator-effectiveness campaign. These totals include the
comparison controllers; they are not 330 trials of this checkpoint alone.
The actuator campaign contains a unit-health condition and prescribed loss
schedules, all with zero configured water current. The repetitions assess
repeatability of that fixed campaign.

This evidence does not establish hardware transfer, current-disturbance
robustness, active fault diagnosis, fault-aware reconfiguration or a formal
stability guarantee. The checkpoint does not observe actuator health.
The demonstration runs in this repository are usage examples, separate
from the frozen paper measurements. New timings depend on the new machine
and runtime.

Model redistribution terms still require owner confirmation; see
[`models/README.md`](../models/README.md).

# Python plant and controller interface

The numerical simulation needs only NumPy and SciPy. ROS, PyTorch and a
display server are not imported by the plant. The released physics, reference,
observation and reward implementations retain the scientific source contract;
the command-line tools provide a smaller public interface around them.

## Run the trained policy

After installing the inference and plotting extras from the repository root:

```bash
python -m pip install '.[inference,viz]'
python -m swim4track.demo --controller learned --trajectory circle_level \
  --seconds 40 --output runs/learned_circle --plot
```

Every rollout writes state/reference samples, command intervals, a compressed
NumPy archive and a JSON summary. `--video` additionally requires FFmpeg.
Output directories must be new. Demonstration summaries include the full
rollout; they are not replacements for the paper's scored evaluation windows.

For an installation check without neural-network dependencies, use
`--controller pid` with a fresh output directory. The frozen PID is a comparison
controller; selecting it does not execute the learned policy.

## Use the numerical API

```python
from swim4track.policy import FrozenPolicy
from swim4track.tracking import Swim4TrackCore
from swim4track.trajectory_reference import evaluation_trajectories

paths = {path.name: path for path in evaluation_trajectories()}
plant = Swim4TrackCore(max_steps=400)
observation = plant.reset(seed=20000, trajectory=paths["circle_level"])
policy = FrozenPolicy(device="cpu")

for _ in range(400):
    action = policy.predict(observation)
    observation, reward, terminated, truncated, info = plant.step(action)
    state = plant.state.copy()
    if terminated or truncated:
        break
```

The model loader verifies the released SHA-256 before loading the checkpoint.
For your own checkpoint, explicitly pass its path and expected digest to
`FrozenPolicy`; do not relabel it as the released model.

| Interface | Contract |
|---|---|
| State | 12 values: NED position, roll/pitch/yaw, FRD body linear/angular velocities |
| State units | m, rad, m/s, rad/s |
| Policy observation | 23 values, `float32`, clipped to `[-1, 1]` |
| Raw policy action | 8 values in `[-1, 1]`, retained in the next observation |
| Python plant command | `0.60 * action`; input to the T200 actuator model |
| Control / integration periods | 0.10 s / 0.02 s |
| Default episode | 400 control steps; 40 simulated seconds |
| Termination | Nonfinite state or position-reference distance greater than 25 m |
| Truncation | Requested episode length reached |

`reset(seed=...)` selects a training trajectory and initial-state perturbation
reproducibly. Passing a trajectory explicitly fixes its geometry; the seed still
controls the initial perturbation. Passing a 12-value `state` also fixes the
initial vehicle state.

Available named demonstration references are `hold_level`, `hold_heading`,
`quintic_coupled`, `circle_level`, `circle_attitude`, `helix_unseen`,
`figure8_unseen` and `coupled_fast`. See `trajectory_reference.py` for their
parameters and analytic derivatives.

## Numerical model boundaries

`plant.py` provides `VehicleParameters`, `ValidatedBlueROV2Heavy` and
`T200ActuatorBank`. The model integrates six-degree-of-freedom dynamics using
RK4 and a dynamic thruster model. `tracking.py` adds trajectory sampling,
the 23-value observation and the tracking reward. `gym_env.py` is the optional
Gymnasium adapter used by the trainer.

The historical class name `ValidatedBlueROV2Heavy` identifies the audited
research implementation; it is not a hardware-certification claim. The
nominal training recipe uses neither domain randomization nor exogenous
disturbances. Numerical support for a disturbance input does not establish
validated Stonefish current robustness.

The Stonefish interface uses a separate force mapping, channel signs and
command filter. Do not send `0.60 * action` to an arbitrary ROS thruster topic:
use the supplied ROS node with its documented simulator conventions.

See [the model card](model_card.md) for every observation block and
[training](training.md) for retraining and diagnostic logs.

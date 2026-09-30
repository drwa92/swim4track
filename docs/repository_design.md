# Implementation map

The repository separates numerical control from deployment and command-line tools.
Importing `swim4track` does not import ROS, PyTorch or a trained model.

| Component | Main files | Responsibility |
|---|---|---|
| Physical plant | `plant.py`, `actuator_dynamics.py` | 6-DoF hydrodynamics, thrust and actuator evolution |
| Tracking task | `core.py`, `tracking.py`, `trajectory_reference.py` | State/reference, 23-value observation, reward and integration |
| Training | `gym_env.py`, `train.py` | Gymnasium adapter and nominal TQC training |
| Policy | `policy.py` | Trusted checkpoint verification and deterministic inference |
| Comparisons | `pid.py`, `classical_control.py` | Frozen PID and its unchanged configuration |
| Deployment | `deployment.py`, `deployment_reference.py`, `runtime.py` | Frame contract, calibration/filtering, reference acquisition and runtime lifecycle |
| ROS | `ros2/swim4track_ros/` | Odometry input, thruster output, explicit enable and telemetry |
| Examples | `demo.py`, `visualize.py`, `training_curves.py` | Recorded rollouts, trajectory figures and recorded training diagnostics |

The plant and ROS adapter share the policy observation/action contract. Their
physical command conversion is intentionally different: the Python plant uses
its T200 model, while the audited Stonefish scene uses the documented inverse
thrust calibration. The previous-action observation always uses the raw actor
action. See [model_card.md](model_card.md) for the complete contract.



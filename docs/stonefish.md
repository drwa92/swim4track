# ROS 2 inference in Stonefish

`swim4track_ros` connects the frozen policy to the audited eight-thruster
BlueROV2 Heavy scene. A small ROS node handles messages and lifecycle; the
ROS-independent `TrackingRuntime` handles the reference, observation, policy,
command conversion and output filter. The same interface supports the frozen
PID comparator.

The Python runtime is tested locally. This refactored ROS node still needs an
integration run in the target Stonefish installation; the historical paper
campaigns used `reef_bluerov2_baseline`. The included video is a project
showcase, not evidence that this new node has passed integration testing.

## Install once

Requirements: Ubuntu 22.04, ROS 2 Humble, Python 3.10, and the working
Stonefish workspace. The following **external** scene packages are needed by
the automatic simulator launcher:

- `marine_control_stacks/stonefish_direct.launch.py`
- `stonefish_bluerov2/scenarios/bluerov2_tank_ros2_interface.scn`

These are the integration packages used in the experiments; this repository
does not include their simulator binaries or scene assets. A stock Stonefish
installation may not include them.

From the repository root, replace the workspace path if necessary:

```bash
bash scripts/install_ros.sh --workspace /home/sf_ws
```

The installer creates `.venv-ros`, installs the Python package and inference
dependencies, and builds only `swim4track_ros`. It runs in a child shell, so a
failure does not exit your interactive container. See
[installation.md](installation.md) for prerequisites and recovery from missing
`venv` or packaging tools. Add `--no-ml` only for a PID-only installation.

## Run one trial

Stop other publishers of `/bluerov2/setpoint/pwm`, including old controller or
bridge processes. Use the scene's direct PWM interface. Then run this complete
block from the repository root:

```bash
bash <<'BASH'
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/sf_ws/install/setup.bash
source .venv-ros/bin/activate
source install/setup.bash
bash scripts/run_stonefish_demo.sh --trajectory hold_level
BASH
```

The script launches Stonefish, waits for typed odometry, loads the learned
policy in a disabled state, starts a ROS bag, and enables tracking when the
command interface is ready. It records one trial and shuts down only the
process groups it started. It never overwrites an output directory or retries
a failed trial automatically.

Choose the trial with `--trajectory`:

| Name | Motion | Evaluation time |
|---|---|---:|
| `hold_level` | Position and attitude hold | 60 s |
| `quintic_coupled` | Smooth point-to-point translation and rotation | 62.5 s |
| `circle_level` | Level circle | 62.5 s |
| `helix_unseen` | Helical path | 74.5 s |
| `figure8_unseen` | Figure-eight path | 66.5 s |

Each run includes a preceding 30 s acquisition from the measured pose. The
moving trajectories use a 2× time scale and a 5 s reference-speed ramp. Add
`--controller pid` to use the frozen comparator through the same interface.

`--attach-simulator` uses an already-running, verified direct scene and leaves
it running afterward. `--output NEW_DIRECTORY` selects the evidence directory;
otherwise a unique directory is created under `demo_runs/`. `--scenario FILE`
selects the scene for an automatic launch. For an attached simulator it only
records the supplied file's provenance; it cannot verify the running scene.

A run directory contains `outcome.json`, `identity.json`, `status.jsonl`, the
ROS bag, console logs, environment details and process exit codes. A
`completed` outcome means the scheduled demonstration finished; it is not a
paper campaign score. Screen capture is a separate operation.

## Manual inference

After sourcing the same environment and starting the simulator, launch a
controller without starting motion:

```bash
ros2 launch swim4track_ros inference.launch.py \
  controller:=learned trajectory_name:=circle_level evaluation_s:=62.5
```

From a second terminal with the same environment, start it explicitly:

```bash
ros2 service call /swim4track/set_enabled std_srvs/srv/SetBool '{data: true}'
```

To stop an active run:

```bash
ros2 service call /swim4track/set_enabled std_srvs/srv/SetBool '{data: false}'
```

Stopping sends an immediate zero command. Restarting requires a fresh enable
request and creates a new run identifier. Status consumers use that identifier
to distinguish the current run from previously latched status messages.

The released checkpoint is found automatically and checked against its frozen
SHA-256 before loading. To test a model you trained, provide its path and the
digest recorded by training:

```bash
bash scripts/run_stonefish_demo.sh --trajectory hold_level \
  --model /absolute/path/to/model.zip \
  --model-sha256 YOUR_RECORDED_64_CHARACTER_SHA256
```

The digest is an artifact identity check, not a security guarantee for an
untrusted model. A custom model must retain the 23-input/8-output contract.
Manual launch exposes the same settings as `model_path` and
`expected_model_sha256`.

## Interface contract

The accepted state is **NED world pose and FRD body-frame twist**. No TF lookup
or ENU/NED conversion is performed. Verify the scene's actual coordinates and
channel ordering before adapting this node to another installation. Frame
names alone do not establish a convention. Set `expected_world_frame` and
`expected_body_frame` as launch arguments to check known frame identifiers;
they are empty by default because the repository does not guess them.

| Interface | Type / content |
|---|---|
| `/bluerov2/odometry_sim` | `nav_msgs/msg/Odometry`; sensor-data QoS |
| `/bluerov2/setpoint/pwm` | `std_msgs/msg/Float64MultiArray`; eight normalized commands |
| `/swim4track/set_enabled` | `std_srvs/srv/SetBool`; explicit start/stop |
| `/swim4track/status` | JSON in `std_msgs/msg/String`; readiness, phase, run ID, errors |
| `/swim4track/identity` | JSON in `std_msgs/msg/String`; controller, model and runtime configuration |
| `/swim4track/action` | Eight raw normalized actor outputs |
| `/swim4track/desired_pwm` | Eight mapped commands before output filtering |
| `/swim4track/reference_state` | Position NED, roll/pitch/yaw, linear velocity NED, angular velocity NED; 12 values |
| `/swim4track/controller_timing` | Elapsed seconds, computation milliseconds, deadline-miss indicator |

Channel order: **FrontRight, FrontLeft, BackRight, BackLeft,
DiveFrontRight, DiveFrontLeft, DiveBackRight, DiveBackLeft**.

At 10 Hz, the runtime constructs the exact 23-element training observation,
evaluates the deterministic actor, converts the training T200 force at
`0.60 × action` through the audited Stonefish inverse thrust law, and applies
channel signs `−1`. The output interface uses a 0.25 s first-order filter,
a 2/s slew limit and a magnitude limit of 1. The observation's action memory
contains the previous raw actor action, before mapping and filtering. The
reference depth offset is −2.5 m for this tank scene.

Reference time follows a monotonic clock; a steady-clock ROS timer continues
even if ROS time pauses. The runtime requires `use_sim_time=false`. Both
odometry reception age and control callback gaps are limited to 0.5 s.
Repeated nonzero source timestamps do not refresh the active watchdog.
Timing telemetry includes controller and interface computation; it is not a
replacement for the paper's original inference benchmark.

Readiness requires exactly one PWM publisher and the expected simulator
subscriber `/stonefish_ros2/stonefish_simulator`. A bag recorder alone cannot
satisfy that check. Set `expected_command_subscriber` explicitly if the
verified simulator node has a different name.

Default workspace bounds are NED depth `[0, 2.85]` m, `|x|, |y| ≤ 3` m and
position error ≤ 2 m. Stale or invalid state, changed frame identifiers,
invalid controller output, competing PWM publishers or loss of the simulator
subscriber abort an active run. Disabled nodes are silent. An active abort or
normal shutdown requests zero once; DDS cannot guarantee its delivery after
a crash or forced process kill. This interface is for simulation, not a
physical-vehicle safety system.

## Diagnose a failed run

- **No odometry:** inspect `simulator.log`, the loaded scene and ROS domain.
- **Not ready:** inspect `ros2 topic info /bluerov2/setpoint/pwm --verbose` and
  stop conflicting publishers through their own terminals.
- **Import failure:** activate `.venv-ros` and verify the built controller's
  first line points to that interpreter. Rebuild with the supplied installer.
- **Aborted run:** read `outcome.json`, `status.jsonl` and `controller.log`;
  preserve the directory before making a new attempt.

ROS references: [Humble QoS](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html),
[rclpy Node](https://docs.ros.org/en/humble/p/rclpy/api/node.html),
[SetBool](https://docs.ros.org/en/humble/p/std_srvs/srv/SetBool.html).

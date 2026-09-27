# Use v0.2.0 in the existing ros2sim container

This page uses your existing `ros2sim` container and `/home/sf_ws` workspace.
Install the clean repository in a separate release directory. Keep the older
starter, campaign package and results intact.

## Copy and extract

Download `swim4track_v0.2.0.zip`. In a **host terminal**:

```bash
docker cp "$HOME/Downloads/swim4track_v0.2.0.zip" ros2sim:/home/sf_ws/
docker exec -it ros2sim bash
```

In the resulting **container terminal**, run this complete block:

```bash
bash <<'BASH'
set -eo pipefail
release_dir=/home/sf_ws/releases/v0.2.0
if [ -e "$release_dir" ]; then
  echo "Already exists: $release_dir. Use that copy; do not extract over it."
  exit 1
fi
mkdir -p "$release_dir"
python3 -m zipfile -e /home/sf_ws/swim4track_v0.2.0.zip "$release_dir"
BASH
```

## Install

In the container:

```bash
cd /home/sf_ws/releases/v0.2.0/swim4track
bash scripts/install_ros.sh --workspace /home/sf_ws
```

This uses a regular package installation and builds the ROS executable with
the virtual environment's interpreter. It replaces the earlier fragile
editable-install sequence. If the installer reports missing `ensurepip`,
install the package it identifies inside the container, then rerun the
installer in the same directory. Do not repeat extraction or move an active
workspace.

## Test one learned-policy run

Stop earlier controller/bridge processes using their terminals, then:

```bash
bash <<'BASH'
set -eo pipefail
cd /home/sf_ws/releases/v0.2.0/swim4track
source /opt/ros/humble/setup.bash
source /home/sf_ws/install/setup.bash
source .venv-ros/bin/activate
source install/setup.bash
bash scripts/run_stonefish_demo.sh --trajectory hold_level
BASH
```

This starts one hold demonstration, records its telemetry, and leaves your
interactive container shell open when finished. It does not run a scientific
campaign. The new ROS node needs this integration check; previous campaign
success does not by itself validate the refactored node.

For point-to-point navigation or a circle, change only the last command to:

```bash
bash scripts/run_stonefish_demo.sh --trajectory quintic_coupled
```

or:

```bash
bash scripts/run_stonefish_demo.sh --trajectory circle_level
```

Use those commands inside the same sourced child-shell block. Each invocation
creates a new directory under `demo_runs/`. Add `--attach-simulator` to use a
verified direct scene that is already running; otherwise the script manages
its own simulator process.

See [stonefish.md](stonefish.md) for the interface, manual start/stop,
custom-model inference and troubleshooting. The included GitHub video can be
used without recording another campaign or installing a video toolchain.

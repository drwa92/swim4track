#!/usr/bin/env bash
# Build only this repo's ROS package; never rebuild or replace the research workspace.
set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/usr/bin/python3
VENV="$REPO/.venv-ros"
STONEFISH_WS="${STONEFISH_WS:-}"
WITH_ML=true
usage() {
  cat <<'EOF'
Usage: bash scripts/install_ros.sh --workspace /path/to/stonefish_workspace
       [--venv PATH] [--python PATH] [--no-ml]

For Ubuntu 22.04 / ROS 2 Humble and an already-working Stonefish workspace.
Creates or reuses a dedicated Python 3.10 venv with system ROS packages.
Installs Swim4Track normally, verifies the model, then builds swim4track_ros.
--no-ml installs only the plant/PID; learned inference needs the default install.
Run using bash, not source. Does not install ROS or simulator assets.
EOF
}
while (($#)); do
  case "$1" in
    --workspace|--venv|--python)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --workspace) STONEFISH_WS="$2" ;;
        --venv) VENV="$2" ;;
        --python) PYTHON="$2" ;;
      esac
      shift 2 ;;
    --no-ml) WITH_ML=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -f /opt/ros/humble/setup.bash ]] || { echo "ROS Humble was not found. Run this inside your ROS container/workstation." >&2; exit 1; }
[[ -n "$STONEFISH_WS" && -f "$STONEFISH_WS/install/setup.bash" ]] || { echo "Provide --workspace pointing to your built Stonefish workspace." >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }
env -u PYTHONPATH -u PYTHONHOME "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "ROS Humble requires its matching Python 3.10 interpreter"'
if [[ ! -f "$VENV/pyvenv.cfg" ]]; then
  [[ ! -e "$VENV" ]] || { echo "STOP: $VENV exists but is not a virtual environment; choose a new --venv path." >&2; exit 1; }
  if ! env -u PYTHONPATH -u PYTHONHOME "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1; then
    echo "Missing Python 3.10 venv support. In this Ubuntu container install python3.10-venv, then rerun this script." >&2
    exit 1
  fi
  env -u PYTHONPATH -u PYTHONHOME "$PYTHON" -m venv --system-site-packages "$VENV"
fi
VENV="$(cd -- "$VENV" && pwd)"
[[ -x "$VENV/bin/python" ]] || { echo "Incomplete virtual environment: $VENV; choose a fresh --venv path." >&2; exit 1; }
touch "$VENV/COLCON_IGNORE"
venv_python() { env -u PYTHONPATH -u PYTHONHOME "$VENV/bin/python" "$@"; }
venv_python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing venv is not Python 3.10"'
if ! venv_python -m pip --version >/dev/null 2>&1; then
  venv_python -m ensurepip --upgrade || {
    echo "This environment has no pip. Install python3.10-venv in the ROS container, then rerun, or choose a fresh --venv path." >&2
    exit 1
  }
fi
venv_python -m pip install --upgrade -c "$REPO/constraints/ros-humble.txt" 'pip>=23' 'setuptools>=68,<81' wheel
PACKAGE="$REPO"
if $WITH_ML; then PACKAGE="$REPO[inference]"; fi
venv_python -m pip install -c "$REPO/constraints/ros-humble.txt" "$PACKAGE" colcon-common-extensions
venv_python -m swim4track.policy
if $WITH_ML; then venv_python -m swim4track.policy --load; fi

# ROS setup is sourced after package installation. No nounset: ROS setup scripts
# read optional unset variables. The caller's environment remains untouched.
source /opt/ros/humble/setup.bash
source "$STONEFISH_WS/install/setup.bash"
source "$VENV/bin/activate"
python -c 'import rclpy; from swim4track.policy import verify_checkpoint; print("ROS Python imports and checkpoint available")'
cd -- "$REPO"
python -c 'from colcon_core.command import main; raise SystemExit(main())' \
  build --symlink-install --base-paths ros2 --packages-select swim4track_ros
source "$REPO/install/setup.bash"
ros2 pkg prefix swim4track_ros
printf '\nBuild complete. In each runtime terminal, source:\n'
printf '  source /opt/ros/humble/setup.bash\n  source %q/install/setup.bash\n' "$STONEFISH_WS"
printf '  source %q/bin/activate\n  source %q/install/setup.bash\n' "$VENV" "$REPO"

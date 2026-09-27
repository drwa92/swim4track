#!/usr/bin/env bash
# Run with bash; this child process never changes the caller's shell options.
set -eo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python3
VENV="$REPO/.venv"
PROFILE=inference
usage() {
  cat <<'EOF'
Usage: bash scripts/install_python.sh [--profile core|inference|train]
       [--venv PATH] [--python PATH]

Creates or reuses a virtual environment and installs a regular package copy.
Default: inference plus plots. Core needs no PyTorch; train also adds TensorBoard.
Requires Python >=3.10, venv/ensurepip, and access to a Python package index.
Re-run after source changes. Run using bash, not source.
EOF
}
while (($#)); do
  case "$1" in
    --profile|--venv|--python)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --profile) PROFILE="$2" ;;
        --venv) VENV="$2" ;;
        --python) PYTHON="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
case "$PROFILE" in
  core) EXTRAS=viz ;;
  inference) EXTRAS=inference,viz ;;
  train) EXTRAS=train,viz ;;
  *) echo "Profile must be core, inference, or train" >&2; exit 2 ;;
esac
command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }
env -u PYTHONPATH -u PYTHONHOME "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 is required"'
if [[ ! -f "$VENV/pyvenv.cfg" ]]; then
  [[ ! -e "$VENV" ]] || { echo "STOP: $VENV exists but is not a virtual environment; choose a new --venv path." >&2; exit 1; }
  if ! env -u PYTHONPATH -u PYTHONHOME "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1; then
    echo "Missing venv/ensurepip. On Ubuntu install the venv package matching this Python (e.g. python3.10-venv), then rerun." >&2
    exit 1
  fi
  env -u PYTHONPATH -u PYTHONHOME "$PYTHON" -m venv "$VENV"
fi
VENV="$(cd -- "$VENV" && pwd)"
[[ -x "$VENV/bin/python" ]] || { echo "Incomplete virtual environment: $VENV; choose a fresh --venv path." >&2; exit 1; }
venv_python() { env -u PYTHONPATH -u PYTHONHOME "$VENV/bin/python" "$@"; }
venv_python -c 'import sys; assert sys.version_info >= (3, 10), "Existing venv requires Python >=3.10"'
if ! venv_python -m pip --version >/dev/null 2>&1; then
  venv_python -m ensurepip --upgrade || {
    echo "This environment has no pip. Install the matching Python venv package, then rerun, or choose a fresh --venv path." >&2
    exit 1
  }
fi
venv_python -m pip install --upgrade 'pip>=23' 'setuptools>=68' wheel
# A regular wheel install does not require the build_editable backend hook.
venv_python -m pip install "$REPO[$EXTRAS]"
venv_python -m swim4track.policy
if [[ "$PROFILE" != core ]]; then
  venv_python -m swim4track.policy --load
fi
printf '\nInstalled Swim4Track. Activate with:\n  source %q/bin/activate\n' "$VENV"

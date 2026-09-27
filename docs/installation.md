# Installation

The Python package is independent of ROS. Use a dedicated environment for
training and a separate ROS-compatible environment for Stonefish inference.
Run the supplied scripts with `bash`, not `source`.

## Python plant, inference, or training

From the repository root:

```bash
bash scripts/install_python.sh --profile inference
source .venv/bin/activate
```

| Profile | Includes |
|---|---|
| `core` | NumPy, SciPy and plotting; plant and PID comparator |
| `inference` | Core plus TQC inference dependencies |
| `train` | Inference plus TensorBoard training logs |

Use `--python /path/to/python` to choose the interpreter or `--venv /path/to/env`
to choose a different environment. A valid existing virtual environment is
reused. An existing directory without `pyvenv.cfg` is preserved and rejected.
The scripts do not install apt packages or modify the system Python.

The scripts install a regular copy of this package, so rerun installation after
editing source. For development in a supported environment, editable installation
is optional:

```bash
python -m pip install --upgrade 'pip>=23' 'setuptools>=68' wheel
python -m pip install -e '.[train,viz]'
```

For a manual regular installation in an already activated environment:

```bash
python -m pip install '.[inference,viz]'
python -m swim4track.policy --load
```

Only the supplied checkpoint or a separately trusted checkpoint should be loaded.
The verifier checks identity before the model archive is deserialized.

## ROS 2 Humble

Inside the ROS workstation/container, from the repository root:

```bash
bash scripts/install_ros.sh --workspace /home/sf_ws
```

Replace the workspace path when appropriate. The installer uses Python 3.10,
creates `.venv-ros` with access to system ROS packages, installs the core and
inference dependencies, and invokes colcon through that same Python. It builds
only `ros2/swim4track_ros`. The repository-root `COLCON_IGNORE` keeps parent
workspace scans from discovering additional release copies; the installer
explicitly selects the `ros2` subtree. It does not alter the existing Stonefish source or
historical results. `--no-ml` supports a plant/PID-only installation.

The ROS installer applies `constraints/ros-humble.txt` to retain NumPy 1.x and
setuptools with the legacy packaging APIs used by older ROS tooling. They are
compatibility bounds, not a complete version lock or proof of runtime validation.
The ROS guide lists the required scene and integration packages.

Installation runs in a child shell. Follow the source commands printed at the
end in each runtime terminal; the installer cannot activate an environment in
its parent shell. Do not set `set -u` before sourcing ROS setup files.

## Troubleshooting

**`ensurepip is not available`**

On Ubuntu, install the venv package matching your selected Python. For the
Python 3.10 ROS Humble container, run this *inside that container* as root:

```bash
apt-get update
apt-get install -y python3.10-venv
```

On a non-root workstation, use your usual package-management privileges. Then
rerun the installer. If the earlier attempt left an incomplete environment,
choose a fresh path such as `--venv .venv-ros-v02`; the scripts never delete it.

**`missing the build_editable hook`**

This means the selected build backend did not provide editable-install support.
The release installers use regular installs, so they do not require that hook.
They update build tools inside the chosen environment and clear inherited
`PYTHONPATH`/`PYTHONHOME` only for package-install commands. ROS setup is loaded
afterward for the ROS build. The previous terminal error alone does not establish
which backend or path caused it.

**`/opt/ros/humble/setup.bash: No such file`**

You are probably in the host shell rather than the ROS container, or ROS Humble
is not installed there. In the existing setup enter `docker exec -it ros2sim bash`
from the host, then run the repository commands inside the container.

**Model dependencies unavailable**

A core install and numerical tests work without PyTorch. The inference/training
profiles need a package index or locally supplied wheels. The release validation
report distinguishes local core checks from optional ML and ROS execution.

## Packaging references

- [pip: regular and editable local installs](https://pip.pypa.io/en/stable/topics/local-project-installs/)
- [setuptools: project configuration and build requirements](https://setuptools.pypa.io/en/stable/userguide/quickstart.html)
- [ROS 2 Humble: Python packages and virtual environments](https://docs.ros.org/en/humble/How-To-Guides/Using-Python-Packages.html)

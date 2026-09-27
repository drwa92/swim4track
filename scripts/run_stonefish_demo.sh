#!/usr/bin/env bash
# Execute this script with bash. Never source it into an interactive shell.
set -eo pipefail

usage() {
  cat <<'TXT'
Usage: bash scripts/run_stonefish_demo.sh [options]
  --controller learned|pid       Default: learned
  --trajectory NAME             hold_level|quintic_coupled|circle_level|helix_unseen|figure8_unseen
  --output NEW_DIRECTORY        Default: unique directory under ./demo_runs
  --model MODEL.zip             Optional explicit learned-policy path
  --model-sha256 SHA256          Recorded trusted digest of a custom checkpoint
  --attach-simulator            Use an already-running audited direct scene
  --scenario FILE.scn           Override scene (same audited interface required)
  --help

Run inside the ROS 2 environment after sourcing Humble and your workspace.
This records telemetry, not the screen. See docs/stonefish.md for the interface.
TXT
}
controller=learned
trajectory=circle_level
output=""
model=""
model_sha256=""
scenario=""
scenario_source="unknown"
attach=0
while (($#)); do
  case "$1" in
    --controller|--trajectory|--output|--model|--model-sha256|--scenario)
      if (($# < 2)); then usage >&2; exit 2; fi
      case "$1" in
        --controller) controller=$2 ;;
        --trajectory) trajectory=$2 ;;
        --output) output=$2 ;;
        --model) model=$2 ;;
        --model-sha256) model_sha256=$2 ;;
        --scenario) scenario=$2 ;;
      esac
      shift 2 ;;
    --attach-simulator) attach=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$controller" in pid|learned) ;; *) echo "Controller must be pid or learned" >&2; exit 2;; esac
if [[ "$controller" == pid && ( -n "$model" || -n "$model_sha256" ) ]]; then
  echo "Model options require --controller learned." >&2; exit 2
fi
if [[ -n "$model_sha256" && ! "$model_sha256" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo "--model-sha256 requires a 64-character hexadecimal digest." >&2; exit 2
fi
if [[ -n "$model_sha256" && -z "$model" ]]; then
  echo "Use --model with --model-sha256 for a custom checkpoint." >&2; exit 2
fi
case "$trajectory" in
  hold_level) duration=60.0; scale=1.0 ;;
  quintic_coupled) duration=62.5; scale=2.0 ;;
  circle_level) duration=62.5; scale=2.0 ;;
  helix_unseen) duration=74.5; scale=2.0 ;;
  figure8_unseen) duration=66.5; scale=2.0 ;;
  *) echo "Unsupported demo trajectory: $trajectory" >&2; exit 2 ;;
esac
for executable in ros2 python3 setsid timeout; do
  command -v "$executable" >/dev/null || { echo "Missing $executable; source the ROS 2 environment first." >&2; exit 1; }
done
ros2 pkg prefix swim4track_ros >/dev/null || { echo "Build and source ros2/swim4track_ros first; see docs/stonefish.md." >&2; exit 1; }
if [[ "$attach" == 0 ]]; then
  ros2 pkg prefix marine_control_stacks >/dev/null || {
    echo "This workstation needs the previously validated marine_control_stacks simulator package." >&2
    echo "Its scene/assets are external prerequisites, not included in this repository." >&2
    exit 1
  }
  if [[ -z "$scenario" ]]; then
    scenario="$(ros2 pkg prefix --share stonefish_bluerov2)/scenarios/bluerov2_tank_ros2_interface.scn"
  fi
  [[ -f "$scenario" ]] || { echo "Missing Stonefish scene: $scenario" >&2; exit 1; }
  scenario_source="launched_by_this_script"
elif [[ -n "$scenario" ]]; then
  [[ -f "$scenario" ]] || { echo "Missing supplied scene: $scenario" >&2; exit 1; }
  scenario_source="user_supplied_attached_scene_path_not_verified_at_runtime"
elif scenario_share=$(ros2 pkg prefix --share stonefish_bluerov2 2>/dev/null); then
  scenario_candidate="$scenario_share/scenarios/bluerov2_tank_ros2_interface.scn"
  if [[ -f "$scenario_candidate" ]]; then
    scenario=$scenario_candidate
    scenario_source="installed_default_candidate_not_verified_as_running_scene"
  fi
fi
if ros2 node list 2>/dev/null | python3 -c 'import sys; sys.exit(0 if "/swim4track_controller" in sys.stdin.read().splitlines() else 1)'; then
  echo "A Swim4Track controller already runs; stop it before creating another demo." >&2
  exit 1
fi
if [[ -z "$output" ]]; then
  mkdir -p demo_runs
  output=$(mktemp -d "demo_runs/${controller}_${trajectory}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
else
  [[ ! -e "$output" ]] || { echo "Refusing to overwrite existing output: $output" >&2; exit 1; }
  mkdir -p "$output"
fi
output=$(cd "$output" && pwd)
printf '%s\n' "$output" >"$output/output_path.txt"
date -u --iso-8601=seconds >"$output/started_utc.txt"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 - "$output" "$repo_root" "$scenario" "$scenario_source" <<'PY'
import hashlib, json, os, pathlib, platform, subprocess, sys
output, repo, scene, scene_source = sys.argv[1:]
def git(*args):
    try:
        return subprocess.check_output(["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
git_root = git("rev-parse", "--show-toplevel")
is_repo = git_root is not None and pathlib.Path(git_root).resolve() == pathlib.Path(repo).resolve()
record = {
    "schema": "swim4track_demo_environment_v1",
    "platform": platform.platform(), "machine": platform.machine(),
    "hostname": platform.node(), "python": sys.version,
    "python_executable": sys.executable, "repository_root": repo,
    "git_revision": git("rev-parse", "HEAD") if is_repo else None,
    "git_worktree_status": git("status", "--porcelain") if is_repo else None,
    "ros_distro": os.environ.get("ROS_DISTRO"),
    "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
    "scenario": {"path": scene or None, "identification": scene_source},
}
if scene:
    record["scenario"]["sha256"] = hashlib.sha256(pathlib.Path(scene).read_bytes()).hexdigest()
pathlib.Path(output, "environment.json").write_text(json.dumps(record, indent=2)+"\n")
PY
python3 -m pip freeze >"$output/pip_freeze.txt" 2>"$output/pip_freeze_stderr.txt" || true
if [[ -n "$scenario" ]]; then sha256sum "$scenario" >"$output/scenario.sha256"; fi
sim_pid=""
controller_pid=""
bag_pid=""
cleaned=0
stop_group() {
  local pid=$1
  local signal tick
  local reaped=0
  last_group_exit="not_started"
  [[ -n "$pid" ]] || return 0
  last_group_exit="unknown"
  # Check the owned process group, even if its launcher has already exited.
  # Each signal gets at most five seconds; an uninterruptible child cannot
  # keep this shell blocked indefinitely.
  for signal in INT TERM KILL; do
    kill -0 -- "-$pid" 2>/dev/null || break
    kill -"$signal" -- "-$pid" 2>/dev/null || true
    for ((tick=0; tick<50; tick++)); do
      if [[ "$reaped" == 0 ]]; then
        if ! kill -0 "$pid" 2>/dev/null; then
          if wait "$pid" 2>/dev/null; then last_group_exit=0; else last_group_exit=$?; fi
          reaped=1
        fi
      fi
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 0.1
    done
  done
  if [[ "$reaped" == 0 ]]; then
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid" 2>/dev/null; then last_group_exit=0; else last_group_exit=$?; fi
    else
      last_group_exit="leader_unreaped_after_bounded_shutdown"
    fi
  fi
  if kill -0 -- "-$pid" 2>/dev/null; then
    printf 'Process group %s remains visible after bounded shutdown (possibly zombie/uninterruptible).\n' "$pid" >&2
  fi
}
cleanup() {
  local code=$?
  [[ "$cleaned" == 0 ]] || return
  cleaned=1
  # A second Ctrl-C must not interrupt bounded teardown and evidence writes.
  trap '' INT TERM
  # Ask for a normal stop while DDS and the recorder are still alive.
  if [[ -n "$controller_pid" ]] && kill -0 "$controller_pid" 2>/dev/null; then
    timeout -k 2 5 ros2 service call /swim4track/set_enabled std_srvs/srv/SetBool '{data: false}' \
      >"$output/disable.log" 2>&1 || true
  fi
  stop_group "$controller_pid"
  printf '%s\n' "$last_group_exit" >"$output/controller_exit_code.txt"
  stop_group "$bag_pid"
  printf '%s\n' "$last_group_exit" >"$output/recorder_exit_code.txt"
  stop_group "$sim_pid"
  printf '%s\n' "$last_group_exit" >"$output/simulator_exit_code.txt"
  printf '%s\n' "$code" >"$output/script_exit_code.txt"
  echo "Demo evidence retained: $output"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$attach" == 0 ]]; then
  setsid ros2 launch marine_control_stacks stonefish_direct.launch.py \
    scenario_file:="$scenario" simulation_rate:=100.0 >"$output/simulator.log" 2>&1 &
  sim_pid=$!
fi
echo "Waiting for typed Stonefish odometry (up to 90 s)..."
timeout 90 ros2 topic echo --once --qos-reliability best_effort \
  /bluerov2/odometry_sim nav_msgs/msg/Odometry >"$output/initial_odometry.txt" 2>"$output/odometry_wait.log"
args=(controller:="$controller" trajectory_name:="$trajectory" evaluation_s:="$duration"
      trajectory_time_scale:="$scale")
[[ -z "$model" ]] || args+=(model_path:="$model")
[[ -z "$model_sha256" ]] || args+=(expected_model_sha256:="$model_sha256")
setsid ros2 launch swim4track_ros inference.launch.py "${args[@]}" >"$output/controller.log" 2>&1 &
controller_pid=$!
setsid ros2 bag record -o "$output/bag" \
  /bluerov2/odometry_sim /bluerov2/altitude /bluerov2/setpoint/pwm \
  /swim4track/action /swim4track/desired_pwm /swim4track/reference_state \
  /swim4track/controller_timing /swim4track/status /swim4track/identity \
  >"$output/recorder.log" 2>&1 &
bag_pid=$!
sleep 2
kill -0 "$controller_pid" || { echo "Controller exited; inspect $output/controller.log" >&2; exit 1; }
kill -0 "$bag_pid" || { echo "Recorder exited; inspect $output/recorder.log" >&2; exit 1; }
echo "Demo: $controller / $trajectory. Start actual screen capture on the GUI host."
ros2 run swim4track_ros demo_watch --output "$output" --readiness-timeout 90 --timeout 150 \
  > >(tee "$output/watcher.log") 2>&1
kill -0 "$bag_pid" || { echo "Recorder exited during the run; evidence is incomplete." >&2; exit 1; }

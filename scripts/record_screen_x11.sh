#!/usr/bin/env bash
# Real X11 screen capture. Execute on the host displaying Stonefish, not blindly in Docker.
set -eo pipefail
if [[ "$1" == "--help" || $# -lt 2 ]]; then
  cat <<'TXT'
Usage: bash scripts/record_screen_x11.sh OUTPUT.mp4 WIDTHxHEIGHT [SECONDS] [X,Y]
Example: bash scripts/record_screen_x11.sh demo_runs/pid_gui.mp4 1280x720 115 0,0
Captures the selected screen region as displayed; no generated imagery.
Use OBS window capture on Wayland. DISPLAY must identify the actual GUI display.
TXT
  exit 0
fi
output=$1
geometry=$2
duration=${3:-115}
offset=${4:-0,0}
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "DISPLAY is not set; run on the GUI host." >&2; exit 1; }
[[ "${XDG_SESSION_TYPE:-}" != "wayland" ]] || {
  echo "Use OBS window capture for this Wayland session; see docs/pid_video.md." >&2; exit 1;
}
command -v ffmpeg >/dev/null || { echo "Install ffmpeg on the capture host first." >&2; exit 1; }
[[ "$geometry" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]] || { echo "Invalid WIDTHxHEIGHT" >&2; exit 2; }
[[ "$offset" =~ ^[0-9]+,[0-9]+$ ]] || { echo "Invalid X,Y offset" >&2; exit 2; }
[[ "$duration" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || { echo "Invalid capture duration" >&2; exit 2; }
mkdir -p "$(dirname "$output")"
exec ffmpeg -hide_banner -n -f x11grab -draw_mouse 0 -framerate 30 \
  -video_size "$geometry" -i "${DISPLAY}+${offset}" -t "$duration" \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -preset medium -crf 20 \
  -pix_fmt yuv420p -movflags +faststart "$output"

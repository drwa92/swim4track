# Record an actual PID demonstration

The first public video should show the **frozen PID baseline controlling the
actual running Stonefish simulator**. It verifies installation and gives users
a reproducible reference. A later learned-policy video can use the same path,
camera view and interface. Do not present PID footage as Swim4Track policy
performance or use a rendered Python trajectory as a Stonefish screen capture.

No real Stonefish GUI video was recorded in the artifact-building environment.
The following procedure is for the user's working simulator machine. The
repository intentionally contains no fabricated underwater demo.

## Capture plan

1. Build the interface and follow [stonefish.md](stonefish.md). Use the
   unmodified frozen `pid_kp_high` baseline, `circle_level`, scale 2, 30 s
   acquisition, and 62.5 s evaluation. The active sequence is approximately
   **92.5 s**, plus launch/readiness overhead.
2. Open Stonefish with the established direct scene, arrange a steady camera
   view, then prepare host screen capture. Keep the vehicle visible throughout
   the circle. Avoid window movements and private desktop material.
3. Begin recording **before** enabling the demo. If preparing the view with an
   already-running audited scene, use `--attach-simulator`. Do not run another
   PWM controller while doing this.
4. Run `bash scripts/run_stonefish_demo.sh --controller pid
   --trajectory circle_level --attach-simulator` in the sourced ROS
   environment. Its bag and status logs accompany the video.
5. Stop recording after `outcome.json` reports `completed`. Preserve failures
   and the full recording; a completion message alone is not a new scored
   scientific result. Check the screen video and corresponding telemetry.
6. Produce a short illustrative excerpt from a fixed evaluation interval,
   e.g. evaluation seconds 10–40. Log the trim offsets. Acquisition ends when
   `/swim4track/status` changes to `evaluation`; manually align this with the
   recording or a visible enable event. Do not assume capture time zero equals
   controller time zero.

## OBS (preferred for Wayland)

On the host, use **Window Capture** for the Stonefish simulator and a fixed
1280×720 or 1920×1080 canvas, 30 fps. Record to MKV and remux to MP4 after the
run so an interrupted recording is recoverable. The ROS container does not
need to own the recorder. Set any title overlay to:

> Frozen PID baseline · BlueROV2 Heavy · Stonefish simulation · Circle tracking

For a learned-policy run, change the label only after selecting
`--controller learned` and checking the recorded model identity.

## X11 alternative

Run the helper on the **GUI host**, where `DISPLAY` is valid. Choose the actual
Stonefish screen rectangle; this example captures 1280×720 at the top left for
150 seconds (allow extra time if starting a fresh simulator):

```bash
bash scripts/record_screen_x11.sh demo_runs/pid_stonefish_full.mp4 1280x720 150 0,0
```

Then start the demo in the ROS terminal. The helper refuses to overwrite an
existing video, checks for ffmpeg, and declines a Wayland session in favour of
OBS. It records the screen exactly as displayed, without generated imagery.

To encode an excerpt after identifying the correct capture offset, replace
`CAPTURE_OFFSET_SECONDS` with the start time in the actual video:

```bash
ffmpeg -n -ss CAPTURE_OFFSET_SECONDS -i demo_runs/pid_stonefish_full.mp4 \
  -t 30 -an -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
  demo_runs/pid_stonefish_excerpt.mp4
```

## Publishable demo record

Store the full capture alongside the run evidence outside Git. Put a compact
poster/GIF or short MP4 in a GitHub Release and link it from the README; keep
large ROS bags out of the source repository. Include:

- Controller identity and the original PID/configuration hashes.
- Scene hash, trajectory, reference time scale and command-interface settings.
- ROS/Stonefish versions, machine information, repository revision and run UTC.
- `outcome.json`, `status.jsonl`, recorder/command logs and bag location.
- Capture resolution/frame rate, full-video hash, excerpt offsets, and any
  speed adjustment (default: real time, 1×).

Caption the video as a **software demonstration**, not an additional campaign
trial or evidence of hardware transfer. The same recording plan can later
produce a matched learned-policy demonstration without retuning either controller.

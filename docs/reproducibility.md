# Reproducibility and release scope

The repository contains the numerical Python plant, actuator model, trajectory
references, tracking observation/reward, nominal TQC training recipe, original
nominal seed-17 checkpoint, frozen PID comparator and a cleaned ROS wrapper.
The preserved numerical source files are identified by SHA-256 in
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). The checkpoint is unchanged.

## Reproduce a runnable example

Follow [installation](installation.md), then run the [Python plant](python_plant.md)
or [ROS/Stonefish](stonefish.md) example. Every demonstration uses a new output
directory. Its records identify the controller, trajectory, software and model.
The command/state time alignment is documented in the plant guide.

Training instructions and log interpretation are in [training.md](training.md).
The trainer preserves the default nominal hyperparameters. Different dependency
versions, hardware, scheduling or seeds can produce different learned weights;
this release does not promise bit-identical retraining. The software versions
inside the model archive are provenance, not a fully validated installation lock.

The example under `docs/examples/pid_circle/` is a 40-second execution of the
Python plant with the frozen PID and seed 20000, with a 2× visualization replay.
It can be regenerated with:

```bash
python -m swim4track.demo --controller pid --trajectory circle_level --seconds 40 --seed 20000 --output runs/recreated_pid --plot --video --playback 2
```

`ffmpeg` is required for video generation. Computation timing varies across
machines. Example metrics include acquisition and do not equal paper scoring
windows.

## Reproduce the supplied paper figure

[examples/paper_figure](../examples/paper_figure/README.md) includes the supplied
270-trial metric table, cell-summary cross-check, frozen protocol and original
figure generator. It reproduces metric-table summaries and the figure without
ROS. It does not rescore bags or rerun the experiments.

## What remains in the research archives

The full historical 60/270-trial orchestration, raw ROS bags, complete controller
bank, TQC-DR checkpoint and five-seed evidence are not duplicated in this compact
repository. Preserve those frozen archives for scientific reproduction. Do not
substitute newly generated demo outputs for the manuscript's measurements.

The cleaned ROS runtime changes startup, watchdog, stop and scheduling handling.
The observation/action calibration remains tied to the frozen model contract;
new wrapper validation is still required in the target simulator. See
[VALIDATION.md](VALIDATION.md) for the checks actually completed for this release.
No hardware-transfer or current-robustness qualification is added by this cleanup.

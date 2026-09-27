# Reproduce the paper robustness figure

This self-contained example reproduces `docs/assets/robustness.png` from the
original frozen trial metrics. It contains the unchanged plotting script,
270-row trial table, independent cell-summary cross-check, and actuator
campaign protocol. It does not require ROS, a model download or rosbag files.

From the repository root, use the same Python environment as the quick start:

```bash
python -m pip install -r examples/paper_figure/requirements.txt
python examples/paper_figure/generate_robustness_figure.py --check-only
python examples/paper_figure/generate_robustness_figure.py
```

The first command installs the plotting dependencies. The second validates
the crossed trial design and recomputes the numerical audit without drawing
the figure. The last command writes `figures/robustness_landscape.pdf`,
`figures/robustness_landscape.svg`, and
`figures/robustness_landscape_300dpi.png` under this directory, together with
auditable derived tables in `data/derived/`. It does not replace the README
image automatically. Rendering can vary slightly with fonts and plotting
library versions; the numerical checks are the reproducibility target.

The script generates `MANIFEST_SHA256.txt` for its current outputs.
`INPUT_SHA256SUMS.txt` separately records the unchanged source input/script
digests. From this directory, `sha256sum -c INPUT_SHA256SUMS.txt` verifies
them on Linux. Retain the original inputs when experimenting with plot
appearance.

The figure shows three controllers, three trajectories, six conditions and
five executions per cell. Historical label “Swim4Track” denotes S4T-TQC;
“TQC-DR” denotes the same framework with domain-randomized training. PID is
the classical comparator. The six conditions include unit health, and all
use zero configured current. The repeats describe the fixed campaign's
numerical repeatability rather than independent environmental uncertainty.

Read [DATA_DICTIONARY.md](DATA_DICTIONARY.md) before comparing metrics:
tracking RMSE uses the full scored interval, while tolerance occupancy and
sustained success exclude its first 20%. The plot script verifies the trial
counts, health schedules, finite values and agreement with the original
cell means. It recomputes the figure from metric tables; it does not rescore
raw bags or rerun the experiment.

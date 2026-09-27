"""Plot recorded training returns and optimizer losses from a new training run.

No data are inferred from model weights, interpolated, or manufactured when
a short run has not yet emitted a metric.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = (
    ("rollout/ep_rew_mean", "Training return", "Mean episode return"),
    ("train/actor_loss", "Actor objective", "Actor loss"),
    ("train/critic_loss", "Critic objective", "Critic loss"),
)


def read_training_scalars(path):
    """Read aligned columns, retaining absent measurements as NaN."""
    keys = ["time/total_timesteps"] + [item[0] for item in METRICS]
    values = {key: [] for key in keys}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if "time/total_timesteps" not in (reader.fieldnames or []):
            raise ValueError("training CSV must contain time/total_timesteps")
        for row in reader:
            for key in keys:
                cell = row.get(key, "")
                try:
                    values[key].append(float(cell) if cell else float("nan"))
                except ValueError as exc:
                    raise ValueError(f"non-numeric value in {key}: {cell!r}") from exc
    values = {key: np.asarray(column) for key, column in values.items()}
    valid_time = np.isfinite(values[keys[0]])
    if not any(np.any(valid_time & np.isfinite(values[key])) for key in keys[1:]):
        raise ValueError("no recorded return or loss samples; run long enough to finish episodes and log updates")
    return values


def plot_training_curves(run, output):
    """Write PNG/PDF diagnostic plots to a new output directory."""
    run, output = Path(run), Path(output)
    data = read_training_scalars(run / "logs" / "progress.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install plotting dependencies: pip install 'swim4track[viz]'") from exc
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    time = data["time/total_timesteps"]
    for ax, (key, title, ylabel) in zip(axes, METRICS):
        valid = np.isfinite(time) & np.isfinite(data[key])
        if np.any(valid):
            ax.plot(time[valid], data[key][valid], color="#007a87", linewidth=1.5)
        else:
            ax.text(.5, .5, "No samples recorded", ha="center", va="center",
                    transform=ax.transAxes, color="#56616d")
        ax.set(title=title, xlabel="Environment transitions", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "training_curves.png", dpi=200)
    figure.savefig(output / "training_curves.pdf")
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="training output directory")
    parser.add_argument("--output", type=Path, required=True, help="new figure directory")
    args = parser.parse_args(argv)
    try:
        output = plot_training_curves(args.run, args.output)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Saved recorded training diagnostics to {output.resolve()}")


if __name__ == "__main__":
    main()

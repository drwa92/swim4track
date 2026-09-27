"""Regenerate plots and an explicitly labelled Python-plant replay from data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def render_demo(directory, video=False, playback=2.0, fps=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
    directory = Path(directory)
    if not np.isfinite(playback) or playback <= 0:
        raise ValueError("playback must be positive and finite")
    with np.load(directory / "rollout.npz", allow_pickle=False) as archive:
        data = dict(archive)
    meta = json.loads((directory / "summary.json").read_text())
    t, state, reference = data["time_s"], data["state"], data["reference_pose"]
    name = "PID baseline" if meta["controller"] == "pid" else "Swim4Track · frozen TQC"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#cad5df", "text.color": "#17344c",
                         "axes.labelcolor": "#17344c", "xtick.color": "#526b7b", "ytick.color": "#526b7b"})
    fig = plt.figure(figsize=(12, 6.8), facecolor="#f4f7fa", layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], height_ratios=[1, 1, 1])
    path_ax = fig.add_subplot(grid[:, 0]); depth_ax = fig.add_subplot(grid[0, 1])
    error_ax = fig.add_subplot(grid[1, 1]); command_ax = fig.add_subplot(grid[2, 1])
    title = fig.suptitle(f"{name}  |  Python plant", fontsize=19, fontweight="bold", x=.025, ha="left")
    for ax in (path_ax, depth_ax, error_ax, command_ax):
        ax.set_facecolor("white"); ax.grid(alpha=.22, color="#9cacbc")
    path_ax.plot(reference[:, 0], reference[:, 1], "--", color="#9daebc", lw=2, label="Reference")
    actual, = path_ax.plot(state[:, 0], state[:, 1], color="#008c95", lw=2.5, label="Vehicle")
    marker, = path_ax.plot([state[-1, 0]], [state[-1, 1]], "o", color="#008c95", ms=8)
    target, = path_ax.plot([reference[-1, 0]], [reference[-1, 1]], "o", mfc="white", mec="#617b92", ms=7)
    path_ax.set(xlabel="North [m]", ylabel="East [m]", title="Trajectory tracking · top view")
    path_ax.set_aspect("equal", adjustable="box"); path_ax.margins(.15); path_ax.legend(loc="upper left", frameon=False)
    progress = path_ax.text(.03, .035, "", transform=path_ax.transAxes, fontsize=10,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    depth_ax.plot(t, reference[:, 2], "--", color="#9daebc", lw=1.5)
    depth_line, = depth_ax.plot(t, state[:, 2], color="#008c95", lw=1.7)
    depth_ax.set(ylabel="Depth [m]", title="Depth · positive down"); depth_ax.invert_yaxis()
    error_line, = error_ax.plot(t, data["position_error_m"], color="#ed964b", lw=1.7)
    error_ax.set(ylabel="Error [m]", title="Position tracking error"); error_ax.set_ylim(bottom=0)
    command_lines = command_ax.plot(t[:-1], data["plant_command"], lw=1.1, alpha=.85)
    command_ax.set(ylabel="Command", xlabel="Simulation time [s]", title="Eight normalized plant commands", ylim=(-.65, .65))
    for ax in (depth_ax, error_ax, command_ax):
        ax.set_xlim(0, max(t[-1], .1))
    progress.set_text(f"{meta['trajectory']}  ·  seed {meta['seed']}\nFull rollout: {t[-1]:.1f} s")
    fig.savefig(directory / "tracking.png", dpi=170)
    fig.savefig(directory / "tracking.pdf")
    if video:
        if not writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP4; rollout and plots have already been saved")
        # Exactly ceil(T / speed * fps) frames; include the final simulated state.
        frame_count = max(2, int(np.ceil(t[-1] / playback * fps)))
        indices = np.rint(np.linspace(0, len(t)-1, frame_count)).astype(int)
        def update(index):
            n = int(index) + 1
            actual.set_data(state[:n, 0], state[:n, 1])
            marker.set_data([state[index, 0]], [state[index, 1]])
            target.set_data([reference[index, 0]], [reference[index, 1]])
            depth_line.set_data(t[:n], state[:n, 2])
            error_line.set_data(t[:n], data["position_error_m"][:n])
            for k, line in enumerate(command_lines):
                line.set_data(t[:index], data["plant_command"][:index, k])
            progress.set_text(f"{meta['trajectory']}  ·  {playback:g}× playback\nt = {t[index]:.1f} s  |  error = {data['position_error_m'][index]:.3f} m")
            return actual, marker, target, depth_line, error_line, progress, *command_lines
        animation = FuncAnimation(fig, update, frames=indices, interval=1000/fps, blit=False)
        animation.save(directory / "demo.mp4", writer=FFMpegWriter(fps=fps, codec="libx264",
            metadata={"title": f"{name} — actual Python plant rollout replay", "comment": f"{playback:g}x playback; not Stonefish footage"},
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]), dpi=100)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--playback", type=float, default=2.0)
    args = parser.parse_args(argv)
    render_demo(args.directory, args.video, args.playback)


if __name__ == "__main__":
    main()

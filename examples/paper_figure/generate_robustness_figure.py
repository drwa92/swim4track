#!/usr/bin/env python3
"""Reproduce the Swim4Track cell-resolved actuator-loss evidence figure.

The script reads only the frozen campaign trial table and protocol copied into
this package. It derives all plotted quantities, cross-checks the supplied cell
summary, writes auditable CSV/JSON outputs, and exports PDF, SVG, and 300-dpi PNG.
No inferential confidence intervals or hypothesis tests are computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "input"
DERIVED_DIR = ROOT / "data" / "derived"
FIGURE_DIR = ROOT / "figures"

TRIAL_CSV = INPUT_DIR / "valid_trial_metrics.csv"
REFERENCE_CELL_CSV = INPUT_DIR / "cell_summaries.csv"
PROTOCOL_JSON = INPUT_DIR / "stonefish_robustness_campaign_v2.json"

CONTROLLER_ORDER = ["pid", "tqc_dr", "swim4track"]
CONTROLLER_LABEL = {
    "pid": "PID",
    "tqc_dr": "TQC-DR",
    "swim4track": "Swim4Track",
}
CONTROLLER_COLOR = {
    "pid": "#4D4D4D",
    "tqc_dr": "#E69F00",
    "swim4track": "#0072B2",
}
CONTROLLER_MARKER = {"pid": "D", "tqc_dr": "s", "swim4track": "o"}

PATH_ORDER = ["circle_level", "helix_unseen", "figure8_unseen"]
PATH_LABEL = {
    "circle_level": "Circle",
    "helix_unseen": "Held-out helix",
    "figure8_unseen": "Held-out figure eight",
}
PATH_SHORT = {
    "circle_level": "Circle",
    "helix_unseen": "Helix",
    "figure8_unseen": "Figure 8",
}
PATH_COLOR = {
    "circle_level": "#009E73",
    "helix_unseen": "#CC79A7",
    "figure8_unseen": "#D55E00",
}
PATH_MARKER = {"circle_level": "o", "helix_unseen": "^", "figure8_unseen": "s"}

CONDITION_ORDER = [
    "nominal_adapter",
    "horizontal_pair_loss20_gradual",
    "horizontal_pair_loss40_gradual",
    "single_horizontal_loss50_gradual",
    "vertical_pair_loss40_gradual",
    "mixed_horizontal_vertical_loss30_abrupt",
]
CONDITION_CODE = {condition: f"C{i}" for i, condition in enumerate(CONDITION_ORDER)}
CONDITION_DISPLAY = {
    "nominal_adapter": ("Unit health", "H0,H1", "1.00", "none"),
    "horizontal_pair_loss20_gradual": ("Horizontal pair: 20% loss", "H0,H1", "0.80", "10 s"),
    "horizontal_pair_loss40_gradual": ("Horizontal pair: 40% loss", "H0,H1", "0.60", "10 s"),
    "single_horizontal_loss50_gradual": ("Single horizontal: 50% loss", "H0", "0.50", "10 s"),
    "vertical_pair_loss40_gradual": ("Vertical pair: 40% loss", "V4,V5", "0.60", "10 s"),
    "mixed_horizontal_vertical_loss30_abrupt": ("Mixed H/V: 30% loss", "H0,V4", "0.70", "abrupt"),
}
CONDITION_TILE_NAME = {
    "nominal_adapter": "Unit health",
    "horizontal_pair_loss20_gradual": "Horizontal pair\n20% loss",
    "horizontal_pair_loss40_gradual": "Horizontal pair\n40% loss",
    "single_horizontal_loss50_gradual": "Single horizontal\n50% loss",
    "vertical_pair_loss40_gradual": "Vertical pair\n40% loss",
    "mixed_horizontal_vertical_loss30_abrupt": "Mixed H/V\n30% loss",
}

REQUIRED_TRIAL_COLUMNS = {
    "trial_id",
    "controller_id",
    "path",
    "condition_id",
    "repetition",
    "current_y_mps",
    "final_health",
    "affected_thrusters",
    "loss_ramp_s",
    "evaluation_duration_s",
    "position_rmse_m",
    "attitude_rmse_deg",
    "time_in_tolerance_fraction",
    "continuous_success",
    "final_success",
    "mean_abs_pwm",
    "total_variation_l1",
    "compute_p99_ms",
    "deadline_miss_fraction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trials = pd.read_csv(TRIAL_CSV)
    reference_cells = pd.read_csv(REFERENCE_CELL_CSV)
    protocol = json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))

    missing = REQUIRED_TRIAL_COLUMNS.difference(trials.columns)
    require(not missing, f"Trial table is missing columns: {sorted(missing)}")
    require(len(trials) == int(protocol["expected_trials"]), "Unexpected trial count")
    require(trials["trial_id"].is_unique, "Trial identifiers are not unique")
    require(set(trials["controller_id"]) == set(CONTROLLER_ORDER), "Controller set changed")
    require(set(trials["path"]) == set(PATH_ORDER), "Trajectory set changed")
    require(set(trials["condition_id"]) == set(CONDITION_ORDER), "Condition set changed")
    require(np.allclose(trials["current_y_mps"].to_numpy(float), 0.0), "Nonzero current detected")

    protocol_conditions = {entry["id"]: entry for entry in protocol["conditions"]}
    require(list(protocol_conditions) == CONDITION_ORDER, "Protocol condition order changed")
    for condition_id, spec in protocol_conditions.items():
        subset = trials.loc[trials["condition_id"] == condition_id]
        expected_channels = ";".join(str(v) for v in spec["affected_thrusters"])
        require(set(subset["affected_thrusters"].astype(str)) == {expected_channels},
                f"Affected channels do not match protocol for {condition_id}")
        require(np.allclose(subset["final_health"], float(spec["final_thruster_health"])),
                f"Final health does not match protocol for {condition_id}")
        require(np.allclose(subset["loss_ramp_s"], float(spec["loss_ramp_s"])),
                f"Ramp duration does not match protocol for {condition_id}")

    numeric_to_check = [
        "position_rmse_m",
        "attitude_rmse_deg",
        "time_in_tolerance_fraction",
        "mean_abs_pwm",
        "total_variation_l1",
        "evaluation_duration_s",
        "compute_p99_ms",
        "deadline_miss_fraction",
    ]
    require(np.isfinite(trials[numeric_to_check].to_numpy(float)).all(), "Non-finite metric detected")
    require(((trials["time_in_tolerance_fraction"] >= 0.0) &
             (trials["time_in_tolerance_fraction"] <= 1.0)).all(),
            "Tolerance occupancy lies outside [0,1]")

    counts = trials.groupby(["controller_id", "path", "condition_id"], observed=True).size()
    require(len(counts) == 54, "Expected 54 controller-trajectory-condition cells")
    require((counts == int(protocol["repetitions_per_cell"])).all(),
            "Every cell must contain five repeat executions")
    repetitions = trials.groupby(
        ["controller_id", "path", "condition_id"], observed=True
    )["repetition"].apply(lambda values: tuple(sorted(int(v) for v in values)))
    require((repetitions == (1, 2, 3, 4, 5)).all(), "Repetition identifiers are incomplete")
    return trials, reference_cells, protocol


def derive_outputs(
    trials: pd.DataFrame, reference_cells: pd.DataFrame, protocol: dict
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict
]:
    working = trials.copy()
    working["command_variation_rate_s_inv"] = (
        working["total_variation_l1"] / working["evaluation_duration_s"]
    )

    cell = (
        working.groupby(["controller_id", "path", "condition_id"], observed=True)
        .agg(
            n=("trial_id", "size"),
            position_rmse_m_mean=("position_rmse_m", "mean"),
            position_rmse_m_repeat_sd=("position_rmse_m", "std"),
            attitude_rmse_deg_mean=("attitude_rmse_deg", "mean"),
            attitude_rmse_deg_repeat_sd=("attitude_rmse_deg", "std"),
            tolerance_fraction_mean=("time_in_tolerance_fraction", "mean"),
            tolerance_fraction_repeat_sd=("time_in_tolerance_fraction", "std"),
            sustained_success_count=("continuous_success", "sum"),
            terminal_success_count=("final_success", "sum"),
        )
        .reset_index()
    )
    cell["controller_label"] = cell["controller_id"].map(CONTROLLER_LABEL)
    cell["trajectory_label"] = cell["path"].map(PATH_SHORT)
    cell["condition_code"] = cell["condition_id"].map(CONDITION_CODE)
    cell = cell.sort_values(
        ["controller_id", "path", "condition_id"],
        key=lambda series: series.map(
            {**{v: i for i, v in enumerate(CONTROLLER_ORDER)},
             **{v: i for i, v in enumerate(PATH_ORDER)},
             **{v: i for i, v in enumerate(CONDITION_ORDER)}}
        ),
    ).reset_index(drop=True)

    reference_keyed = reference_cells.set_index(["controller_id", "path", "condition_id"])
    derived_keyed = cell.set_index(["controller_id", "path", "condition_id"])
    crosscheck = {}
    for source_col, derived_col in [
        ("position_rmse_m_mean", "position_rmse_m_mean"),
        ("attitude_rmse_deg_mean", "attitude_rmse_deg_mean"),
        ("time_in_tolerance_fraction_mean", "tolerance_fraction_mean"),
    ]:
        aligned = reference_keyed.loc[derived_keyed.index, source_col].to_numpy(float)
        calculated = derived_keyed[derived_col].to_numpy(float)
        crosscheck[source_col] = float(np.max(np.abs(aligned - calculated)))
    require(max(crosscheck.values()) < 1e-12, "Derived cells disagree with frozen cell summary")

    unit = (
        cell.loc[cell["condition_id"] == "nominal_adapter",
                 ["controller_id", "path", "position_rmse_m_mean"]]
        .rename(columns={"position_rmse_m_mean": "unit_health_position_rmse_m"})
    )
    penalty = cell.loc[cell["condition_id"] != "nominal_adapter", [
        "controller_id", "controller_label", "path", "trajectory_label",
        "condition_id", "condition_code", "position_rmse_m_mean",
    ]].merge(unit, on=["controller_id", "path"], validate="many_to_one")
    penalty = penalty.rename(columns={"position_rmse_m_mean": "fault_position_rmse_m"})
    penalty["degradation_penalty_ratio"] = (
        penalty["fault_position_rmse_m"] / penalty["unit_health_position_rmse_m"]
    )

    pivot = cell.pivot(
        index=["path", "condition_id"], columns="controller_id",
        values=["position_rmse_m_mean", "attitude_rmse_deg_mean", "tolerance_fraction_mean"],
    )
    paired_rows = []
    for path in PATH_ORDER:
        for condition in CONDITION_ORDER:
            idx = (path, condition)
            s_pos = float(pivot.loc[idx, ("position_rmse_m_mean", "swim4track")])
            t_pos = float(pivot.loc[idx, ("position_rmse_m_mean", "tqc_dr")])
            p_pos = float(pivot.loc[idx, ("position_rmse_m_mean", "pid")])
            s_att = float(pivot.loc[idx, ("attitude_rmse_deg_mean", "swim4track")])
            t_att = float(pivot.loc[idx, ("attitude_rmse_deg_mean", "tqc_dr")])
            s_tol = float(pivot.loc[idx, ("tolerance_fraction_mean", "swim4track")])
            t_tol = float(pivot.loc[idx, ("tolerance_fraction_mean", "tqc_dr")])
            paired_rows.append({
                "path": path,
                "trajectory_label": PATH_SHORT[path],
                "condition_id": condition,
                "condition_code": CONDITION_CODE[condition],
                "pid_position_rmse_m": p_pos,
                "tqc_dr_position_rmse_m": t_pos,
                "swim4track_position_rmse_m": s_pos,
                "swim4track_position_lower_than_pid": bool(s_pos < p_pos),
                "swim4track_position_lower_than_tqc_dr": bool(s_pos < t_pos),
                "tqc_dr_to_swim4track_attitude_rmse_ratio": t_att / s_att,
                "swim4track_attitude_lower_than_tqc_dr": bool(s_att < t_att),
                "swim4track_minus_tqc_dr_tolerance_percentage_points": 100.0 * (s_tol - t_tol),
                "swim4track_tolerance_higher_than_tqc_dr": bool(s_tol > t_tol),
            })
    paired = pd.DataFrame(paired_rows)

    condition_rows = []
    protocol_conditions = {entry["id"]: entry for entry in protocol["conditions"]}
    for condition in CONDITION_ORDER:
        spec = protocol_conditions[condition]
        label, software_channels, display_health, onset = CONDITION_DISPLAY[condition]
        condition_rows.append({
            "condition_code": CONDITION_CODE[condition],
            "condition_id": condition,
            "figure_label": label,
            "affected_software_channels": software_channels,
            "affected_thruster_indices": ";".join(map(str, spec["affected_thrusters"])),
            "final_health_factor": float(spec["final_thruster_health"]),
            "loss_ramp_s": float(spec["loss_ramp_s"]),
            "figure_onset_label": onset,
            "display_health_crosscheck": float(display_health),
        })
    condition_definitions = pd.DataFrame(condition_rows)

    overall = cell.groupby("controller_id", observed=True).agg(
        position_rmse_m=("position_rmse_m_mean", "mean"),
        attitude_rmse_deg=("attitude_rmse_deg_mean", "mean"),
        tolerance_fraction=("tolerance_fraction_mean", "mean"),
    )
    pid = overall.loc["pid"]
    tqc = overall.loc["tqc_dr"]
    swim = overall.loc["swim4track"]

    unit_trials = working.loc[working["condition_id"] == "nominal_adapter"]
    unit_means = unit_trials.groupby("controller_id", observed=True).agg(
        mean_abs_pwm=("mean_abs_pwm", "mean"),
        command_variation_rate_s_inv=("command_variation_rate_s_inv", "mean"),
    )
    success_counts = working.groupby("controller_id", observed=True)[
        ["continuous_success", "final_success"]
    ].sum()

    headline_rows = [
        {
            "claim_id": "position_vs_pid",
            "value": 100.0 * (1.0 - swim["position_rmse_m"] / pid["position_rmse_m"]),
            "unit": "percent lower",
            "definition": "Equal-weight mean of 18 cell means; PID is denominator",
        },
        {
            "claim_id": "position_vs_tqc_dr",
            "value": 100.0 * (swim["position_rmse_m"] / tqc["position_rmse_m"] - 1.0),
            "unit": "percent higher",
            "definition": "Equal-weight mean of 18 cell means; TQC-DR is denominator",
        },
        {
            "claim_id": "attitude_vs_tqc_dr",
            "value": 100.0 * (1.0 - swim["attitude_rmse_deg"] / tqc["attitude_rmse_deg"]),
            "unit": "percent lower",
            "definition": "Equal-weight mean of 18 cell means; TQC-DR is denominator",
        },
        {
            "claim_id": "tolerance_vs_tqc_dr",
            "value": 100.0 * (swim["tolerance_fraction"] - tqc["tolerance_fraction"]),
            "unit": "percentage points higher",
            "definition": "Equal-weight mean of 18 cell means",
        },
        {
            "claim_id": "unit_health_mean_abs_command_vs_tqc_dr",
            "value": 100.0 * (1.0 - unit_means.loc["swim4track", "mean_abs_pwm"] /
                              unit_means.loc["tqc_dr", "mean_abs_pwm"]),
            "unit": "percent lower",
            "definition": "Unit-health nominal-adapter trials only",
        },
        {
            "claim_id": "unit_health_command_variation_rate_vs_tqc_dr",
            "value": 100.0 * (1.0 - unit_means.loc["swim4track", "command_variation_rate_s_inv"] /
                              unit_means.loc["tqc_dr", "command_variation_rate_s_inv"]),
            "unit": "percent lower",
            "definition": "Mean total variation divided by run duration; unit-health trials only",
        },
    ]
    headline_table = pd.DataFrame(headline_rows)

    audit = {
        "schema": "swim4track_robustness_evidence_landscape_audit_v1",
        "source_files": {
            TRIAL_CSV.name: {"sha256": sha256(TRIAL_CSV), "rows": int(len(trials))},
            REFERENCE_CELL_CSV.name: {"sha256": sha256(REFERENCE_CELL_CSV),
                                      "rows": int(len(reference_cells))},
            PROTOCOL_JSON.name: {"sha256": sha256(PROTOCOL_JSON)},
        },
        "design": {
            "trial_count": int(len(trials)),
            "unique_trial_count": int(trials["trial_id"].nunique()),
            "cell_count": int(len(cell)),
            "controllers": CONTROLLER_ORDER,
            "trajectories": PATH_ORDER,
            "conditions": CONDITION_ORDER,
            "executions_per_cell": sorted(int(v) for v in cell["n"].unique()),
            "all_configured_currents_mps": sorted(float(v) for v in trials["current_y_mps"].unique()),
            "repeat_interpretation": "deterministic numerical repeatability, not independent environmental samples",
        },
        "frozen_cell_summary_max_abs_difference": crosscheck,
        "equal_weight_cell_means": {
            controller: {
                "position_rmse_m": float(overall.loc[controller, "position_rmse_m"]),
                "attitude_rmse_deg": float(overall.loc[controller, "attitude_rmse_deg"]),
                "tolerance_fraction": float(overall.loc[controller, "tolerance_fraction"]),
            }
            for controller in CONTROLLER_ORDER
        },
        "directional_cell_counts": {
            "swim4track_position_lower_than_pid": int(paired["swim4track_position_lower_than_pid"].sum()),
            "swim4track_position_lower_than_tqc_dr": int(paired["swim4track_position_lower_than_tqc_dr"].sum()),
            "swim4track_attitude_lower_than_tqc_dr": int(paired["swim4track_attitude_lower_than_tqc_dr"].sum()),
            "swim4track_tolerance_higher_than_tqc_dr": int(paired["swim4track_tolerance_higher_than_tqc_dr"].sum()),
            "denominator_cells": 18,
        },
        "headline_values": {row["claim_id"]: float(row["value"]) for row in headline_rows},
        "success_execution_counts": {
            controller: {
                "sustained": int(success_counts.loc[controller, "continuous_success"]),
                "terminal_pose": int(success_counts.loc[controller, "final_success"]),
                "total": int((working["controller_id"] == controller).sum()),
            }
            for controller in CONTROLLER_ORDER
        },
        "timing": {
            "swim4track_mean_trial_level_p99_ms": float(
                working.loc[working["controller_id"] == "swim4track", "compute_p99_ms"].mean()
            ),
            "maximum_recorded_deadline_miss_fraction": float(working["deadline_miss_fraction"].max()),
        },
        "degradation_penalty_definition": (
            "fault-condition cell-mean position RMSE divided by the trajectory-matched "
            "unit-health cell-mean position RMSE for the same controller"
        ),
        "interpretation_limits": [
            "A degradation-penalty ratio below one is not evidence that actuator loss is beneficial.",
            "No interval in this package is an inferential population confidence interval.",
            "All campaign conditions use zero configured water current.",
            "The campaign evaluates passive closed-loop resilience, not fault diagnosis or reconfiguration.",
        ],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    return cell, penalty, paired, condition_definitions, headline_table, audit


def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.3,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.10) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold",
            ha="left", va="bottom")


def draw_condition_strip(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_label(ax, "(a)", x=0.0, y=0.92)
    ax.text(0.045, 0.985, "Frozen actuator-effectiveness conditions", transform=ax.transAxes,
            fontsize=8.4, fontweight="bold", va="top")
    gap = 0.012
    left = 0.02
    width = (0.96 - gap * 5) / 6
    y, height = 0.07, 0.69
    fills = ["#F3F3F3", "#FFF4D6", "#FFE7BE", "#FBE3D5", "#DCEEF5", "#E8DFF2"]
    for idx, condition in enumerate(CONDITION_ORDER):
        _, channels, health, onset = CONDITION_DISPLAY[condition]
        name = CONDITION_TILE_NAME[condition]
        x = left + idx * (width + gap)
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes, linewidth=0.65, edgecolor="#A0A0A0",
            facecolor=fills[idx], clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height - 0.08, f"C{idx}", transform=ax.transAxes,
                fontsize=7.4, fontweight="bold", ha="center", va="top")
        ax.text(x + width / 2, y + height - 0.28, name, transform=ax.transAxes,
                fontsize=5.85, fontweight="bold", ha="center", va="center",
                linespacing=0.90)
        ax.text(x + width / 2, y + 0.255, f"channels {channels}", transform=ax.transAxes,
                fontsize=6.0, ha="center", va="center")
        ax.text(x + width / 2, y + 0.09, f"health {health}; {onset}", transform=ax.transAxes,
                fontsize=6.0, ha="center", va="center", color="#333333")


def draw_absolute_panel(fig: plt.Figure, subspec, trials: pd.DataFrame, cell: pd.DataFrame) -> None:
    subgrid = subspec.subgridspec(1, 3, wspace=0.12)
    axes = [fig.add_subplot(subgrid[0, i]) for i in range(3)]
    x_base = np.arange(len(CONDITION_ORDER), dtype=float)
    controller_offset = {"pid": -0.22, "tqc_dr": 0.0, "swim4track": 0.22}
    repeat_offset = np.linspace(-0.052, 0.052, 5)

    for path_idx, (ax, path) in enumerate(zip(axes, PATH_ORDER)):
        subset_path = trials.loc[trials["path"] == path]
        for controller in CONTROLLER_ORDER:
            for condition_idx, condition in enumerate(CONDITION_ORDER):
                raw = (
                    subset_path.loc[
                        (subset_path["controller_id"] == controller) &
                        (subset_path["condition_id"] == condition)
                    ]
                    .sort_values("repetition")
                )
                require(len(raw) == 5, "Plotting expected five repeat executions")
                x = condition_idx + controller_offset[controller]
                ax.scatter(
                    x + repeat_offset,
                    raw["position_rmse_m"],
                    s=10.0,
                    marker=CONTROLLER_MARKER[controller],
                    facecolor=CONTROLLER_COLOR[controller],
                    edgecolor="none",
                    alpha=0.28,
                    zorder=2,
                )
                mean_row = cell.loc[
                    (cell["controller_id"] == controller) &
                    (cell["path"] == path) &
                    (cell["condition_id"] == condition)
                ]
                require(len(mean_row) == 1, "Missing cell mean")
                ax.scatter(
                    [x], [float(mean_row.iloc[0]["position_rmse_m_mean"])],
                    s=34,
                    marker=CONTROLLER_MARKER[controller],
                    facecolor=CONTROLLER_COLOR[controller],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=4,
                )
        ax.set_title(PATH_LABEL[path], pad=4.0, fontweight="bold")
        ax.set_xlim(-0.55, 5.55)
        ax.set_ylim(0.0, 0.34)
        ax.set_xticks(x_base, [f"C{i}" for i in range(6)])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        if path_idx == 0:
            ax.set_ylabel("Position RMSE (m)")
            panel_label(ax, "(b)", x=-0.04, y=1.16)
        else:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)
        style_axis(ax)

    handles = [
        Line2D([0], [0], marker=CONTROLLER_MARKER[c], color="none",
               markerfacecolor=CONTROLLER_COLOR[c], markeredgecolor="white",
               markersize=5.2, label=CONTROLLER_LABEL[c])
        for c in CONTROLLER_ORDER
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True, ncol=1,
                   borderpad=0.35, handletextpad=0.35, labelspacing=0.25,
                   framealpha=0.92, edgecolor="#D0D0D0")
    axes[0].text(0.025, 0.70, "small: execution\nlarge: cell mean",
                 transform=axes[0].transAxes, fontsize=5.8, color="#555555",
                 va="top")
    axes[0].text(0.02, 0.02, "Swim4Track < PID: 18/18 cells",
                 transform=axes[0].transAxes, fontsize=6.3, va="bottom")
    axes[2].text(0.98, 0.02, "TQC-DR < Swim4Track: 15/18 cells",
                 transform=axes[2].transAxes, fontsize=6.3, va="bottom", ha="right")
    axes[1].text(0.5, 1.20, "Position RMSE across all 54 fixed cells",
                 transform=axes[1].transAxes, fontsize=7.5, fontweight="bold",
                 ha="center", va="bottom")


def draw_penalty_panel(fig: plt.Figure, subspec, penalty: pd.DataFrame) -> None:
    subgrid = subspec.subgridspec(1, 4, width_ratios=[1, 1, 1, 0.065], wspace=0.22)
    axes = [fig.add_subplot(subgrid[0, i]) for i in range(3)]
    color_ax = fig.add_subplot(subgrid[0, 3])
    fault_conditions = CONDITION_ORDER[1:]
    cmap = LinearSegmentedColormap.from_list(
        "fault_penalty", ["#2C7BB6", "#D7EEF7", "#FFFFFF", "#FDD49E", "#D7301F"], N=256
    )
    norm = TwoSlopeNorm(vmin=0.60, vcenter=1.0, vmax=1.85)
    image = None

    for controller_idx, (ax, controller) in enumerate(zip(axes, CONTROLLER_ORDER)):
        table = (
            penalty.loc[penalty["controller_id"] == controller]
            .pivot(index="path", columns="condition_id", values="degradation_penalty_ratio")
            .reindex(index=PATH_ORDER, columns=fault_conditions)
        )
        require(not table.isna().any().any(), "Missing degradation penalty")
        values = table.to_numpy(float)
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=6.15, color="white" if luminance < 0.50 else "#202020",
                        fontweight="bold" if abs(value - 1.0) >= 0.25 else "normal")
        ax.set_title(CONTROLLER_LABEL[controller], pad=3.0, fontweight="bold")
        ax.set_xticks(np.arange(5), [f"C{i}" for i in range(1, 6)])
        ax.set_yticks(np.arange(3), [PATH_SHORT[p] for p in PATH_ORDER])
        if controller_idx == 0:
            panel_label(ax, "(c)", x=-0.06, y=1.19)
        else:
            ax.tick_params(labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#A0A0A0")

    require(image is not None, "Heatmap was not created")
    colorbar = fig.colorbar(image, cax=color_ax, ticks=[0.6, 1.0, 1.4, 1.8])
    colorbar.ax.tick_params(labelsize=6.1, length=2.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label("Fault / unit-health position RMSE", fontsize=6.6, labelpad=3)
    axes[1].text(0.5, 1.16,
                 "Trajectory-matched degradation penalty (>1 means higher error)",
                 transform=axes[1].transAxes, ha="center", va="bottom",
                 fontsize=7.4, fontweight="bold")


def draw_pair_panels(fig: plt.Figure, subspec, paired: pd.DataFrame) -> None:
    subgrid = subspec.subgridspec(1, 2, wspace=0.30)
    ax_att = fig.add_subplot(subgrid[0, 0])
    ax_tol = fig.add_subplot(subgrid[0, 1])
    path_offsets = {"circle_level": -0.15, "helix_unseen": 0.0, "figure8_unseen": 0.15}

    for path in PATH_ORDER:
        subset = paired.loc[paired["path"] == path].set_index("condition_id").loc[CONDITION_ORDER]
        x = np.arange(6, dtype=float) + path_offsets[path]
        ax_att.scatter(
            x, subset["tqc_dr_to_swim4track_attitude_rmse_ratio"],
            s=28, marker=PATH_MARKER[path], facecolor=PATH_COLOR[path],
            edgecolor="white", linewidth=0.55, zorder=3, label=PATH_SHORT[path],
        )
        ax_tol.scatter(
            x, subset["swim4track_minus_tqc_dr_tolerance_percentage_points"],
            s=28, marker=PATH_MARKER[path], facecolor=PATH_COLOR[path],
            edgecolor="white", linewidth=0.55, zorder=3, label=PATH_SHORT[path],
        )

    for ax in (ax_att, ax_tol):
        ax.set_xlim(-0.5, 5.5)
        ax.set_xticks(np.arange(6), [f"C{i}" for i in range(6)])
        ax.set_xlabel("Condition code")
        style_axis(ax)

    ax_att.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
    ax_att.set_ylim(0.80, 1.98)
    ax_att.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
    ax_att.set_ylabel("Attitude RMSE ratio\nTQC-DR / Swim4Track")
    ax_att.set_title("Cell-paired attitude error", pad=4.0, fontweight="bold")
    ax_att.text(0.98, 0.95, "Swim4Track lower in 17/18",
                transform=ax_att.transAxes, ha="right", va="top", fontsize=6.6,
                bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "pad": 2.0})
    panel_label(ax_att, "(d)", x=-0.04, y=1.14)

    ax_tol.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
    ax_tol.set_ylim(-5.0, 96.0)
    ax_tol.set_yticks([0, 20, 40, 60, 80])
    ax_tol.set_ylabel("Tolerance difference (pp)\nSwim4Track - TQC-DR")
    ax_tol.set_title("Cell-paired tolerance occupancy", pad=4.0, fontweight="bold")
    ax_tol.text(0.98, 0.95, "Swim4Track higher in 18/18",
                transform=ax_tol.transAxes, ha="right", va="top", fontsize=6.6,
                bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "pad": 2.0})
    panel_label(ax_tol, "(e)", x=-0.04, y=1.14)
    ax_tol.legend(loc="upper left", frameon=True, ncol=1, borderpad=0.35,
                  handletextpad=0.35, labelspacing=0.25, framealpha=0.92,
                  edgecolor="#D0D0D0")


def render_figure(trials: pd.DataFrame, cell: pd.DataFrame,
                  penalty: pd.DataFrame, paired: pd.DataFrame) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.20, 7.90))
    grid = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[0.88, 2.35, 1.90, 1.62],
        left=0.090, right=0.950, bottom=0.065, top=0.980,
        hspace=0.48,
    )
    ax_conditions = fig.add_subplot(grid[0, 0])
    draw_condition_strip(ax_conditions)
    draw_absolute_panel(fig, grid[1, 0], trials, cell)
    draw_penalty_panel(fig, grid[2, 0], penalty)
    draw_pair_panels(fig, grid[3, 0], paired)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / "robustness_landscape"
    fig.savefig(base.with_suffix(".pdf"), dpi=300, metadata={
        "Title": "Swim4Track actuator-effectiveness evidence landscape",
        "Subject": "Cell-resolved deterministic Stonefish campaign results",
    })
    fig.savefig(base.with_suffix(".svg"), dpi=300)
    fig.savefig(FIGURE_DIR / "robustness_landscape_300dpi.png", dpi=300)
    plt.close(fig)


def write_outputs(cell: pd.DataFrame, penalty: pd.DataFrame, paired: pd.DataFrame,
                  condition_definitions: pd.DataFrame, headline_table: pd.DataFrame,
                  audit: dict) -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    cell.to_csv(DERIVED_DIR / "cell_metrics.csv", index=False, float_format="%.12g")
    penalty.to_csv(DERIVED_DIR / "position_degradation_penalties.csv", index=False,
                   float_format="%.12g")
    paired.to_csv(DERIVED_DIR / "paired_swim4track_vs_tqc_dr.csv", index=False,
                  float_format="%.12g")
    condition_definitions.to_csv(DERIVED_DIR / "condition_definitions.csv", index=False,
                                 float_format="%.12g")
    headline_table.to_csv(DERIVED_DIR / "headline_checks.csv", index=False,
                          float_format="%.12g")
    (DERIVED_DIR / "numerical_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    means = audit["equal_weight_cell_means"]
    counts = audit["directional_cell_counts"]
    claims = audit["headline_values"]
    lines = [
        "Swim4Track robustness evidence landscape - numerical audit",
        "",
        f"Trials: {audit['design']['trial_count']} unique / {audit['design']['cell_count']} cells / "
        f"{audit['design']['executions_per_cell'][0]} executions per cell",
        "Repeat interpretation: deterministic numerical repeatability; not independent environmental samples.",
        "Configured water current in every trial: 0.0 m/s.",
        "",
        "Equal-weight means over 18 trajectory-condition cells:",
    ]
    for controller in CONTROLLER_ORDER:
        value = means[controller]
        lines.append(
            f"  {CONTROLLER_LABEL[controller]}: position RMSE={value['position_rmse_m']:.6f} m; "
            f"attitude RMSE={value['attitude_rmse_deg']:.6f} deg; "
            f"tolerance={value['tolerance_fraction']:.6f}"
        )
    lines.extend([
        "",
        "Direction counts across 18 fixed cells:",
        f"  Swim4Track position RMSE < PID: {counts['swim4track_position_lower_than_pid']}/18",
        f"  Swim4Track position RMSE < TQC-DR: {counts['swim4track_position_lower_than_tqc_dr']}/18",
        f"  Swim4Track attitude RMSE < TQC-DR: {counts['swim4track_attitude_lower_than_tqc_dr']}/18",
        f"  Swim4Track tolerance > TQC-DR: {counts['swim4track_tolerance_higher_than_tqc_dr']}/18",
        "",
        "Headline arithmetic:",
        f"  Position RMSE vs PID: {claims['position_vs_pid']:.4f}% lower",
        f"  Position RMSE vs TQC-DR: {claims['position_vs_tqc_dr']:.4f}% higher (TQC-DR denominator)",
        f"  Attitude RMSE vs TQC-DR: {claims['attitude_vs_tqc_dr']:.4f}% lower",
        f"  Tolerance vs TQC-DR: {claims['tolerance_vs_tqc_dr']:.4f} percentage points higher",
        f"  Unit-health mean absolute command vs TQC-DR: "
        f"{claims['unit_health_mean_abs_command_vs_tqc_dr']:.4f}% lower",
        f"  Unit-health variation rate vs TQC-DR: "
        f"{claims['unit_health_command_variation_rate_vs_tqc_dr']:.4f}% lower",
        f"  Swim4Track mean of per-run p99 controller-computation times: "
        f"{audit['timing']['swim4track_mean_trial_level_p99_ms']:.6f} ms",
        "",
        "Degradation penalty = fault-condition position-RMSE cell mean / trajectory-matched "
        "unit-health position-RMSE cell mean for the same controller.",
        "A ratio below 1 is descriptive and is not evidence that an actuator fault is beneficial.",
        "No result in this package establishes active fault diagnosis, reconfiguration, current robustness, "
        "or hardware transfer.",
    ])
    (DERIVED_DIR / "numerical_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest_path = ROOT / "MANIFEST_SHA256.txt"
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and "tmp" not in path.parts
    )
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate and regenerate derived tables/audits without rendering the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials, reference_cells, protocol = load_and_validate()
    cell, penalty, paired, condition_definitions, headline_table, audit = derive_outputs(
        trials, reference_cells, protocol
    )
    write_outputs(cell, penalty, paired, condition_definitions, headline_table, audit)
    if not args.check_only:
        render_figure(trials, cell, penalty, paired)
    write_manifest()
    print(json.dumps({
        "validated_trials": audit["design"]["trial_count"],
        "validated_cells": audit["design"]["cell_count"],
        "figure_rendered": not args.check_only,
        "directional_counts": audit["directional_cell_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()

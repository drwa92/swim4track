# Data dictionary

## Input tables

### `data/input/valid_trial_metrics.csv`

One row per valid trial (270 rows). The plot uses the identifiers
`controller_id`, `path`, `condition_id`, and `repetition`; the primary plotted
metrics are `position_rmse_m`, `attitude_rmse_deg`, and
`time_in_tolerance_fraction`. Fault-contract fields are `final_health`,
`affected_thrusters`, and `loss_ramp_s`.

`position_rmse_m` and `attitude_rmse_deg` use the full scored interval.
`time_in_tolerance_fraction` (tolerance occupancy) and the sustained-success
decision exclude the first 20% of the scored interval. This window distinction
is part of the metric definition, not missing data or post-hoc filtering.

### `data/input/cell_summaries.csv`

Frozen campaign-analysis output. The generator does not plot directly from
this file. It independently recomputes cell means from the trial table and
requires agreement below `1e-12` for position RMSE, attitude RMSE, and tolerance
occupancy.

### `data/input/stonefish_robustness_campaign_v2.json`

Frozen controller, trajectory, condition, repetition, and trial-count
contract. It also documents that all robustness conditions use zero configured
water current.

## Derived tables

### `data/derived/cell_metrics.csv`

One row per controller-trajectory-condition cell (54 rows).

- `n`: number of repeat executions, always 5.
- `*_mean`: arithmetic mean over the five executions.
- `*_repeat_sd`: sample standard deviation across the five deterministic
  executions; this is a numerical-repeatability descriptor, not population
  uncertainty.
- `sustained_success_count`: successful executions out of five, evaluated
  after excluding the first 20% of each scored interval.
- `terminal_success_count`: terminally successful executions out of five; this
  is distinct from the sustained-success window.

### `data/derived/position_degradation_penalties.csv`

One row per controller-trajectory-fault cell (45 rows). The
`degradation_penalty_ratio` is the fault position-RMSE cell mean divided by the
matched unit-health position-RMSE cell mean for the same controller and
trajectory.

### `data/derived/paired_swim4track_vs_tqc_dr.csv`

One row per trajectory-condition cell (18 rows). It contains absolute position
means, directional comparison flags, the attitude-error ratio used in panel
(d), and the tolerance percentage-point difference used in panel (e).

### `data/derived/condition_definitions.csv`

Plot labels and protocol-derived channel, health, and ramp definitions for C0
through C5.

### `data/derived/headline_checks.csv`

Exact arithmetic behind the rounded manuscript claims, including the stated
denominator and aggregation rule.

### `data/derived/numerical_audit.json` and `.txt`

Machine-readable and human-readable validation records. The JSON includes
input hashes, design checks, cross-check errors, equal-weight cell means,
direction counts, success counts, timing checks, and interpretation limits.

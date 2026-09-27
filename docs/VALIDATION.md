# Release validation — v0.2.0

Checked on 27 September 2026. These checks concern the cleaned repository;
they are not a rerun of the paper's experiments.

| Check | Observed result |
|---|---|
| Unit suite | **38 discovered: 37 passed, 1 skipped** |
| Numerical source | Nine plant/core/reference/actuator/safety/environment files unchanged from the previous packaging snapshot |
| Frozen model | SHA-256 and 2,879,572-byte size match the original nominal seed-17 artifact |
| Training recipe | Defaults compared directly with checkpoint JSON metadata; architecture and principal hyperparameters match |
| Python wheel | Built successfully, installed in a separate venv, imported from outside the repository |
| Installed model lookup | Located and verified model under the environment's `share/swim4track/models` |
| Installed plant rollout | PID circle: 400 control steps / 40 simulated seconds completed; state/command CSV, NPZ, summary and plots saved |
| Source distribution | Builds and includes checkpoint, manifest and training module |
| Training curves | Reader tests preserve missing values; PNG/PDF renderer exercised using a temporary synthetic test fixture, not supplied as research data |
| ROS Python source | Syntax/bytecode compilation passed |
| Shell scripts | Syntax, help and invalid-argument handling passed |
| Figure data | Original 270-row table and 54 controller/path/condition cells pass the supplied figure generator's checks |
| Source provenance | Recorded release hashes refreshed after the documented changes |

The tests exercise dynamics structure, reference derivatives, observation-frame
invariance, deterministic plant stepping, frozen PID identity, actuator limits,
checkpoint corruption rejection, rollout alignment and failed-run evidence,
stale-state handling, explicit start/stop, clock gaps, raw-action memory,
completion and per-run identity. The model verifier reports hash verification
separately from actual inference.

## Environment and limits

Linux; Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8.
The wheel was built with existing local build tools using `--no-build-isolation`
and installed with `--no-deps` into a separate `--system-site-packages` virtual
environment. This exercises packaging and installed paths. It does not test
fresh dependency resolution or the install scripts' network operations.

PyTorch, Gymnasium, SB3-Contrib and ROS 2 are unavailable in this preparation
environment. Consequently:

- The optional actor test skipped; the checkpoint was verified through its
  digest and ZIP metadata without deserialization or inference.
- The cleaned training loop has not been executed here. No new trained weights
  or measured learning curves are claimed.
- The refactored ROS node has not been run against Stonefish here. Its pure
  numerical/runtime components pass tests, but a target ROS smoke test is needed.
- The GitHub CI workflow defines core, learned-inference and short training-startup
  checks on Python 3.10/3.12; no remote CI execution is claimed.

The included actual Stonefish video comes from the user's earlier recording.
It does not prove that this refactored wrapper was used. The Python PID example
under `docs/examples/` is retained unchanged from v0.1.0 and remains labelled
with its original software version. Historical experiment records are separate.

## Re-run checks

After installation, from the repository root:

```bash
python -m unittest discover -s tests -v
python -m swim4track.policy
python -m swim4track.train --print-config
python -m compileall -q src ros2
for script in scripts/*.sh; do bash -n "$script"; done
python examples/paper_figure/generate_robustness_figure.py --check-only
sha256sum -c RELEASE_MANIFEST_SHA256.txt
```

With inference dependencies installed, explicitly exercise the real actor:

```bash
python -m swim4track.policy --load
python -m swim4track.demo --controller learned --trajectory circle_level --seconds 40 --output runs/validation_learned
```

Follow [stonefish.md](stonefish.md) for one recorded ROS trial. The original
model and research evidence need not be regenerated for these release checks.
After intentional source edits, differences from the release manifest are
expected; keep the original archive when making those changes.

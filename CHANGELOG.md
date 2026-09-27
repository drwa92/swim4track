# Changelog

## 0.2.0 — 2026-09-27

- Organize the release around Python simulation, nominal TQC training and frozen
  ROS inference, with separate installation guides and command examples.
- Include the real edited Stonefish demo and poster in the README.
- Use regular package installation by default; check virtual-environment and
  ROS prerequisites before building, without closing the caller's shell.
- Add lightweight checkpoint verification and improve installed-model discovery.
- Improve training observability and input validation while preserving the
  nominal recipe and original model weights.
- Strengthen runtime/trial monitoring and document the remaining target ROS
  smoke test explicitly.

Numerical source changes, if any, are recorded in `docs/SOURCE_PROVENANCE.json`.
The model checksum remains unchanged. These are release engineering changes;
they are not a new controller or new experimental evidence.

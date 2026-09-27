# Frozen model

`tqc_nominal_seed17_final.zip` is the unchanged nominal Swim4Track TQC
checkpoint used in the recorded Stonefish campaigns. Its paper controller
name is **S4T-TQC**; historical campaign files use **Swim4Track**.

```text
SHA-256 304f281a7843d0c0b2e2016ad470a8c923524a4e236f536e07764a51d7967f2f
Bytes   2879572
```

Verify the bundled artifact after installing the Python package:

```bash
python -m swim4track.policy
```

This check prints the file path, digest, size and repository package version.
It does not deserialize the checkpoint. With inference dependencies installed,
add `--load` to check deterministic actor prediction. The bundled file is also
included in built Python wheels; an installed package can find it without
requiring the current working directory to be the repository root.

Read [the model card](../docs/model_card.md) for the observation order,
command interpretation, training metadata and limitations.
[`model_manifest.json`](model_manifest.json) contains the machine-readable
record. Archive-recorded training versions are provenance, not an installation
lock for this release.

To use a newly trained policy, supply both its path and an independently
recorded digest:

```bash
python -m swim4track.policy --model /absolute/path/to/model.zip --model-sha256 YOUR_RECORDED_SHA256 --load
```

Replace `YOUR_RECORDED_SHA256` with the 64-character digest from that run's
artifact record. Its observation and action contract must remain compatible.
The release checkpoint is kept unchanged when new training runs are created.

The ZIP is an SB3 training checkpoint and contains serialized Python metadata
and PyTorch state. Load only a trusted checkpoint after verifying its hash;
a digest establishes file identity, not trust in an unknown source.

The supplied code declares Apache-2.0. A separate weight-redistribution license
was not present in the source evidence; the owner must establish those terms
before publishing the checkpoint. This repository does not infer a weight
license from the code license.

# Contributing

Use an issue to describe a reproducible bug or proposed behavior change.
Include the command, Python/ROS versions, checkpoint hash, trajectory,
configuration and relevant log excerpt. Avoid uploading credentials or
unrelated machine information.

For a code change, keep the Python core independent of ROS imports. Add a
focused test when changing the observation, reference, coordinate frames,
actuator mapping or controller behavior. Explain the resulting behavior
and report the checks you ran. A refactor must preserve the frozen model's
23-input/eight-output contract.

Do not replace the supplied checkpoint or silently overwrite paper evidence.
New models, controller settings and experiments need their own identifiers
and hashes. Keep PID, learned-policy and fault-adapter results distinguishable.
Label generated illustrations and synthetic examples accurately; simulator
video must come from an actual recorded run.

Contributions must be yours to submit and compatible with the applicable
license. Preserve third-party attribution and identify the source and usage
terms of new datasets, weights, images or meshes.

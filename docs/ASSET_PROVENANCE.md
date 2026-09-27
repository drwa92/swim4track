# Asset provenance

| File | Origin and interpretation |
|---|---|
| `docs/assets/Swim4Track_GitHub_demo.mp4` and `Swim4Track_video_poster.jpg` | Edited from the user-supplied real Stonefish screen recording dated 27 September 2026. 75 seconds, 1280×720, silent, normal 1× playback; source was 952×1052 and was cropped/upscaled. Controller identity was not independently recovered. Scene labels describe visible motion, not comparative controller performance. |
| `docs/assets/framework.png` | Unchanged export from the approved Swim4Track framework v4. Diagram labels were composed programmatically; the underwater ROV scene was generated with ChatGPT's OpenAI image tool on 26 September 2026. It is an explanatory illustration, not a simulator screenshot, photograph, measurement or exact mechanical drawing. |
| `docs/assets/robustness.png` | Unchanged `robustness_landscape_300dpi.png` export from the corrected Priority-2 analysis package. Computed from the supplied 270 valid trial metrics and frozen actuator-effectiveness protocol. |
| `models/tqc_nominal_seed17_final.zip` | Unchanged seed-17 nominal checkpoint from the supplied locked five-seed evidence archive. SHA-256 and archive metadata are recorded in `models/model_manifest.json`. |
| `docs/examples/pid_circle/` | Actual rollout of the supplied Python plant with the frozen PID comparator, rendered as a plot/video replay. `summary.json`, state/command tables and `rollout.npz` accompany it. It is not Stonefish footage. |

The framework preserves the separation between source-simulator training
and frozen Stonefish inference. Its ROV rendering must not appear in a
demo thumbnail or video in a way that implies it is recorded simulator
footage. The generating tool did not expose an exact backend model/version;
none is asserted here.

The robustness figure retains historical controller labels. “Swim4Track”
means S4T-TQC and “TQC-DR” means S4T-TQC-DR; both learned variants use the
proposed representation. Its five executions per cell describe repeat
spread in a fixed campaign. They are not five independently sampled
environments. All six conditions have zero configured current.

The minimal original plotting pipeline is included under
[`examples/paper_figure/`](../examples/paper_figure/README.md): the unchanged
generator, trial metrics, cell-summary cross-check and frozen protocol.
It reproduces the plot and numerical audit without bags. This is metric-table
reproduction, not raw-bag rescoring or reexecution of the campaign. Full
rosbag recordings are not included.

The inherited Apache-2.0 code declaration does not establish usage terms
for these separate media or model artifacts. The rights holder should
confirm their redistribution terms before public release. Preserve this
provenance when adding those terms. Record future real demo footage with
its controller/configuration, run identifier, capture date, playback speed
and corresponding bag/metadata location.

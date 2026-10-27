# Local underwater orientation evaluation

Evaluated 2026-10-27, using RightWayUp from ORTUS AI. Both files were downloaded
directly from the pinned Hugging Face revision
`69076d99f16fafbc3d7e12958390972b5f291e81`, with upstream SHA-256 verification.
Code and weights are Apache-2.0. The application uses Max INT8; Fast FP32 was
evaluated as a smaller alternative. Model binaries and camera frames remain
local and are not republished.

## Dataset and direction convention

Ten samples per clip, at 5%, 15%, …, 95% of duration, extracted as 640-pixel
previews before grading. Each was also tested after synthetic clockwise
rotations of 90°, 180°, and 270°. This gives 40 original samples / 160 views
per model, grouped into 16 clip/control cases.

Ground truth was checked using upright visual previews of C0201, C0202, C0212,
and C0213. Clockwise correction: C0201 = 0°, C0202 = 270° (90° counterclockwise),
C0212/C0213 = 90°. An initial description had the clockwise labels reversed;
the evaluation script and the results here use the visually verified labels.

These four related underwater clips are a development set, not an independent
test of underwater performance. Rotated variants are correlated controls,
not additional independent videos.

## Decisions

The first experiment required eight samples to exceed the upstream *strict*
confidence threshold, which measures probability within about ±10° of a
continuous angle. The dominant direction was correct on all four original
clips, but this precise-angle threshold abstained on most sideways clips.

We then defined a separate **quarter-turn** decision before running its
evaluation: integrate the 360-bin output into the four nearest right-angle
sectors; require at least eight of ten samples to assign >=0.9 probability to
the same correction, and require the mean probability of that correction over
all ten samples to be >=0.9. This task-specific rule was developed on these
clips. These probabilities are not calibrated underwater error rates.

| Max INT8 original clip | Correct direction | Strong agreeing votes | Mean sector probability | Application result |
| --- | --- | ---: | ---: | --- |
| C0201 | 0° | 10/10 | 0.9980 | Keep upright |
| C0202 | 90° counterclockwise | 9/10 | 0.8969 | Uncertain; leave unchanged and suggest manual correction |
| C0212 | 90° clockwise | 10/10 | 0.9844 | Rotate clockwise |
| C0213 | 90° clockwise | 9/10 | 0.9662 | Rotate clockwise |

Across the 16 clip/control cases, Max INT8 accepted 14 and Fast FP32 accepted
9. Neither produced an incorrect accepted decision. Max therefore became the
optional application model. We retained the threshold when C0202 narrowly
missed it; a correct majority alone does not automatically rotate a clip.

Max's ten-frame CPU inference took about 3.1 seconds per case with ONNX Runtime
1.31.0 and four threads, excluding frame extraction and session initialization.
Production detection reproduced the table above on the four original files.
No claim of general accuracy, especially on featureless water or moving camera
orientation, follows from this small experiment.

## Reproduce

```bash
python3 tools/prepare_orientation_model.py --models-dir models --tier both
.venv-models/bin/python -m benchmarks.orientation \
  --directory /path/to/original/videos --tier both --controls --decision quarter
```

The downloader's `--models-dir` is the root model directory; the benchmark
defaults to the resulting `models/rightwayup` subdirectory. Use `--decision
strict` to reproduce the continuous-angle abstention experiment. PNG samples
and full per-frame probability reports are written under
`benchmarks/results/orientation/`, ignored by Git.

Upstream: [model card](https://huggingface.co/ortusai/rightwayup),
[implementation](https://github.com/ortusaitech/rightwayup),
[project and limitations](https://cheqit.ortusai.io/resources/rightwayup/).

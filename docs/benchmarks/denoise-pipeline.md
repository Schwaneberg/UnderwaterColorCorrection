# Overlapping CPU preparation and neural inference

Measured on 2026-10-27 with the RTX 3060 Ti, NVIDIA driver 580.178.04,
ONNX Runtime GPU 1.31.0, and the validated mixed FP16 FastDVDnet graph.
Input: the first 12 frames of C0201, 1920×1080. The clip is interpreted as
Rec.709 for this performance test; its unconfirmed color signaling makes this
unsuitable as a color-accuracy assessment.

Previously the denoiser decoded/transformed each input, packed its five-frame
tensor, ran the model, and converted the output sequentially. Color workers
were already separate, but input preparation could not overlap inference.

The pipeline now has one preparation worker. It reads and converts the next
input frame, assembles the next temporal window, and packs the next full-frame
GPU tensor while the caller runs inference on the current window. At most one
future window is prepared. Model calls remain sequential, in source order.
Tiled inference also overlaps decoding/transfer conversion of the next frame;
its per-tile tensor packing remains sequential.

Full-frame packing writes directly into contiguous NCHW storage, avoiding
unnecessary full-frame copies. Cancellation/early close first interrupts the
decoder, then joins the preparation worker before closing the pipe. CPU
fallback retains the same source windows even if a GPU tensor was prefetched.

## Measurement

Two full-resolution predictions warmed the model. Serial and prefetched runs
then alternated order across three repeats, using the same model session and
the same improved tensor-packing code. Timings include FFmpeg decoding, transfer
conversion, tensor assembly, inference, output conversion, and consumption of
each output with SHA-256. They exclude color grading, encoding, and audio muxing.

| Mode | Repeat 1 | Repeat 2 | Repeat 3 | Median per frame |
| --- | ---: | ---: | ---: | ---: |
| Serial | 5.583 s | 5.382 s | 5.343 s | 0.4485 s |
| One-window prefetch | 4.479 s | 4.488 s | 4.499 s | 0.3740 s |

Median throughput improved **1.199×**, or about **16.6% less elapsed time**.
All 12 output checksums were identical across all six runs. This isolates the
overlap benefit; it does not quantify the additional tensor-copy reduction
against the previous application version. It is a short warm benchmark, not a
completed full-video export or a prediction for every GPU/resolution.

## Reproduce

```bash
.venv-gpu/bin/python -m benchmarks.denoise_pipeline \
  --clip tests/clips/C0201.MP4 --frames 12 --repeats 3 \
  --report benchmarks/results/denoise-pipeline/c0201.json
```

The command requires an active CUDA provider unless `--allow-cpu` is specified.
Without `--clip`, it uses deterministic synthetic pixels at 1920×1080 (dimensions
can be overridden). JSON measurements are generated artifacts, ignored by Git.

Prefetch is enabled by default. Set `UWCC_DENOISE_PREFETCH=0` before launching
UWCC to compare with serial denoising or reduce preparation memory. Full-frame
HD prefetch adds a roughly 119 MiB float32 input tensor, a noise map, and the
next source frame, plus temporary conversion buffers; it is separate from the
GPU arena and color-worker memory budgets. CPU inference may gain less or
contend for CPU/memory bandwidth. No GPU-session calls run concurrently.

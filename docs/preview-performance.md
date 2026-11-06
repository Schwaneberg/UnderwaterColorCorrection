# Preview playback performance

Measured on the local Intel Core i5-8250U with Python 3.14, at 640×360 proxy
resolution. These synthetic measurements isolate costs; they do not guarantee
the same rate for camera codecs, HDR/log material, or GUI painting.

| CPU stage | Compiled kernels | NumPy fallback |
| --- | ---: | ---: |
| Previous Rec.709 original-view conversion | 44.4 ms | 69.2 ms |
| Direct Rec.709 original view | <0.01 ms | <0.01 ms |
| Analyzed color LUT | 22.9 ms | 135.1 ms |
| Analyzed LUT plus temperature adjustment | 105.2 ms | 235.8 ms |
| One display image conversion | 2.3 ms | 2.4 ms |

A 30 fps presentation has 33.3 ms per frame. Removing the redundant
Rec.709 transfer/matrix/gamut round trip leaves about 27.5 ms for the measured
correction and two display image conversions with compiled kernels.
Log/HDR previews retain their display transform. Exports use the existing graph.

For a four-second, 1280×720 H.264 test source at 30 fps, the previous processing
path needed 9.20 seconds for 120 frames even without pacing. Updated playback
presented 113 frames through timestamp 3.967 in 4.20 seconds, including startup,
decoding, timestamp extraction, correction, and image conversion. This benchmark
does not paint the widgets or play audio. Late frames now skip correction to
avoid accumulating delay against the playback/audio clock.

The source checkout initially lacked `uwcc._native`. Build it with
`python setup.py build_ext --inplace`; packaged Windows releases already include
it. Compilation materially helps LUT processing but does not eliminate the cost
of manual controls. Those controls, log/HDR conversion, and full-resolution
source decoding remain candidates for further profiling with an affected clip.

Reproduce component measurements with:

```powershell
.venv-build\Scripts\python.exe -m benchmarks.preview --repeats 20
```

Add `--clip path/to/rec709.mp4` for a decode/playback comparison using a synthetic
automatic recipe. It measures the whole clip and interprets it as Rec.709.
The command prints JSON with compiled and fallback stage timings.

## Tunnel.mp4 follow-up

The user's clip is 1920×1080 Rec.709 at 50 fps (65.52 seconds long). An
eight-second sample needs 400 frames to retain the original cadence. Decoding
alone took 2.06 seconds; the previous original-plus-correction path took
31.85 seconds without pacing. Tests here use the same representative automatic
recipe, rather than the user's saved grade or a new neural analysis.

Playback now divides pointwise correction across up to four persistent CPU
workers. Dehaze and sharpness use the full frame to preserve spatial filtering.
Preview decoder and timestamp-reader pools are capped at four threads to limit
competition with correction on this four-core CPU. Export thread settings are
unchanged. Parallel color results are checked against the full-frame graph,
including log/HDR and keyframed manual adjustments.

Runs presented about 30–40 fps while keeping the playback clock moving. A final
offscreen Qt window test, including viewer/scopes painting and control updates,
displayed 292 frames and reached video timestamp 8.0 in 8.34 seconds. Audio was
disabled for that test. This is substantially faster, but still falls short of
displaying every frame of the 50 fps source; expensive manual/spatial controls
can reduce throughput further. Visible-window and audible-sync validation
remain for manual testing. Results vary with CPU load and thermal limits.

For a bounded real-clip benchmark, use:

```powershell
.venv-build\Scripts\python.exe -m benchmarks.preview --clip C:\Users\olive\Videos\Tunnel.mp4 --seconds 8
```

Add `--skip-legacy` to omit the slow previous-path clip comparison.

Validation: playback tests cover normal cadence, skipping late frames, direct
Rec.709 originals, and unchanged log/HDR display transforms. Color and native
kernel parity tests pass. The wider media checks still encounter the documented
Windows variable-frame-rate export failures in `tests/test_media.py`.

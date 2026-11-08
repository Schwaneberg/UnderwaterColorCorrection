# C0045 raw-frame copy and pipe optimization

Measured on the same Windows/Ryzen 7 5800X system as the
[fusion/SIMD experiments](C0045-fusion-experiment.md), using production AVX2,
four color workers, stabilized 1080p/50 fps footage, and no denoise.
Source media, experiment drivers, and results remain local in Git-ignored
`benchmarks/results/C0045-export/`.

## Experiments

The original reader received a temporary `bytes` object, copied it into a
`bytearray`, and converted that into float32. Windows' default anonymous pipe
capacity was queried with `GetNamedPipeInfo`: **4,096 bytes**, compared with
12,441,600 bytes in one RGB48 frame.

The alternatives filled a reusable uint16 buffer using `readinto` and/or
increased the pipe capacity. Each normalized float frame still owns independent
storage. The first 200 decoded frames were warped and graded with the same plan;
output was discarded, so these timings exclude motion analysis and encoding.

| Reader / pipe capacity | Time | Throughput |
| --- | ---: | ---: |
| Original reader / default 4 KiB | 6.20 s | 32.2 fps |
| `readinto` / default | 5.34 s | 37.4 fps |
| Original reader / 64 KiB | 5.09 s | 39.3 fps |
| Original reader / 1 MiB | 4.95 s | 40.4 fps |
| `readinto` / 1 MiB | **4.17 s** | **47.9 fps** |
| `readinto` / 4 MiB | 4.34 s | 46.1 fps |

Twelve full-frame comparisons across the alternatives matched exactly. A matched
full stabilized export pair, including motion analysis/muxing/validation, took
**63.47 seconds originally versus 50.56 seconds** with `readinto` and 1 MiB
pipes (20% less time). The complete MP4 files were byte-identical. These are
single-run measurements, not a statistical throughput guarantee; 4 MiB showed
no advantage over 1 MiB in this test.

## Production integration and verification

`FrameReader._decoded` now fills a reusable uint16 scratch buffer directly,
handles partial reads, and yields separately allocated float32 frames with the
same division by 65535. It does not reuse float buffers still held by color
workers or temporal denoise. EOF, truncated-frame, decoder-error, and cancellation
behavior are tested.

On Windows, decoder stdout and encoder stdin use dedicated pipes sized to the
frame, capped at 1 MiB (minimum 4 KiB). The implementation calls Win32 `CreatePipe`
and passes only the child endpoint through ordinary `Popen` handle duplication.
It does not patch subprocess globals or CPython private APIs. Parent endpoints
are non-inheritable; child endpoints are closed in the parent after startup,
and both endpoints are cleaned up on startup failure. Other platforms keep
standard subprocess pipes; `readinto` is portable.

The actual production export took **48.24 seconds** with all 1,032 frames,
20.64 seconds, and audio preserved. This confirmation run is separate from the
matched experimental pair. The production and reference MP4 files shared SHA-256:

```text
373408c54d84f05b343d135f0c406095b4ec8f0764449bd16066a058b6ead226
```

Full Windows suite: **302 passed, 15 skipped**. New coverage includes transfers
larger than the pipe, EOF, capped Windows capacity, startup failure/interrupt
cleanup, portable subprocess behavior, short reads, frame ownership, truncated
frames, decoder errors, and cancellation during a partial frame. Linux native
execution remains for CI; no local Linux environment is installed.

The existing fusion/SIMD experiments have been kept. The raw-copy driver also
retains its legacy reader and pipe factory so its control remains available
after production integration. Re-running benchmarks on a changed application
or machine may produce different timings.

Artifacts under `benchmarks/results/C0045-export/`:

- `raw_copies_experiment.py`, `raw_copies_export.py`: experiment drivers.
- `raw-copies/pipe-capacity.json`, `raw-copies/stage-report.json`,
  `raw-copies/parity.json`, `raw-copies/export-report.json`: measurements.
- `raw-current.mp4`, `raw-optimized.mp4`: matched experimental outputs.
- `production_raw.py`, `production-raw-report.json`, `production-raw.mp4`:
  actual production-path confirmation.

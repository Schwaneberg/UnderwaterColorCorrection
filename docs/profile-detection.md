# Automatic input color profile detection

Opening a video detects its supported transfer curve and color primaries before
automatic color analysis. No model download or extra dependency is needed.
The GUI displays the evidence; `uwcc probe video.MP4` includes `profile` and
`profile_evidence`. CLI `analyze`, `batch`, and `learn-reference` default to
`--profile auto`. Batch detection runs independently for each source.

An explicit profile selection or a preset containing one overrides detection.
Saved concrete project profiles remain unchanged. Reference learning still
requires Rec.709. Unknown input requires a manual choice; CLI analysis fails
with guidance and batch records a failed item rather than guessing.

## Detection order

1. Embedded Sony `NonRealTimeMeta` capture declarations take precedence over
   generic video stream tags. Both `CaptureGammaEquation` and
   `CaptureColorPrimaries` must match a supported pair:

   | Gamma | Primaries | UWCC input |
   | --- | --- | --- |
   | `rec709`, `rec709-xvycc` | `rec709` | Rec.709 |
   | `s-log2` | `s-gamut` | S-Log2 / S-Gamut |
   | `s-log3`, `s-log3-cine` | `s-gamut3-cine` | S-Log3 / S-Gamut3.Cine |
   | `rec2100-hlg` | `rec2020` | Rec.2020 HLG |
   | `rec2100-pq` | `rec2020` | Rec.2020 PQ |

   Unsupported, incomplete, malformed, or conflicting capture declarations
   require manual selection. S-Cinetone and plain S-Gamut3 are not silently
   treated as Rec.709 or S-Gamut3.Cine.
2. Without capture declarations, use FFprobe stream tags: `bt709` or
   `iec61966-2-4` transfer with `bt709` primaries selects Rec.709;
   `arib-std-b67` or `smpte2084` with `bt2020` primaries selects HLG or PQ.
   The YUV matrix tag alone cannot identify the capture profile.
3. Otherwise, ask for a supported input profile.

The MP4 reader seeks over encoded video data and reads at most 2 MiB of metadata
payloads with depth and box-count limits. It supports embedded metadata, not
external XML sidecars or a complete Sony real-time metadata parser.

## Checked camera files and limits

Read-only checks of C0201, C0202, C0212, and C0213 from the supplied
Jupiter / Emerald Charter directory found `iec61966-2-4` transfer, `bt709`
primaries, and `bt709` matrix. Their embedded ILCE-6500 XML does not declare
capture gamma or gamut. They now default to Rec.709 based on those stream tags.
This is a declaration-based fallback, not proof of the camera picture-profile
setting. Override it if you know the camera recorded log or custom gamma.

FFmpeg defines the positive-range xvYCC transfer using the same coefficients as
[BT.709](https://www.ffmpeg.org/doxygen/trunk/libavfilter_2vf__colorspace_8c.html).
This permits a Rec.709-compatible fallback within UWCC's existing nonnegative
RGB processing; it does not add full extended-gamut xvYCC preservation.

Sony's newer capture XML can explicitly identify log curves; a
[published original FX3A metadata dump](https://gist.github.com/MarwanRefaat/5f5be706a542e2c195c39c66d41352e4)
contains `s-log3-cine` / `s-gamut3-cine` alongside a Rec.709 coding matrix.
That format is covered with synthetic metadata fixtures; no original newer
camera log video was available for end-to-end validation.

Camera model, filename, histogram, and underwater color cast cannot reliably
recover a missing recording profile. Picture-profile presets can also be
customized. Detection therefore reports metadata evidence and always allows
an explicit override.

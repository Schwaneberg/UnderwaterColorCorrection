# UWCC Product Specification

Status: v1 requirements with current implementation coverage

Implementation baseline: 0.4.3

Last reviewed: 2026-11-08

This document defines product intent. **Shall** identifies a v1 requirement,
not a claim that the current release implements it. Section 11 records current
coverage and remaining work; the [technical specification](technical-spec.md)
describes the code and actual persistence formats. Release history belongs in
the [changelog](../CHANGELOG.md).

## 1. Purpose

UWCC shall make repetitive underwater video correction substantially simpler
than a general-purpose nonlinear editor. A user shall be able to open a clip,
obtain a useful natural-looking correction with one action, refine it with a
small control set, and render that correction or reuse it across a batch.

The application is intended for footage in which water has attenuated red light
and shifted the image toward blue or green. It must also behave sensibly when a
red or magenta capture filter has already compensated part of that loss. The
application cannot recreate scene information that was never recorded; it
shall disclose that limitation rather than disguising noise or invented detail
as recovered color.

## 2. Users and primary outcomes

### 2.1 Primary user

The primary user is a diver or underwater videographer who:

- records 8-bit Sony XAVC S footage, commonly on an Alpha 6300 or 6500;
- may use no filter, a red filter, or a magenta filter;
- wants consistent results without learning node-based grading;
- wants to tune one clip and reuse the same intent on related clips; and
- may still finish or edit the rendered master in another application.

No professional color-grading knowledge shall be assumed.

### 2.2 Success outcomes

A successful v1 shall let the user:

1. open and correctly interpret a supported video;
2. generate an adaptive correction locally;
3. understand and modify the result without exposing implementation details;
4. compare corrected and original images at arbitrary points in the clip;
5. animate adjustments when automatic adaptation is insufficient;
6. save and reuse settings safely; and
7. export a technically valid master or delivery file without losing audio,
   timing, orientation, or declared color characteristics.

## 3. Scope

### 3.1 Included in v1

- Video input and output only.
- A single active clip in the correction workspace.
- Background analysis of the full clip.
- Automatic correction that adapts within a shot and resets at cuts.
- Manual keyframes for correction strength and finishing controls.
- Presets and a multi-file batch queue.
- SDR Rec.709, Sony S-Log2/S-Gamut, Sony S-Log3/S-Gamut3.Cine,
  Rec.2020 HLG, and Rec.2020 PQ input interpretation.
- SDR Rec.709, HLG, and PQ output transforms.
- Software encoding on every supported installation and optional detected
  hardware encoding.
- Ubuntu 26.04 packages and Windows standalone/Microsoft Store MSIX packaging.
- Optional local UDnet reference enhancement, FastDVDnet temporal denoise,
  and RightWayUp orientation detection, with explicit model/runtime setup.
- Export-only background stabilization and conventional spatial/temporal denoise.
- Learning a clip-specific grade from an aligned corrected SDR reference.
- Optional, explicitly requested cloud grading advice with reviewed proposals;
  local correction and export shall remain usable without this feature.

### 3.2 Explicit non-goals for v1

- Still-image editing.
- Cutting, ripple editing, transitions, titles, compositing, or a multi-clip
  sequence timeline.
- Audio editing, mixing or effects.
- Object masks or subject tracking.
- RAW video development.
- Training a model inside the application.
- Cloud rendering, account synchronization, social sharing, telemetry, or
  automatic network downloads.
- Fabricating objects, textures, or channel detail with generative inference.

## 4. Core concepts

### 4.1 Input interpretation

The input interpretation describes how encoded values become scene-linear
working values. It consists of an input color profile, nominal range, and
capture-filter hint.

The profile choices shall be:

- `Auto from metadata`
- `Rec.709 / standard gamma`
- `Sony S-Log2 / S-Gamut`
- `Sony S-Log3 / S-Gamut3.Cine`
- `Rec.2020 HLG`
- `Rec.2020 PQ`

Metadata may select a profile only when it is unambiguous. Sony picture-profile
metadata is frequently incomplete; in that case the UI shall show
`Profile confirmation required`, retain the best non-destructive preview, and
require a user selection before final export.

The capture-filter choices shall be `Auto`, `No filter`, `Red filter`, and
`Magenta filter`. Automatic detection shall show its selected value and
confidence and shall always remain user-overridable.

### 4.2 Automatic correction

Automatic correction is a time-varying, shot-aware grade generated locally by
adaptive levels/white balance, optional neural reference inference, or legacy
balance. A verified model shall be optional; the inspector shall identify the
actual engine and any fallback. It shall target a conservative natural result:

- neutral objects should move toward neutrality when supported by the signal;
- color separation and contrast should improve without a mandatory vivid look;
- recovered red shall be bounded by measured red-channel signal quality;
- already-correct or filtered footage shall not be forced through an excessive
  red boost; and
- output shall remain inside the selected output gamut unless the user enables
  an explicit gamut-warning diagnostic.

The user-visible `Correction strength` shall blend between an identity transform
at 0% and the generated transform at 100%. The control range shall be 0–200%;
values above 100% are intentionally stronger and shall not change the generated
analysis data.

### 4.3 Manual correction

Manual controls are finishing operations applied after automatic correction.
They shall remain usable when automatic correction is bypassed.

| Control | Range | Neutral value | Meaning |
| --- | ---: | ---: | --- |
| Correction strength | 0–200% | 100% | Blend generated correction with identity |
| Temperature | -100–100 | 0 | Perceptual blue-to-amber balance |
| Tint | -100–100 | 0 | Perceptual green-to-magenta balance |
| Exposure | -4.0–4.0 EV | 0 EV | Scene-linear exposure offset |
| Contrast | -100–100 | 0 | Midtone-centered contrast |
| Highlights | -100–100 | 0 | Highlight-region tone adjustment |
| Shadows | -100–100 | 0 | Shadow-region tone adjustment |
| Black point | -100–100 | 0 | Toe and minimum-density adjustment |
| Saturation | 0–200% | 100% | Perceptual chroma scale |
| Hue | -180–180 degrees | 0 degrees | Global hue rotation |
| Dehaze | -100–100 | 0 | Bounded local-contrast/backscatter adjustment |
| Sharpness | 0–100 | 0 | Edge-limited output sharpening |

Every control shall have a numeric entry, slider, individual reset action, and
keyboard-accessible fine adjustment. Double-clicking a slider shall reset only
that control. Resetting the grade shall require confirmation only when there are
unsaved changes.

### 4.4 Time and keyframes

Automatic correction and manual keyframes are separate layers. Re-analysis may
replace the automatic layer but shall never alter manual keyframes.

- Every manual control, including correction strength, may be keyframed.
- A keyframe stores the displayed control value at a source timestamp.
- Values between keyframes shall use linear interpolation in the control's
  perceptual parameter domain.
- Values before the first and after the last keyframe shall hold the nearest
  keyframe value.
- Adding a keyframe shall capture the currently evaluated value, avoiding a
  visible jump.
- Deleting a keyframe shall reveal the interpolation between its neighbors.
- Undo and redo shall include all keyframe and parameter operations.

## 5. User experience

### 5.1 Application states

The application shall expose the following high-level states:

1. **Welcome** — open a video, open a recent project, or open the batch queue.
2. **Probing** — read media characteristics without decoding the full clip.
3. **Profile confirmation** — resolve an ambiguous input profile if necessary.
4. **Ready, not analyzed** — show a color-managed source preview and allow
   manual correction.
5. **Analyzing** — progressively produce the automatic curve while editing and
   playback remain available.
6. **Ready, analyzed** — show the complete generated curve and confidence.
7. **Rendering** — render in the background with progress and cancellation.
8. **Error** — preserve the project and provide an actionable recovery path.

Opening another clip with unsaved changes shall offer Save, Discard, and Cancel.
There shall be no modal prompt merely because analysis is still running;
analysis can be canceled and its completed results retained as partial data.

### 5.2 Workspace layout

The default workspace shall contain:

- a large viewer in the center;
- transport and time display below the viewer;
- a compact timeline below the transport;
- correction controls in a right-hand inspector; and
- Open, Save, Analyze, Batch, and Export actions in a stable top-level toolbar.

The viewer shall provide corrected, original, vertical split, horizontal split,
and momentary bypass modes. Split position shall be draggable. Zoom choices
shall include Fit, 50%, 100%, 200%, and a free zoom; panning shall work at zoom
levels larger than the viewer.

Playback shall prioritize audiovisual continuity. Late frames may be dropped
before correction rather than accumulating delay. Every displayed corrected
frame shall use the selected grade, and playback compromises shall not alter
export quality. Automatic proxy-resolution adaptation remains a target; the
current playback and paused previews use 640- and up-to-960-pixel proxies.

### 5.3 Scopes

Histogram, waveform, RGB parade, and vectorscope shall be available in a
collapsible scopes panel. Scopes shall reflect the displayed, output-referred
image after correction. The UI shall label scopes `Preview resolution` when
computed from proxy frames.

### 5.4 Analysis feedback

Analysis shall report overall progress and the current stage: scene detection,
sampling, ML inference, transform fitting, or temporal stabilization. When
complete, it shall show:

- model name and version;
- capture-filter decision and confidence;
- a low/medium/high correction confidence summary;
- scene boundaries on the timeline; and
- warnings for clipped channels, low red-channel signal-to-noise ratio,
  ambiguous profiles, or out-of-distribution input.

A low-confidence result shall remain previewable but shall not be presented as
authoritative. The user can accept it, reduce strength, switch to deterministic
fallback, or continue manually.

### 5.5 Project persistence

`Save Project` shall create a human-readable `.uwcc-project.json` sidecar.
Current projects store generated analysis inline; an external analysis cache
is not required. Source video is never copied into the project.

The application shall autosave recoverable editing state after an edit settles
for two seconds and at least once every 30 seconds while dirty. Recovery data
shall live in the user data directory, never beside the video unless the user
explicitly saves there. Successfully saving or deliberately discarding a
project shall remove its recovery entry.

If the source has moved, the application shall request a replacement and verify
its fingerprint before relinking. A fingerprint mismatch requires explicit user
confirmation and invalidates automatic analysis while preserving manual edits.

### 5.6 Presets

A preset shall store:

- input-profile and capture-filter policy (`Auto` or a fixed choice);
- automatic engine settings and default strength;
- all manual control values; and
- output color intent.

A preset shall not store source paths, source fingerprints, generated automatic
curves, model caches, output paths, or credentials.

Applying a preset to an open clip shall be one undoable operation. The UI shall
distinguish built-in read-only presets from user presets. v1 shall include
`Natural`, `Neutralize only`, `Red-filter natural`, and `Vivid` presets, with
`Natural` as the default.

### 5.7 Batch processing

The batch queue shall accept files and folders through a picker or drag and
drop. Each queue item shall contain source, selected preset, input override,
output profile, destination, and current state.

By default, each item shall be independently analyzed and then receive the
preset's manual offsets. An advanced `Use exact current grade` action may copy
the current automatic curve only to another source with the same duration,
dimensions, frame rate, and verified fingerprint; otherwise it shall be
disabled.

Queue states shall be `Waiting`, `Analyzing`, `Rendering`, `Paused`, `Complete`,
`Canceled`, and `Failed`. The user may reorder waiting items, pause after the
current frame group, cancel an item, retry a failure, and reveal an output.
Completed entries shall persist across restarts until cleared. Partial output
shall use a `.partial` suffix and shall never replace an existing valid output.

## 6. Export behavior

### 6.1 Export profiles

v1 shall provide these named profiles:

| Profile | Container and video | Audio | Intended use |
| --- | --- | --- | --- |
| DNxHR HQX master | MOV, 10-bit DNxHR HQX | PCM 24-bit | Editing interchange |
| ProRes 422 HQ master | MOV, 10-bit ProRes 422 HQ | PCM 24-bit | Editing interchange |
| FFV1 archival master | Matroska, FFV1 | FLAC | Lossless archive |
| H.264 SDR delivery | MP4, libx264, 8-bit 4:2:0 | AAC | Broad compatibility |
| HEVC SDR delivery | MP4, libx265 Main10 | AAC | Compact 10-bit SDR |
| HEVC HDR delivery | MP4, libx265 Main10 | AAC | Rec.2020 HLG or PQ |

Software encoders are normative. Optional hardware profiles shall be visually
marked as hardware-dependent and shall never replace the software profiles.

### 6.2 Defaults and preservation

- Log input shall default to an SDR Rec.709 output; HDR output requires an
  explicit selection.
- Output intent shall be explicit. The current default is SDR Rec.709 for all
  inputs; preserving HLG or PQ requires selecting that HDR intent and a compatible
  HEVC or FFV1 profile. Preview remains SDR regardless of export intent.
- Resolution, frame rate, scan type, display aspect, pixel aspect, and rotation
  shall default to source values.
- The entire source duration shall be exported; v1 has no trim semantics.
- A compatible source audio stream may be copied for delivery output. Masters
  shall decode once and encode to the profile's lossless audio format.
- Creation time, camera make/model, lens metadata, and location metadata shall
  be preserved when representable and shall be removable with a `Strip private
  metadata` option enabled by default for location only.
- Unknown data streams shall be reported and omitted, never silently claimed as
  preserved.

Before rendering, UWCC shall validate destination writability, free-space
estimate, encoder availability, unresolved input profile, output overwrite, and
HDR/profile compatibility. Existing files shall not be overwritten without an
explicit per-job decision.

## 7. Errors and recovery

Errors shall name the affected source or output, describe what failed in plain
language, preserve actionable technical detail behind a disclosure, and offer
only valid actions.

Required cases include:

- unsupported or corrupt source;
- unavailable decoder or encoder;
- ambiguous or inconsistent color metadata;
- source modified, moved, or disconnected;
- model missing, corrupt, incompatible, or license-unverified;
- GPU provider failure with automatic CPU retry;
- analysis canceled or partially complete;
- insufficient memory, storage, or file permissions;
- output collision; and
- application interruption during rendering.

On GPU inference failure, the current operation shall retry once with the CPU
provider and record the change. On export failure, `.partial` data shall remain
only when it is useful for diagnosis or recovery and the UI shall offer to
delete it.

## 8. Privacy, accessibility, and localization

- Ordinary media processing, thumbnails, analysis, projects, and logs shall
  remain local. Optional cloud advice is an explicit exception: it sends
  downscaled proxy frames to the selected provider after user initiation.
- Local correction, playback, and export shall make no network request.
  Model/runtime setup shall download only after explicit user action; merely
  offering setup at startup shall not start a download.
- Cloud advice shall require user-provided credentials and review of the
  proposed changes. Credentials shall not enter projects or presets. There
  shall be no UWCC account requirement or telemetry.
- Logs shall exclude decoded image content, precise location, and full source
  paths unless the user explicitly generates a diagnostic bundle.
- Every operation shall be keyboard accessible and have a visible focus state.
- Text and controls shall remain usable at 200% UI scaling.
- Meaning shall not rely on color alone; warning colors require icons and text.
- The initial language shall be English, but UI strings shall be externalized.
- Time values shall display as source timecode when present and as
  `HH:MM:SS:frames` otherwise.

## 9. Product acceptance criteria

v1 is acceptable only when:

1. the complete primary workflow succeeds on the supported Sony test corpus;
2. auto correction improves the defined reference metrics without violating
   clipping or temporal-stability thresholds in the technical specification;
3. every control works both manually and with keyframes;
4. projects, recovery data, presets, and batch state survive restart;
5. all normative export profiles round-trip with correct timing, audio, and
   color signaling;
6. the CPU path completes every supported workflow without a GPU;
7. packaged applications run on clean Ubuntu 26.04 and Windows installations,
   including model setup under a standard Windows user account;
8. local workflows operate offline, optional network actions are explicit,
   and the application contains no telemetry endpoint; and
9. the release license report contains no unknown or incompatible artifact.

## 10. Deferred roadmap

After v1, work may add still images, trim
points, additional camera profiles, signed model distribution, AV1 output,
and masking. These features shall not shape v1 UI or persistence in ways that
delay the core correction workflow, except that schemas and path handling must
remain forward-compatible.

## 11. Current coverage and open v1 work

The 0.4.3 code implements the primary single-clip correction workflow, automatic
analysis, twelve keyframable controls, comparison/scopes, projects/recovery,
presets, independent persistent batch processing, and all five codec families
listed above (HEVC has one profile with selectable SDR/HDR intent). It also
implements the optional model, orientation, finishing, reference-learning, and
cloud-advice workflows in section 3.1. See the [user guide](user-guide.md) for
the available actions and [technical specification](technical-spec.md) for
implementation limits. This replaces the old standalone implementation checklist.

The following requirements are still targets or need broader validation:

- **Color and quality:** calibrated display management, reviewed log/HDR
  behavior, calibrated model/filter confidence, broad real-camera/color-chart
  evaluation, and certified temporal/flicker metrics. C0201 reference comparisons
  measure similarity to a particular grade, not general physical color accuracy.
- **Editing and analysis UX:** recent-project list, a complete GUI relink flow,
  progressive publication of analysis, horizontal split/momentary bypass/free
  zoom, and complete accessibility/localization review. Analysis currently
  publishes its curve only on completion; canceled work is not a retained
  partial analysis. The current toolbar starts analysis automatically rather
  than offering the originally proposed Analyze button.
- **Batch and export UX:** folder/drop intake in the batch dialog, reordering
  and per-item actions, exact-current-grade transfer, disk-space estimation,
  comprehensive metadata controls/reporting, and global resource budgets.
  Current batch pause stops the operation and resume restarts that item.
- **Timing and playback:** exact final-packet duration for unusual VFR input,
  complete stream-level audio-offset validation, and visible-window/audible-sync
  checks. Historical Windows VFR export failures from the removed FFmpeg
  filter-script option are addressed by the compatible inline-filter path.
  Preview
  audio requires optional Qt Multimedia; export audio is handled by FFmpeg.
- **Distribution:** dependency locks/SBOM and complete binary-license provenance,
  broader clean-machine testing, and standard-user Store model-bootstrap
  verification. The reported `0xC0000022` Python startup failure has a candidate
  DLL-search isolation fix in 0.4.2, pending rebuilt-package retest.
  DirectML availability in a package is not hardware-validation evidence.

Numeric quality and timing gates in the technical specification remain v1
acceptance targets. Historical test counts and machine-specific benchmarks do
not establish that the current release satisfies all of them.


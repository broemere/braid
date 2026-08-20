# Changelog

## 0.10.0 - External recording handoff

### Added

- Added a generic `--open <recording-path>` launch argument for TIFF, AVI, and
  MKV recordings.
- Routed externally supplied recordings into the active analysis session after
  validating that the file exists and uses a supported format.
- Documented the path-only handoff so acquisition and analysis applications can
  remain independently installed and operated.

## 0.9.7 - Stability update

### Fixed

- Prevented frozen Windows and macOS geometry workers from launching additional
  BRAID windows.
- Repaired saved-session loading, including legacy sessions that contain
  structured telemetry arrays and the former end-only trim setting.
- Fixed geometry calculation for retained ranges shorter than five frames. The
  existing behavior remains unchanged: short ranges use their raw, unsmoothed
  geometry values.
- Stopped geometry cleanly when segmentation is empty, an ROI crop is empty, or
  the retained range has no distance variation.
- Preserved the original worker error when geometry processing fails and showed
  an actionable message instead of allowing a secondary missing-result error.
- Prevented the same session from queueing duplicate Get Geometry jobs and
  disabled the button while its job is active.
- Initialized the export path on every new pipeline so Open Folder can be used
  safely before the first export.
- Removed the extra Windows console window from packaged GUI builds.

### Release safeguards

- Windows packaging now runs the regression suite before building.
- An existing release asset can be replaced only when its tag points to the
  exact commit being built. Otherwise, the workflow requires a version bump.
- SHA-256 checksums are included for both macOS DMGs so uploaded and downloaded
  artifacts can be verified byte-for-byte.

### Analysis compatibility

- No geometry, mechanics, relaxation, last-pull, smoothing, threshold, or
  interpolation formulas were changed.
- No units, CSV columns, export formats, or telemetry timestamps were changed.
- Invalid inputs now stop processing rather than generating substitute values.

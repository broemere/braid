# BRAID maintenance policy and v0.9.7 fix record

BRAID produces research analysis, so maintenance changes must preserve the
existing scientific behavior unless a separately reviewed change explicitly
redefines it. A stability fix may correct a crash, restore an existing feature,
prevent stale or duplicated work, or improve an error message. It must not
silently replace missing measurements or alter formulas, constants, units,
threshold mappings, smoothing parameters, or exported values.

## Required workflow

1. Reproduce the defect with a regression test when practical.
2. Make the smallest change that restores the existing intended behavior.
3. Stop with a clear error when valid analysis cannot be completed.
4. Run the complete test suite and compilation checks.
5. Record the behavior preserved and any compatibility impact.

## v0.9.7 stability fixes

| Defect | Root cause | Correction | Preserved behavior | Regression coverage |
| --- | --- | --- | --- | --- |
| Geometry opened many BRAID windows in a frozen build | Multiprocessing children re-entered GUI imports before frozen-process routing | Run `freeze_support()` before lazy GUI imports | Geometry worker and segmentation algorithms are unchanged | `tests/test_main_entrypoint.py` |
| Saved sessions failed to open | Session refresh called methods and signals removed by the current UI architecture | Publish restored state through the signals currently consumed by the tabs | Existing session keys and legacy end-only trim behavior remain supported | `SavedSessionTests` |
| Structured telemetry failed after a JSON save/load cycle | Structured dtype text and JSON-converted record lists were not reconstructed correctly | Decode the existing dtype representation and restore record tuples | The on-disk session representation remains backward compatible | `SavedSessionTests.test_structured_telemetry_survives_the_existing_json_session_format` |
| Geometry failed for fewer than five retained frames | The no-smoothing branch assigned six arrays to five variables | Assign all six raw arrays, including volume | Short ranges still bypass Savitzky-Golay smoothing | `GeometryDimensionTests.test_less_than_five_frames_uses_raw_values_for_every_geometry_field` |
| Empty segmentation caused array-shape and video errors | Downstream code assumed every mask was non-empty and two-dimensional | Validate masks before geometry or threshold-video processing and stop on invalid input | No replacement dimensions, areas, or volumes are generated | `GeometryDimensionTests` |
| Constant-distance ranges crashed ROI interpolation | ROI percentage calculation divided by a zero distance span | Reject the unusable range before starting workers | ROI interpolation math for valid ranges is unchanged | `GeometryRequestTests.test_constant_distance_range_is_rejected_without_queuing_geometry` |
| Failed workers produced misleading secondary errors | Worker exceptions were printed and ignored before ordered results were reconstructed | Propagate the original exception to a task-specific user message | Successful worker result ordering is unchanged | `TaskManagerFailureTests` |
| Repeated clicks queued duplicate geometry jobs | Get Geometry remained enabled with no active-job guard | Track active geometry and disable the button until completion or failure | A normally started geometry job follows the same pipeline | `GeometryRequestTests.test_duplicate_geometry_request_is_rejected` |
| Windows Open Folder could access a missing attribute | `exported_file` was initialized as a local variable | Initialize it on the pipeline | Export path and CSV behavior are unchanged | `PipelineInitializationTests` |
| Windows release could ship without tests or overwrite a mismatched tag asset | Packaging preceded validation and existing tags were not verified | Run tests first and compare the release tag commit with the build commit | Archive naming and release format are unchanged | Workflow validation |

## Intentionally deferred

The following audit items require representative recordings, performance
measurements, or an explicit analysis-contract decision and are not part of
v0.9.7:

- Replacing telemetry row positions with the optional `frameIdx` field.
- Changing process counts, task batching, or voxel-array retention.
- Adding AVI decoding support.
- Changing cancellation semantics for an already-running geometry worker.
- Changing mechanics handling for zero or non-finite calculated values.


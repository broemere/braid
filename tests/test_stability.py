import json
import unittest
from unittest.mock import patch

import numpy as np

from data_pipeline import DataPipeline
from processing.data_transform import deserialize_objects, serialize_objects
from processing.data_loader import _normalize_and_format_mask
from processing.task_manager import TaskManager


def make_telemetry(distances=(9.0, 8.0, 7.0, 1.0, 5.0, 10.0)):
    dtype = [
        ("time_s", "f8"),
        ("distance", "f8"),
        ("force", "f8"),
        ("cycle", "i4"),
    ]
    return np.array(
        [
            (float(index), float(distance), float(index), index // 2)
            for index, distance in enumerate(distances)
        ],
        dtype=dtype,
    )


class RecordingSignal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


class RecordingTaskManager:
    def __init__(self):
        self.queued = []
        self.status_updated = RecordingSignal()

    def queue_task(self, function, *args, **kwargs):
        self.queued.append((function, args, kwargs))


class PipelineInitializationTests(unittest.TestCase):
    def test_exported_file_is_initialized_on_the_pipeline(self):
        self.assertIsNone(DataPipeline().exported_file)


class SavedSessionTests(unittest.TestCase):
    def test_structured_telemetry_survives_the_existing_json_session_format(self):
        source = make_telemetry()

        encoded = json.loads(json.dumps(serialize_objects(source)))
        restored = deserialize_objects(encoded)

        np.testing.assert_array_equal(restored, source)
        self.assertEqual(restored.dtype, source.dtype)

    def test_legacy_session_load_restores_range_and_current_signals(self):
        source = make_telemetry()
        source_dict = {name: source[name].copy() for name in source.dtype.names}
        pipeline = DataPipeline()
        emitted_data = []
        emitted_ranges = []
        pipeline.data_available.connect(emitted_data.append)
        pipeline.trim_range_changed.connect(
            lambda start, end: emitted_ranges.append((start, end))
        )

        pipeline.load_session({
            "data": source_dict,
            "data_trimmed": source[:5],
            "trim_time": 4.0,
            "video": None,
        })

        self.assertTrue(pipeline.loaded_state)
        self.assertEqual(pipeline.trim_start_time, 0.0)
        self.assertEqual(pipeline.trim_end_time, 4.0)
        self.assertEqual(pipeline.trim_time, 4.0)
        np.testing.assert_array_equal(pipeline.active_frame_indices, np.arange(5))
        self.assertEqual(len(emitted_data), 1)
        self.assertEqual(emitted_ranges[-1], (0.0, 4.0))


class GeometryDimensionTests(unittest.TestCase):
    @staticmethod
    def make_segment(mask):
        return {"mask": np.asarray(mask, dtype=bool), "offset_x": 0, "offset_y": 0}

    def test_less_than_five_frames_uses_raw_values_for_every_geometry_field(self):
        pipeline = DataPipeline()
        mask = np.ones((2, 2), dtype=bool)
        pipeline.first_segments = [self.make_segment(mask) for _ in range(3)]
        pipeline.second_segments = [self.make_segment(mask) for _ in range(3)]
        pipeline.conversion_factor = 1.0
        pipeline._dispatch_geometry = lambda: None

        pipeline.calculate_dimensions()

        self.assertEqual(pipeline._smoothed_geometry, pipeline._raw_geometry)
        self.assertEqual(len(pipeline._smoothed_geometry["volume"]), 3)

    def test_empty_segmentation_is_rejected_before_geometry_math(self):
        pipeline = DataPipeline()
        pipeline.first_segments = [self.make_segment(np.array([], dtype=bool))]
        pipeline.second_segments = [self.make_segment(np.ones((2, 2), dtype=bool))]

        with self.assertRaisesRegex(ValueError, "frame 0.*empty"):
            pipeline.calculate_dimensions()

    def test_empty_segmentation_is_rejected_before_threshold_video_formatting(self):
        with self.assertRaisesRegex(ValueError, "Threshold video mask.*empty"):
            _normalize_and_format_mask(np.array([], dtype=bool))


class TaskManagerFailureTests(unittest.TestCase):
    def test_task_specific_error_callback_can_present_a_recoverable_failure(self):
        manager = TaskManager()
        handled = []
        unexpected = []
        manager.error_occurred.connect(unexpected.append)
        manager.is_running = True
        manager.task_callbacks["geometry"] = (
            None,
            lambda err_tb: handled.append(str(err_tb[0])) or True,
        )

        manager._on_error("geometry", (ValueError("bad mask"), "traceback"))

        self.assertEqual(handled, ["bad mask"])
        self.assertEqual(unexpected, [])


class GeometryRequestTests(unittest.TestCase):
    @staticmethod
    def configured_pipeline(distances):
        pipeline = DataPipeline()
        pipeline.video = "recording.tif"
        pipeline.data = {"distance": list(distances)}
        pipeline.data_trimmed = make_telemetry(distances)
        pipeline.active_frame_indices = np.arange(len(distances))
        pipeline.min_distance_data_index = int(np.argmin(distances))
        pipeline.max_distance_data_index = int(np.argmax(distances))
        pipeline.roi_data = {
            "min": [{}, {}],
            "max": [{}, {}],
        }
        pipeline.task_manager = RecordingTaskManager()
        return pipeline

    def test_constant_distance_range_is_rejected_without_queuing_geometry(self):
        pipeline = self.configured_pipeline((2.0, 2.0, 2.0))

        with patch("data_pipeline.user_error") as warning:
            started = pipeline.get_geometry()

        self.assertFalse(started)
        self.assertEqual(pipeline.task_manager.queued, [])
        warning.assert_called_once()

    def test_duplicate_geometry_request_is_rejected(self):
        pipeline = self.configured_pipeline((1.0, 2.0, 3.0))
        pipeline.geometry_in_progress = True

        with patch("data_pipeline.user_error") as warning:
            started = pipeline.get_geometry()

        self.assertFalse(started)
        self.assertEqual(pipeline.task_manager.queued, [])
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()

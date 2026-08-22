import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tifffile import TiffWriter

from processing.trimming import (
    iter_selected_frames,
    map_distance_extrema_to_source,
    select_time_range,
)
from processing.telemetry import build_analysis_records

def load_pipeline_class():
    """Load DataPipeline with small Qt/OpenCV stubs in lightweight CI runtimes."""
    if importlib.util.find_spec("PySide6") is None:
        pyside_module = types.ModuleType("PySide6")
        qtcore_module = types.ModuleType("PySide6.QtCore")
        qtgui_module = types.ModuleType("PySide6.QtGui")

        class SignalInstance:
            def __init__(self, *_args):
                self._slots = []

            def connect(self, slot):
                self._slots.append(slot)

            def emit(self, *args):
                for slot in self._slots:
                    slot(*args)

        class QObject:
            def __init__(self, parent=None):
                self.parent = parent

        def Slot(*_args):
            return lambda function: function

        class QTimer:
            @staticmethod
            def singleShot(_delay, callback):
                callback()

        class QImage:
            Format_Grayscale8 = 1
            Format_RGBA8888 = 2
            Format_RGB888 = 3

        class QPixmap:
            @staticmethod
            def fromImage(_image):
                return QPixmap()

        qtcore_module.QObject = QObject
        qtcore_module.Signal = SignalInstance
        qtcore_module.SignalInstance = SignalInstance
        qtcore_module.Slot = Slot
        qtcore_module.QTimer = QTimer
        qtcore_module.QRect = object
        qtgui_module.QImage = QImage
        qtgui_module.QPixmap = QPixmap
        pyside_module.QtCore = qtcore_module
        pyside_module.QtGui = qtgui_module
        sys.modules.update({
            "PySide6": pyside_module,
            "PySide6.QtCore": qtcore_module,
            "PySide6.QtGui": qtgui_module,
        })

    if importlib.util.find_spec("cv2") is None:
        cv2_module = types.ModuleType("cv2")
        cv2_module.MORPH_ELLIPSE = 1
        cv2_module.getStructuringElement = lambda *_args: np.ones((1, 1), dtype=np.uint8)
        sys.modules["cv2"] = cv2_module

    resource_loader_module = types.ModuleType("processing.resource_loader")
    resource_loader_module.resource_path = lambda relative_path: relative_path
    sys.modules["processing.resource_loader"] = resource_loader_module

    from data_pipeline import DataPipeline
    return DataPipeline


DataPipeline = load_pipeline_class()


def make_telemetry():
    dtype = [
        ("time_s", "f8"),
        ("distance", "f8"),
        ("force", "f8"),
        ("cycle", "i4"),
    ]
    rows = [
        (0.0, 9.0, 0.0, 0),
        (1.0, 8.0, 1.0, 0),
        (2.0, 7.0, 2.0, 1),
        (3.0, 1.0, 3.0, 1),
        (4.0, 5.0, 4.0, 2),
        (5.0, 10.0, 5.0, 2),
    ]
    return np.array(rows, dtype=dtype)


class TelemetryRecordTests(unittest.TestCase):
    def test_auxiliary_dictionary_metadata_is_ignored(self):
        data = {
            "time_s": [0.0, 0.1],
            "distance": [1.0, 2.0],
            "force": [3.0, 4.0],
            "cycle": [0, 0],
            "frameIdx": [0, 1],
            "crop_roi": [{"x": 10}, {"x": 10}],
        }

        records = build_analysis_records(data)

        self.assertEqual(
            records.dtype.names,
            ("time_s", "distance", "force", "cycle", "frameIdx"),
        )
        np.testing.assert_array_equal(records["force"], [3.0, 4.0])

    def test_missing_required_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "force"):
            build_analysis_records({
                "time_s": [0.0],
                "distance": [1.0],
                "cycle": [0],
            })


class TimeRangeSelectionTests(unittest.TestCase):
    def test_selection_is_inclusive_and_preserves_original_time(self):
        data = make_telemetry()

        selected, source_indices = select_time_range(data, 2.0, 4.0)

        np.testing.assert_array_equal(selected["time_s"], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(source_indices, [2, 3, 4])

    def test_selection_between_samples_uses_samples_inside_the_range(self):
        data = make_telemetry()

        selected, source_indices = select_time_range(data, 1.1, 4.8)

        np.testing.assert_array_equal(selected["time_s"], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(source_indices, [2, 3, 4])

    def test_empty_selection_does_not_invent_a_sample(self):
        selected, source_indices = select_time_range(make_telemetry(), 1.1, 1.9)

        self.assertEqual(selected.size, 0)
        self.assertEqual(source_indices.size, 0)


class ExtremaMappingTests(unittest.TestCase):
    def test_local_extrema_map_back_to_source_video_frames(self):
        data = make_telemetry()[2:5]

        result = map_distance_extrema_to_source(data, np.array([2, 3, 4]))

        self.assertEqual(result, (0, 1, 2, 3))

    def test_mismatched_source_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            map_distance_extrema_to_source(make_telemetry()[:2], np.array([0]))


class PipelineTrimMappingTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = DataPipeline()
        pipeline.min_distance_index = 3
        pipeline.max_distance_index = 5
        loaded_frames = []
        pipeline.load_frames = lambda indices: loaded_frames.extend(indices)
        return pipeline, loaded_frames

    def test_pipeline_stores_local_and_source_extrema_separately(self):
        data = make_telemetry()
        trimmed = data[2:5]
        pipeline, loaded_frames = self.make_pipeline()

        pipeline.set_trimmed_data(2.0, 4.0, trimmed, np.array([2, 3, 4]))

        self.assertEqual(pipeline.max_distance_data_index, 0)
        self.assertEqual(pipeline.min_distance_data_index, 1)
        self.assertEqual(pipeline.max_distance_index, 2)
        self.assertEqual(pipeline.min_distance_index, 3)
        self.assertEqual(loaded_frames, [2])

    def test_changed_range_invalidates_derived_arrays_and_increments_revision(self):
        data = make_telemetry()
        pipeline, _ = self.make_pipeline()
        pipeline.set_trimmed_data(0.0, 5.0, data, np.arange(len(data)))
        pipeline.geometry_data = {"frames": list(range(len(data)))}
        pipeline.mechanics_payload = {"time_s": data["time_s"].tolist()}
        pipeline.relaxation_payload = {"time_s": data["time_s"].tolist()}
        pipeline.last_pull_stretch = np.array([1.0, 1.1])
        pipeline._raw_geometry = {"frames": list(range(len(data)))}
        pipeline._smoothed_geometry = {"frames": list(range(len(data)))}
        pipeline.first_segments = [object()]
        pipeline.second_segments = [object()]

        pipeline.set_trimmed_data(2.0, 4.0, data[2:5], np.array([2, 3, 4]))

        self.assertEqual(pipeline.trim_revision, 1)
        self.assertIsNone(pipeline.geometry_data)
        self.assertIsNone(pipeline.mechanics_payload)
        self.assertIsNone(pipeline.relaxation_payload)
        self.assertIsNone(pipeline.last_pull_stretch)
        self.assertFalse(hasattr(pipeline, "_raw_geometry"))
        self.assertFalse(hasattr(pipeline, "_smoothed_geometry"))
        self.assertFalse(hasattr(pipeline, "first_segments"))
        self.assertFalse(hasattr(pipeline, "second_segments"))

    def test_end_time_alias_and_range_signal_state_are_preserved(self):
        data = make_telemetry()
        pipeline, _ = self.make_pipeline()

        pipeline.set_trimmed_data(1.0, 4.0, data[1:5], np.arange(1, 5))

        self.assertEqual(pipeline.trim_start_time, 1.0)
        self.assertEqual(pipeline.trim_end_time, 4.0)
        self.assertEqual(pipeline.trim_time, 4.0)

    def test_new_dataset_invalidates_results_even_when_range_is_unchanged(self):
        data = make_telemetry()
        pipeline, _ = self.make_pipeline()
        pipeline.set_trimmed_data(0.0, 5.0, data, np.arange(len(data)))
        pipeline.geometry_data = {"frames": list(range(len(data)))}
        replacement = data.copy()
        replacement["force"] += 10.0

        pipeline.set_trimmed_data(0.0, 5.0, replacement, np.arange(len(replacement)))

        self.assertEqual(pipeline.trim_revision, 1)
        self.assertIsNone(pipeline.geometry_data)

    def test_pipeline_rejects_mismatched_source_mapping(self):
        pipeline, _ = self.make_pipeline()

        with self.assertRaises(ValueError):
            pipeline.set_trimmed_data(0.0, 1.0, make_telemetry()[:2], np.array([0]))


class SelectedFrameReaderTests(unittest.TestCase):
    @staticmethod
    def make_fake_capture():
        class FakeCapture:
            def __init__(self):
                self.position = 0
                self.set_calls = []

            def isOpened(self):
                return True

            def set(self, _property, value):
                self.position = int(value)
                self.set_calls.append(self.position)
                return True

            def read(self):
                value = self.position
                self.position += 1
                return True, np.full((2, 2, 3), value, dtype=np.uint8)

            def release(self):
                pass

        return FakeCapture()

    def test_tiff_reader_uses_requested_source_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frames.tif"
            with TiffWriter(path) as tif:
                for frame_number in range(5):
                    tif.write(np.full((4, 4), frame_number, dtype=np.uint8))

            frames = list(iter_selected_frames(str(path), np.array([4, 1, 3])))

        self.assertEqual([int(frame[0, 0]) for frame in frames], [4, 1, 3])

    def test_mkv_reader_seeks_once_for_a_contiguous_range(self):
        fake_capture = self.make_fake_capture()
        fake_cv2 = SimpleNamespace(
            VideoCapture=lambda _path: fake_capture,
            CAP_PROP_POS_FRAMES=1,
        )
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            frames = list(iter_selected_frames("recording.mkv", np.array([2, 3, 4])))

        self.assertEqual(fake_capture.set_calls, [2])
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [2, 3, 4])

    def test_mkv_reader_seeks_each_requested_noncontiguous_frame(self):
        fake_capture = self.make_fake_capture()
        fake_cv2 = SimpleNamespace(
            VideoCapture=lambda _path: fake_capture,
            CAP_PROP_POS_FRAMES=1,
        )
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            frames = list(iter_selected_frames("recording.mkv", np.array([4, 1, 3])))

        self.assertEqual(fake_capture.set_calls, [4, 1, 3])
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [4, 1, 3])


if __name__ == "__main__":
    unittest.main()

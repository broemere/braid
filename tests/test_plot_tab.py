import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from data_pipeline import DataPipeline
    from tabs.plot_tab import PlotTab
except ModuleNotFoundError:
    QApplication = None
    DataPipeline = None
    PlotTab = None


@unittest.skipUnless(QApplication is not None, "BRAID GUI runtime dependencies are not installed")
class PlotTabRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.pipeline = DataPipeline()
        self.pipeline.min_distance_index = 3
        self.pipeline.max_distance_index = 5
        self.pipeline.load_frames = lambda _indices: None
        self.widget = PlotTab(self.pipeline)
        self.data = {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "distance": [9.0, 8.0, 7.0, 1.0, 5.0, 10.0],
            "force": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cycle": [0, 0, 1, 1, 2, 2],
            "crop_roi": [{"x": 10, "y": 20, "width": 100, "height": 80}] * 6,
        }
        self.widget.on_new_data_received(self.data)

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_start_and_end_controls_drive_pipeline_range(self):
        self.assertNotIn("crop_roi", self.widget.data.dtype.names)
        self.assertEqual(self.widget.trim_start_spinbox.value(), 0.0)
        self.assertEqual(self.widget.trim_end_spinbox.value(), 5.0)

        cycle_zero = next(
            button
            for button in self.widget.cycle_selection_group.buttons()
            if button.text() == "Cycle 0"
        )
        cycle_zero.click()
        self.widget.trim_start_spinbox.setValue(2.0)
        self.widget.trim_end_spinbox.setValue(4.0)
        self.widget.apply_trimming()

        np.testing.assert_array_equal(self.widget.data_trimmed["time_s"], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(self.pipeline.active_frame_indices, [2, 3, 4])
        self.assertEqual(self.pipeline.trim_start_time, 2.0)
        self.assertEqual(self.pipeline.trim_end_time, 4.0)
        self.assertEqual(self.pipeline.cycle_selection, "All Cycles")

    def test_empty_range_preserves_last_applied_selection(self):
        self.widget.trim_start_spinbox.setValue(2.0)
        self.widget.trim_end_spinbox.setValue(4.0)
        self.widget.apply_trimming()

        with patch("tabs.plot_tab.QMessageBox.warning") as warning:
            self.widget.trim_start_spinbox.setValue(1.1)
            self.widget.trim_end_spinbox.setValue(1.9)
            self.widget.apply_trimming()

        warning.assert_called_once()
        np.testing.assert_array_equal(self.widget.data_trimmed["time_s"], [2.0, 3.0, 4.0])
        self.assertEqual(self.widget.trim_start_spinbox.value(), 2.0)
        self.assertEqual(self.widget.trim_end_spinbox.value(), 4.0)


if __name__ == "__main__":
    unittest.main()

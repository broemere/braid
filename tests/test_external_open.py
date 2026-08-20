import tempfile
import unittest
from pathlib import Path

from window import MainWindow


class _AnalysisHarness:
    def __init__(self):
        self.opened = []

    def on_file_selected(self, path):
        self.opened.append(path)


class _TabsHarness:
    def __init__(self, analysis):
        self.analysis = analysis

    def currentWidget(self):
        return self.analysis


class _WindowHarness:
    def __init__(self):
        self.analysis = _AnalysisHarness()
        self.super_tabs = _TabsHarness(self.analysis)
        self.raised = False
        self.activated = False

    def raise_(self):
        self.raised = True

    def activateWindow(self):
        self.activated = True


class ExternalOpenTests(unittest.TestCase):
    def test_supported_recording_routes_to_the_active_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "Run 2 specimen_video.tif"
            recording.write_bytes(b"placeholder")
            window = _WindowHarness()

            opened = MainWindow.open_video_path(window, str(recording))

            self.assertTrue(opened)
            self.assertEqual(window.analysis.opened, [str(recording.resolve())])
            self.assertTrue(window.raised)
            self.assertTrue(window.activated)


if __name__ == "__main__":
    unittest.main()

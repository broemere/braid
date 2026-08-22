import tempfile
import unittest
from pathlib import Path

from window import MainWindow


class _AnalysisHarness:
    def __init__(self, video=None):
        self.opened = []
        self.pipeline = type("PipelineHarness", (), {"video": video})()

    def on_file_selected(self, path):
        self.opened.append(path)


class _TabsHarness:
    def __init__(self, analysis):
        self.analyses = [analysis]
        self.current_index = 0

    def currentWidget(self):
        return self.analyses[self.current_index]

    def widget(self, index):
        return self.analyses[index]

    def add(self, analysis):
        self.analyses.append(analysis)
        self.current_index = len(self.analyses) - 1
        return self.current_index


class _WindowHarness:
    def __init__(self, video=None):
        self.analysis = _AnalysisHarness(video)
        self.super_tabs = _TabsHarness(self.analysis)
        self.raised = False
        self.activated = False

    def add_new_super_tab(self):
        return self.super_tabs.add(_AnalysisHarness())

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

    def test_external_recording_reuses_an_initial_blank_session(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "Run 3 specimen_video.tif"
            recording.write_bytes(b"placeholder")
            window = _WindowHarness()

            opened = MainWindow.open_video_path(
                window, str(recording), new_session=True
            )

            self.assertTrue(opened)
            self.assertEqual(len(window.super_tabs.analyses), 1)
            self.assertEqual(
                window.analysis.opened,
                [str(recording.resolve())],
            )

    def test_external_recording_preserves_an_occupied_session(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "Run 4 specimen_video.tif"
            recording.write_bytes(b"placeholder")
            window = _WindowHarness(video="C:/Data/existing.tif")

            opened = MainWindow.open_video_path(
                window, str(recording), new_session=True
            )

            self.assertTrue(opened)
            self.assertEqual(len(window.super_tabs.analyses), 2)
            self.assertEqual(window.analysis.opened, [])
            self.assertEqual(
                window.super_tabs.currentWidget().opened,
                [str(recording.resolve())],
            )


if __name__ == "__main__":
    unittest.main()

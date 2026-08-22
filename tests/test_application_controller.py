import unittest

from application_controller import ApplicationWindowController


class _SignalHarness:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _WindowHarness:
    def __init__(self):
        self.destroyed = _SignalHarness()
        self.visible = False
        self.opened = []

    def show(self):
        self.visible = True

    def isVisible(self):
        return self.visible

    def open_video_path(self, path, new_session=False):
        self.opened.append((path, new_session))


class ApplicationWindowControllerTests(unittest.TestCase):
    def setUp(self):
        self.created = []

        def factory():
            window = _WindowHarness()
            self.created.append(window)
            return window

        self.controller = ApplicationWindowController(factory)

    def test_normal_launch_request_creates_another_visible_window(self):
        first = self.controller.create_window()

        self.controller.handle_request({"action": "new_window"})

        self.assertEqual(len(self.controller.windows), 2)
        self.assertTrue(first.visible)
        self.assertTrue(self.created[1].visible)

    def test_external_open_targets_main_window_as_a_new_session(self):
        first = self.controller.create_window()
        self.controller.create_window()

        self.controller.handle_request(
            {"action": "open", "path": "C:/recordings/specimen.tif"}
        )

        self.assertEqual(
            first.opened,
            [("C:/recordings/specimen.tif", True)],
        )
        self.assertEqual(self.created[1].opened, [])

    def test_external_open_uses_a_remaining_visible_window(self):
        first = self.controller.create_window()
        second = self.controller.create_window()
        first.visible = False

        self.controller.handle_request(
            {"action": "open", "path": "C:/recordings/specimen.tif"}
        )

        self.assertEqual(second.opened, [("C:/recordings/specimen.tif", True)])


if __name__ == "__main__":
    unittest.main()

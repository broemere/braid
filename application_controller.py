"""Coordinate BRAID windows and requests received from later launches."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping


log = logging.getLogger(__name__)


class ApplicationWindowController:
    """Keep multiple BRAID windows alive and route launch requests to them."""

    def __init__(self, window_factory: Callable[[], object]):
        self._window_factory = window_factory
        self._windows: list[object] = []

    @property
    def windows(self) -> tuple[object, ...]:
        """Return the windows retained by the application process."""

        return tuple(self._windows)

    def create_window(self):
        """Create, retain, and show a separate BRAID window."""

        window = self._window_factory()
        self._windows.append(window)
        window.show()
        destroyed = getattr(window, "destroyed", None)
        if destroyed is not None:
            destroyed.connect(
                lambda _object=None, retained=window: self._forget_window(retained)
            )
        return window

    def handle_request(self, request: Mapping[str, object]) -> None:
        """Apply a validated request forwarded by another BRAID invocation."""

        action = request.get("action")
        if action == "new_window":
            self.create_window()
            return
        if action != "open":
            log.warning("Ignoring unknown application request: %r", action)
            return

        path = request.get("path")
        if not isinstance(path, str) or not path:
            log.warning("Ignoring an external-open request without a path.")
            return

        window = self._main_visible_window() or self.create_window()
        window.open_video_path(path, new_session=True)

    def _main_visible_window(self):
        for window in self._windows:
            try:
                if window.isVisible():
                    return window
            except RuntimeError:
                continue
        return None

    def _forget_window(self, window) -> None:
        try:
            self._windows.remove(window)
        except ValueError:
            pass

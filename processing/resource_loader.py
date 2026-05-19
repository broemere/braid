import os
import sys
import logging
import logging.handlers
import tempfile
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QCursor
from config import APP_NAME


def setup_logging():
    """
    Sets up logging to a safe, platform-specific location.
    On macOS, this will be ~/Library/Logs/proper/
    On Windows, C:/Users/user/AppData/Local/Temp/proper/
    """
    if sys.platform == "darwin":  # Mac
        # Get the user's Library/Logs directory
        log_dir = Path.home() / "Library" / "Logs" / APP_NAME
    else:
        # Fallback to current directory if not frozen
        log_dir = Path(tempfile.gettempdir()) / APP_NAME if getattr(sys, "frozen", False) else Path.cwd()

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = log_dir / f"{APP_NAME.lower()}.log"

    # Use a rotating file handler to prevent log files from getting too large
    handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=1
    )

    # In development, also print to console. When bundled, only log to file.
    if getattr(sys, "frozen", False):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[handler]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[handler, logging.StreamHandler(sys.stdout)]
        )
    print(f"Logging to {log_file_path}")
    return logging.getLogger(__name__)


def resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource, works for dev and for PyInstaller.
    The 'relative_path' should be the path to the resource relative to the
    project's root directory.
    """
    if getattr(sys, "frozen", False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        # If the application is run as a script, find the project root
        # by going up one level from this file's directory (processing/).
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    return os.path.join(base_path, relative_path)

def load_icon():
    if sys.platform == 'darwin':  # macOS
        icon_filename = 'app.icns'
    else:  # Windows (and other platforms)
        icon_filename = 'app.ico'

    icon_path = resource_path(f"resources/{icon_filename}")
    return icon_path
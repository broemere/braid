from PySide6.QtCore import Qt, Slot, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QComboBox, QFormLayout
from PySide6.QtGui import QDesktopServices
from data_pipeline import DataPipeline
#from processing.data_transform import format_value
import platform
import subprocess
import logging
import os
log = logging.getLogger(__name__)


class ExportTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.first_labels = {}
        self.last_labels = {}
        self.init_ui()
        self.connect_signals()


    def init_ui(self):
        """Initializes the user interface."""
        main_layout = QVBoxLayout(self)

        export_layout = QHBoxLayout()
        export_layout.addStretch()

        # 1. Style the Export Button
        self.export_btn = QPushButton("📥 Export Results")
        self.export_btn.setMinimumHeight(35)
        self.export_btn.setMinimumWidth(150)
        self.export_btn.setStyleSheet("""
                    QPushButton {
                        border-radius: 5px;
                        border: 1px solid #223620;
                        font-weight: bold;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background-color: #88C484;
                    }
                    QPushButton:pressed {
                        background-color: #40633D;  /* Darker when clicked */
                        padding-left: 17px;        /* Subtle "push" effect */
                        padding-top: 7px;
                    }
                    QPushButton:disabled {
                        background-color: #a0a0a0;
                        border: 1px solid #808080;
                    }
                """)

        # 2. Style the Open Folder Button (Secondary Action)
        self.open_btn = QPushButton("📂 Open Folder")
        self.open_btn.setMinimumHeight(35)
        self.open_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #333333;
                        border: 1px solid #cccccc;
                        border-radius: 5px;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background-color: #f0f0f0;
                        border-color: #999999;
                    }
                """)

        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.open_btn)
        export_layout.addStretch()
        main_layout.addLayout(export_layout)

    def connect_signals(self):
        """Connects UI widget signals to pipeline and pipeline signals to UI slots."""
        self.export_btn.clicked.connect(self.pipeline.generate_report)
        self.open_btn.clicked.connect(self.open_current_directory)

    @Slot()
    def open_current_directory(self):
        """Opens the OS file explorer at the directory of the current CSV."""
        # Assuming self.pipeline.csv_path exists based on your description
        video_path = getattr(self.pipeline, 'video', None)

        if not video_path or not os.path.exists(video_path):
            log.warning("Cannot open folder: Video is invalid or does not exist.")
            return

        folder_path = os.path.dirname(os.path.abspath(video_path))

        # The Cross-Platform Way
        if platform.system() == "Windows":
            # On Windows, QDesktopServices works, but using 'explorer' directly
            # allows you to pre-select the file if you wanted.
            # For just the folder, this is the most reliable:
            if self.pipeline.exported_file is not None:
                subprocess.run(['explorer', '/select,', os.path.normpath(self.pipeline.exported_file)])
            else:
                os.startfile(folder_path)
        else:
            # On macOS and Linux, QDesktopServices is excellent
            folder_url = QUrl.fromLocalFile(folder_path)
            QDesktopServices.openUrl(folder_url)

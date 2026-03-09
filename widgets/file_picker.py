import os
from pathlib import Path
from PySide6.QtCore import Signal, QSettings, Qt
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStyle, QWidget, QFileDialog


class FilePickerWidget(QWidget):
    video_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        v = QVBoxLayout(self)
        row_video = QHBoxLayout()
        btn_video = QPushButton('Import Video')
        btn_video.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        btn_video.clicked.connect(self.choose_video)
        row_video.addWidget(btn_video)
        self.video_path = QLabel('')
        #self.video_path.setStyleSheet('font-size: 14pt;')
        self.video_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_video.addWidget(self.video_path)
        self.video_label = QLabel('')
        self.video_label.setStyleSheet('font-size: 14pt; color: #00a007;')
        self.video_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_video.addWidget(self.video_label)
        row_video.addStretch()
        v.addLayout(row_video)

    def choose_video(self):
        last_dir = self.settings.value("last_dir", "") or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select Video file", last_dir,
                                               "Video Files (*.avi *.mkv *.tif *.tiff);;All Files (*)")
        if path:
            self.settings.setValue("last_dir", str(Path(path).parent))
            self.video_selected.emit(path)

    def set_video_label(self, path):
        self.video_path.setText(str(Path(path).parent)+os.path.sep)
        self.video_label.setText(Path(path).name)

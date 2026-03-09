import numpy as np
from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStyle, QSizePolicy, QFrame
from data_pipeline import DataPipeline
from processing.data_transform import numpy_to_qpixmap
from widgets.roi_widget import ROICanvas


class ROITab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        canvas_layout = QHBoxLayout()

        # --- Image Section ---
        self.min_container = self._create_canvas_group("Minimum Distance Image")
        self.canvas_min = self.min_container['canvas']
        canvas_layout.addWidget(self.min_container['frame'], stretch=1)
        self.max_container = self._create_canvas_group("Maximum Distance Image")
        self.canvas_max = self.max_container['canvas']
        canvas_layout.addWidget(self.max_container['frame'], stretch=1)
        main_layout.addLayout(canvas_layout)

        # Global Actions (Optional: buttons controlling both)
        global_ctrl_row = QHBoxLayout()
        self.reset_all_btn = QPushButton("Reset Both Canvases")
        global_ctrl_row.addStretch()
        global_ctrl_row.addWidget(self.reset_all_btn)
        global_ctrl_row.addStretch()
        main_layout.addLayout(global_ctrl_row)

    def _create_canvas_group(self, title: str):
        """Helper to create a canvas with a label and local controls."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        layout = QVBoxLayout(frame)
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        canvas = ROICanvas()
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Local controls for this specific canvas
        btn_layout = QHBoxLayout()
        undo_btn = QPushButton(f"Undo ({self.pipeline.ctrl_key}+Z)")
        refresh_btn = QPushButton(self.style().standardIcon(QStyle.SP_BrowserReload), "")
        refresh_btn.setToolTip("Reload original image and clear boxes.")
        btn_layout.addStretch()
        btn_layout.addWidget(undo_btn)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()

        layout.addWidget(label)
        layout.addWidget(canvas, stretch=1)
        layout.addLayout(btn_layout)

        return {
            'frame': frame,
            'canvas': canvas,
            'undo': undo_btn,
            'refresh': refresh_btn
        }

    def connect_signals(self):
        # 1. Pipeline -> UI (Loading images)
        self.pipeline.roi_min_image_loaded.connect(self.on_min_image_loaded)
        self.pipeline.roi_max_image_loaded.connect(self.on_max_image_loaded)

        # 2. UI -> Pipeline (Sending ROI box data)
        # Using lambda to specify which image the data is for
        self.canvas_min.roi_updated.connect(
            lambda rois: self.pipeline.receive_roi_data(rois, target="min")
        )
        self.canvas_max.roi_updated.connect(
            lambda rois: self.pipeline.receive_roi_data(rois, target="max")
        )

        # 3. Local Button Controls
        self.min_container['undo'].clicked.connect(self.canvas_min.undo_last_roi)
        self.min_container['refresh'].clicked.connect(self._reload_min_image)
        self.max_container['undo'].clicked.connect(self.canvas_max.undo_last_roi)
        self.max_container['refresh'].clicked.connect(self._reload_max_image)
        self.reset_all_btn.clicked.connect(self._reset_everything)

    @Slot(np.ndarray)
    def on_min_image_loaded(self, img_array: np.ndarray):
        pixmap = numpy_to_qpixmap(img_array)
        self.canvas_min.set_background(pixmap)

    @Slot(np.ndarray)
    def on_max_image_loaded(self, img_array: np.ndarray):
        pixmap = numpy_to_qpixmap(img_array)
        self.canvas_max.set_background(pixmap)

    def _reload_min_image(self):
        # Accessing your pipeline's frame data for the min index
        img = self.pipeline.frame_data[self.pipeline.min_distance_index]
        self.on_min_image_loaded(img)

    def _reload_max_image(self):
        img = self.pipeline.frame_data[self.pipeline.max_distance_index]
        self.on_max_image_loaded(img)

    def _reset_everything(self):
        self.canvas_min.reset_rois()
        self.canvas_max.reset_rois()
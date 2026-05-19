import os
import numpy as np
from PySide6.QtCore import Slot, Qt, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
)
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QFont, QDesktopServices
import pyqtgraph as pg
from data_pipeline import DataPipeline
from processing.data_transform import numpy_to_qpixmap


class ResizableImageLabel(QLabel):
    """
    A custom label that dynamically scales its pixmap to fit its geometry
    without forcing layout changes or entering an infinite resize loop.
    """

    def __init__(self, text_placeholder="Waiting for image..."):
        super().__init__(text_placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #444; background: #222; color: #888;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)
        self._pixmap = None

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.setText("")
        self.update()

    def paintEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        scaled_pix = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        x_offset = (self.width() - scaled_pix.width()) // 2
        y_offset = (self.height() - scaled_pix.height()) // 2

        painter.drawPixmap(x_offset, y_offset, scaled_pix)


class GeometryTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline

        pg.setConfigOptions(antialias=True)

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- Top Controls ---
        control_layout = QHBoxLayout()
        self.btn_get_geometry = QPushButton("Get Geometry")
        self.btn_get_geometry.setMinimumHeight(40)

        # NEW: Checkbox for toggling smoothing
        from PySide6.QtWidgets import QCheckBox  # Added locally if not in global import
        self.chk_smoothing = QCheckBox("Apply Data Smoothing")
        self.chk_smoothing.setChecked(getattr(self.pipeline, 'use_smoothing', True))

        self.btn_open_folder = QPushButton("📁 Open Output Folder")
        self.btn_open_folder.setMinimumHeight(40)
        self.btn_open_folder.setEnabled(False)  # Disabled until video is ready


        control_layout.addWidget(self.btn_get_geometry)
        control_layout.addStretch()
        control_layout.addWidget(self.chk_smoothing)  # Inserted in the middle
        control_layout.addStretch()
        control_layout.addWidget(self.btn_open_folder)


        layout.addLayout(control_layout)

        # --- Main Content Area (Left Column + Right Plots) ---
        content_layout = QHBoxLayout()

        # >> LEFT COLUMN: Images and Swap Button
        left_column = QVBoxLayout()

        # 1. X-Y Plane Title & Image
        lbl_xy_title = QLabel("X-Y Orientation (Top)")
        lbl_xy_title.setAlignment(Qt.AlignCenter)
        lbl_xy_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.lbl_img_xy = ResizableImageLabel("Waiting for X-Y ROI...")

        # 2. Swap Button
        self.btn_swap = QPushButton("↕ Swap")
        self.btn_swap.setMinimumHeight(30)

        # 3. Z Plane Title & Image
        lbl_z_title = QLabel("Z Orientation (Mirror)")
        lbl_z_title.setAlignment(Qt.AlignCenter)
        lbl_z_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.lbl_img_z = ResizableImageLabel("Waiting for Z ROI...")

        # Pack the left column
        left_column.addWidget(lbl_xy_title)
        left_column.addWidget(self.lbl_img_xy, stretch=1)
        left_column.addWidget(self.btn_swap)
        left_column.addWidget(lbl_z_title)
        left_column.addWidget(self.lbl_img_z, stretch=1)

        content_layout.addLayout(left_column, stretch=1)

        # >> RIGHT COLUMN: Plotting Area
        self.plot_widget = pg.GraphicsLayoutWidget()
        content_layout.addWidget(self.plot_widget, stretch=4)

        layout.addLayout(content_layout, stretch=1)

        # --- Setup Plots (2x2 Grid) ---

        # ROW 1: Y Length & Z Length
        self.plot_y = self.plot_widget.addPlot(title="Avg. Object Width (Y-Axis / 2D Projection)")
        self.plot_y.setLabel('left', 'Width (Y)', units='m')
        self.curve_y = self.plot_y.plot(pen=pg.mkPen(color='#00d2ff', width=2))

        self.plot_z = self.plot_widget.addPlot(title="Avg. Object Thickness (Z-Axis / 2D Projection)")
        self.plot_z.setLabel('left', 'Thickness (Z)', units='m')
        self.curve_z = self.plot_z.plot(pen=pg.mkPen(color='#ff007f', width=2))

        self.plot_widget.nextRow()

        # ROW 2: Y-Z Area & X Length
        self.plot_area = self.plot_widget.addPlot(title="Avg. YZ Cross-Sectional Area (3D Voxel Reconstruction)")
        self.plot_area.setLabel('left', 'Area (Y-Z Plane)', units='m²')
        self.plot_area.setLabel('bottom', 'Frame Index')
        self.curve_area = self.plot_area.plot(pen=pg.mkPen(color='#00ff00', width=2))

        self.plot_x = self.plot_widget.addPlot(title="Max. Object Length (X-Axis / Stretch Direction)")
        self.plot_x.setLabel('left', 'Length (X)', units='m')
        self.plot_x.setLabel('bottom', 'Frame Index')
        self.curve_x = self.plot_x.plot(pen=pg.mkPen(color='#ffaa00', width=2))

        # Link all X-axes together for synchronized panning/zooming
        self.plot_z.setXLink(self.plot_y)
        self.plot_area.setXLink(self.plot_y)
        self.plot_x.setXLink(self.plot_y)

    def connect_signals(self):
        # UI -> Pipeline
        self.btn_get_geometry.clicked.connect(self.pipeline.get_geometry)
        self.btn_swap.clicked.connect(self.pipeline.swap_dimensions)

        # Connect the checkbox to instantly toggle data output
        self.chk_smoothing.toggled.connect(self.pipeline.set_smoothing_enabled)

        # Pipeline -> UI
        self.pipeline.geometry_available.connect(self.on_new_data_received)
        self.pipeline.dimension_images_ready.connect(self.on_dimension_images_ready)
        self.pipeline.video_compiled_ready.connect(self.on_video_compiled)
        self.btn_open_folder.clicked.connect(self.on_open_folder_clicked)

    def showEvent(self, event):
        super().showEvent(event)
        self.pipeline.request_dimension_images()

    @Slot(np.ndarray, np.ndarray)
    def on_dimension_images_ready(self, xy_img: np.ndarray, z_img: np.ndarray):
        """Draws red coordinate lines on raw numpy image copies, converts, and passes to UI."""

        pix_xy = numpy_to_qpixmap(xy_img)
        pix_z = numpy_to_qpixmap(z_img)

        # --- Draw X & Y Lines on Image 1 ---
        painter_xy = QPainter(pix_xy)
        pen_xy = QPen(QColor(Qt.red), max(2, pix_xy.height() // 50))
        painter_xy.setPen(pen_xy)

        # Setup dynamic font for labels
        font_xy = painter_xy.font()
        font_xy.setPixelSize(max(12, pix_xy.height() // 15))
        font_xy.setBold(True)
        painter_xy.setFont(font_xy)

        mid_x_xy = pix_xy.width() // 2
        mid_y_xy = pix_xy.height() // 2

        # Draw X (Horizontal)
        painter_xy.drawLine(0, mid_y_xy, pix_xy.width(), mid_y_xy)
        painter_xy.drawText(pix_xy.width() - int(font_xy.pixelSize() * 1), mid_y_xy - 5, "X")

        # Draw Y (Vertical)
        painter_xy.drawLine(mid_x_xy, 0, mid_x_xy, pix_xy.height())
        painter_xy.drawText(mid_x_xy + 10, int(font_xy.pixelSize() * 1), "Y")

        painter_xy.end()

        # --- Draw Z Line on Image 2 ---
        painter_z = QPainter(pix_z)
        pen_z = QPen(QColor(Qt.red), max(2, pix_z.width() // 50))
        painter_z.setPen(pen_z)

        font_z = painter_z.font()
        font_z.setPixelSize(max(12, pix_z.height() // 15))
        font_z.setBold(True)
        painter_z.setFont(font_z)

        mid_x_z = pix_z.width() // 2

        # Draw Z (Vertical)
        painter_z.drawLine(mid_x_z, 0, mid_x_z, pix_z.height())
        painter_z.drawText(mid_x_z + 10, int(font_z.pixelSize() * 1), "Z")

        painter_z.end()

        # Hand them to our smart labels
        self.lbl_img_xy.set_image(pix_xy)
        self.lbl_img_z.set_image(pix_z)

    @Slot(dict)
    def on_new_data_received(self, data: dict):
        """
        Slot to receive new data and update the plots using strict XYZ coordinate keys.
        """
        print(f"Data received: {data.keys()}")

        frames = np.array(data.get('frames', []))

        # Strict fetches using only the new coordinate conventions
        x_data_mm = np.array(data.get('dim_x', []))
        y_data_mm = np.array(data.get('dim_y', []))
        z_data_mm = np.array(data.get('dim_z', []))
        area_data_mm2 = np.array(data.get('area', []))

        if len(frames) > 0:
            if len(y_data_mm) == len(frames):
                # Convert mm to m
                self.curve_y.setData(frames, y_data_mm * 1e-3)

            if len(z_data_mm) == len(frames):
                # Convert mm to m
                self.curve_z.setData(frames, z_data_mm * 1e-3)

            if len(x_data_mm) == len(frames):
                # Convert mm to m
                self.curve_x.setData(frames, x_data_mm * 1e-3)

            if len(area_data_mm2) == len(frames):
                # Convert mm² to m² (1e-6)
                self.curve_area.setData(frames, area_data_mm2 * 1e-6)

    @Slot()
    def on_open_folder_clicked(self):
        """Cross-platform method to open the output folder."""
        if self.output_dir and os.path.exists(self.output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))

    @Slot(str)
    def on_video_compiled(self, directory_path: str):
        """Enables the 'Open Folder' button and stores the path once background task finishes."""
        self.output_dir = directory_path
        self.btn_open_folder.setEnabled(True)
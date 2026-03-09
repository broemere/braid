import numpy as np
from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
)
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap
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
        self.setMinimumSize(100, 100) # Prevents it from collapsing to 0
        self._pixmap = None

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.setText("") # Clear the placeholder text
        self.update()    # Trigger a repaint

    def paintEvent(self, event):
        # If no image, fallback to standard QLabel behavior (draws the text)
        if self._pixmap is None or self._pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Calculate the scaled size keeping aspect ratio
        scaled_pix = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Center the image inside the label
        x_offset = (self.width() - scaled_pix.width()) // 2
        y_offset = (self.height() - scaled_pix.height()) // 2

        painter.drawPixmap(x_offset, y_offset, scaled_pix)


class GeometryTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline

        # Enable anti-aliasing for prettier, smoother plot lines
        pg.setConfigOptions(antialias=True)

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- Top Controls ---
        control_layout = QHBoxLayout()
        self.btn_get_geometry = QPushButton("Get Geometry")
        self.btn_get_geometry.setMinimumHeight(40)

        control_layout.addStretch()
        control_layout.addWidget(self.btn_get_geometry)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # --- Main Content Area (Left Column + Right Plots) ---
        content_layout = QHBoxLayout()

        # >> LEFT COLUMN: Images and Swap Button
        left_column = QVBoxLayout()

        # 1. Width Title & Image
        lbl_width_title = QLabel("Width Orientation")
        lbl_width_title.setAlignment(Qt.AlignCenter)
        lbl_width_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.lbl_img_width = ResizableImageLabel("Waiting for Width ROI...")

        # 2. Swap Button
        self.btn_swap = QPushButton("↕ Swap")
        self.btn_swap.setMinimumHeight(30)

        # 3. Thickness Title & Image
        lbl_thickness_title = QLabel("Thickness Orientation")
        lbl_thickness_title.setAlignment(Qt.AlignCenter)
        lbl_thickness_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.lbl_img_thickness = ResizableImageLabel("Waiting for Thickness ROI...")

        # Pack the left column
        left_column.addWidget(lbl_width_title)
        left_column.addWidget(self.lbl_img_width, stretch=1)
        left_column.addWidget(self.btn_swap)
        left_column.addWidget(lbl_thickness_title)
        left_column.addWidget(self.lbl_img_thickness, stretch=1)
        # Notice: No addStretch() here! The stretch=1 on the images handles expansion.

        content_layout.addLayout(left_column, stretch=1)

        # >> RIGHT COLUMN: Plotting Area
        self.plot_widget = pg.GraphicsLayoutWidget()
        content_layout.addWidget(self.plot_widget, stretch=4)

        layout.addLayout(content_layout, stretch=1)

        # --- Setup Plots ---
        self.plot_width = self.plot_widget.addPlot(title="Object Width")
        self.plot_width.setLabel('left', 'Width', units='mm')
        self.curve_width = self.plot_width.plot(pen=pg.mkPen(color='#00d2ff', width=2))

        self.plot_widget.nextRow()

        self.plot_thickness = self.plot_widget.addPlot(title="Object Thickness")
        self.plot_thickness.setLabel('left', 'Thickness', units='mm')
        self.curve_thickness = self.plot_thickness.plot(pen=pg.mkPen(color='#ff007f', width=2))

        self.plot_widget.nextRow()

        self.plot_area = self.plot_widget.addPlot(title="Cross Sectional Area")
        self.plot_area.setLabel('left', 'Area', units='mm²')
        self.plot_area.setLabel('bottom', 'Frame Index')
        self.curve_area = self.plot_area.plot(pen=pg.mkPen(color='#00ff00', width=2))

        self.plot_thickness.setXLink(self.plot_width)
        self.plot_area.setXLink(self.plot_width)

    def connect_signals(self):
        # UI -> Pipeline
        self.btn_get_geometry.clicked.connect(self.pipeline.get_geometry)
        self.btn_swap.clicked.connect(self.pipeline.swap_dimensions)  # Hook up the swap!

        # Pipeline -> UI
        self.pipeline.geometry_available.connect(self.on_new_data_received)
        self.pipeline.dimension_images_ready.connect(self.on_dimension_images_ready)

    def showEvent(self, event):
        super().showEvent(event)
        # Ask the pipeline for the current state images whenever this tab is viewed
        self.pipeline.request_dimension_images()

    @Slot(np.ndarray, np.ndarray)
    def on_dimension_images_ready(self, w_img: np.ndarray, t_img: np.ndarray):
        """Draws red lines on raw numpy image copies, converts, and passes to UI."""

        # Convert raw numpy arrays to QPixmaps
        pix_w = numpy_to_qpixmap(w_img)
        pix_t = numpy_to_qpixmap(t_img)

        # Draw on the native resolution pixmap BEFORE handing it to the custom label
        painter_w = QPainter(pix_w)
        pen_w = QPen(QColor(Qt.red), max(2, pix_w.height() // 50))
        painter_w.setPen(pen_w)
        mid_y = pix_w.height() // 2
        painter_w.drawLine(0, mid_y, pix_w.width(), mid_y)
        painter_w.end()

        painter_t = QPainter(pix_t)
        pen_t = QPen(QColor(Qt.red), max(2, pix_t.width() // 50))
        painter_t.setPen(pen_t)
        mid_x = pix_t.width() // 2
        painter_t.drawLine(mid_x, 0, mid_x, pix_t.height())
        painter_t.end()

        # Hand them to our new smart labels
        self.lbl_img_width.set_image(pix_w)
        self.lbl_img_thickness.set_image(pix_t)

    @Slot(dict)
    def on_new_data_received(self, data: dict):
        """
        Slot to receive new data and update the plots.
        Expected data format:
        {
            'frames': [0, 1, 2, 3...],
            'width': [100, 102, 105...],
            'length': [200, 210, 220...],
            'area': [20000, 21420, 23100...]
        }
        """
        print(f"Data received: {data.keys()}")

        # Extract the arrays safely
        frames = np.array(data.get('frames', []))
        width_data = np.array(data.get('width', []))
        thickness_data = np.array(data.get('thickness', []))
        area_data = np.array(data.get('area', []))

        # Check if we have matching X and Y data before plotting
        if len(frames) > 0:
            if len(width_data) == len(frames):
                self.curve_width.setData(frames, width_data)

            if len(thickness_data) == len(frames):
                self.curve_thickness.setData(frames, thickness_data)

            if len(area_data) == len(frames):
                self.curve_area.setData(frames, area_data)
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSizePolicy, QBoxLayout
)
from PySide6.QtGui import QPixmap, QPainter
from processing.data_transform import numpy_to_qpixmap
from data_pipeline import DataPipeline


class ImageDisplayWidget(QWidget):
    """A passive canvas that scales an image to fit its layout space without fighting it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        # Allow it to expand perfectly equally
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Low minimum size ensures it never forces the window to grow
        self.setMinimumSize(50, 50)

    def set_pixmap(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        if not self.pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background if desired (matches your previous stylesheet)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        # Calculate the scaled size keeping aspect ratio
        scaled_pixmap = self.pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Calculate coordinates to center the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2

        # Draw the image
        painter.drawPixmap(x, y, scaled_pixmap)


class ThreshTab(QWidget):
    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.image_widgets = []  # Changed from labels to our custom widgets
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- Image Display Row ---
        #self.img_row = QHBoxLayout()

        self.img_row = QBoxLayout(QBoxLayout.Direction.TopToBottom)

        for i in range(4):
            # Use our custom widget instead of QLabel
            widget = ImageDisplayWidget()
            self.image_widgets.append(widget)

            # stretch=1 now works perfectly because ImageDisplayWidget
            # doesn't have an unpredictable sizeHint skewing the math.
            self.img_row.addWidget(widget, stretch=1)

        layout.addLayout(self.img_row, stretch=1)

        # --- Control Section ---
        control_layout = QVBoxLayout()

        # 1. Edge Smoothness Slider (Controls Mu)
        smoothness_layout = QHBoxLayout()
        self.smoothness_label = QLabel("Edge Smoothness: 50%")
        self.smoothness_label.setMinimumWidth(150)

        self.smoothness_slider = QSlider(Qt.Horizontal)
        self.smoothness_slider.setRange(0, 100)
        self.smoothness_slider.setValue(50)

        smoothness_layout.addWidget(self.smoothness_label)
        smoothness_layout.addWidget(self.smoothness_slider)

        # 2. Shadow Recovery Slider (Controls Gamma & Lambda1)
        shadow_layout = QHBoxLayout()
        self.shadow_label = QLabel("Shadow Recovery: 0%")
        self.shadow_label.setMinimumWidth(150)

        self.shadow_slider = QSlider(Qt.Horizontal)
        self.shadow_slider.setRange(0, 100)
        self.shadow_slider.setValue(0)  # Default to 0 (flat lighting)

        shadow_layout.addWidget(self.shadow_label)
        shadow_layout.addWidget(self.shadow_slider)

        control_layout.addLayout(smoothness_layout)
        control_layout.addLayout(shadow_layout)
        layout.addLayout(control_layout)

    def showEvent(self, event):
        super().showEvent(event)
        # Trigger an initial calculation when the tab is opened
        if self.pipeline.layout == "vertical" and self.img_row.direction() == QBoxLayout.Direction.LeftToRight:
            self.img_row.setDirection(QBoxLayout.Direction.TopToBottom)
        if self.pipeline.layout == "horizontal" and self.img_row.direction() == QBoxLayout.Direction.TopToBottom:
            self.img_row.setDirection(QBoxLayout.Direction.LeftToRight)
        self._on_sliders_moved()


    def connect_signals(self):
        # UI -> Pipeline (Both sliders trigger the same function)
        self.smoothness_slider.valueChanged.connect(self._on_sliders_moved)
        self.shadow_slider.valueChanged.connect(self._on_sliders_moved)

        # Pipeline -> UI
        self.pipeline.threshed_images_ready.connect(self.update_displays)

    def _on_sliders_moved(self, _=None):
        """Grabs both values, updates UI text, and sends to pipeline."""
        smooth_val = self.smoothness_slider.value()
        shadow_val = self.shadow_slider.value()

        # Update the friendly UI labels
        self.smoothness_label.setText(f"Edge Smoothness: {smooth_val}%")
        self.shadow_label.setText(f"Shadow Recovery: {shadow_val}%")

        # Tell the pipeline to process current crops with both values
        self.pipeline.apply_threshold(smooth_val, shadow_val)

    # @Slot(list)
    # def update_displays(self, threshed_pixmaps):
    #     """Receives a list of 4 QPixmaps from the pipeline."""
    #     for label, pixmap in zip(self.image_labels, threshed_pixmaps):
    #         # Scale to fit label while keeping aspect ratio
    #         scaled_pix = pixmap.scaled(
    #             label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
    #         )
    #         label.setPixmap(scaled_pix)

    @Slot(list)
    def update_displays(self, threshed_pixmaps):
        """Receives a list of 4 QPixmaps from the pipeline."""
        for widget, pixmap in zip(self.image_widgets, threshed_pixmaps):
            # We no longer manually scale here! We just hand the pixmap
            # to the widget, and it handles the scaling in real-time.
            widget.set_pixmap(pixmap)
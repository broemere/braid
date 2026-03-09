from PySide6.QtCore import Qt, Signal, QPointF, QRect, QRectF
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen, QCursor


class ROICanvas(QWidget):
    """
    A canvas for selecting Regions of Interest (ROIs).
    - Allows exactly two boxes to be drawn.
    - Drawing a third box replaces the first one (FIFO).
    - Supports Undo (Ctrl+Z) and Reset.
    - Emits all current ROI coordinates whenever they change.
    """

    # Emits a list of QRects representing the boxes in image-space
    roi_updated = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix_full: QPixmap | None = None
        self.rois: list[QRect] = []

        self.current_start_pt: QPointF | None = None
        self._mouse_pos: QPointF | None = None

        self.active_color = QColor(Qt.red)
        self.completed_color = QColor(Qt.green)

        # We cache these to make coordinate conversion fast and accurate
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._scale_factor = 1.0
        self._scaled_w = 0
        self._scaled_h = 0

        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ——————————————
    # Public API

    def set_background(self, pixmap: QPixmap):
        """Load a new image and clear existing ROIs."""
        self._pix_full = pixmap
        self.reset_rois()

    def reset_rois(self):
        """Clear all boxes."""
        self.rois = []
        self.current_start_pt = None
        self.update()
        self.roi_updated.emit(self.rois)

    def undo_last_roi(self):
        """Remove the most recently added box (LIFO)."""
        if self.rois:
            self.rois.pop()
            self.update()
            self.roi_updated.emit(self.rois)

    # ——————————————
    # Event Handlers

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._pix_full is None:
            return

        img_pt = self._widget_to_image(event.position())
        if img_pt is None:
            return

        if self.current_start_pt is None:
            self.current_start_pt = img_pt
        else:
            # Finalize the second point
            new_roi = QRect(
                self.current_start_pt.toPoint(),
                img_pt.toPoint()
            ).normalized()

            if len(self.rois) >= 2:
                self.rois.pop(0)

            self.rois.append(new_roi)
            self.current_start_pt = None
            self.roi_updated.emit(self.rois)

        self.update()

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.position()
        if self.current_start_pt:
            self.update()

    def keyPressEvent(self, event):
        # Ctrl + Z for Undo
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_last_roi()

        # Escape to cancel current drawing
        elif event.key() == Qt.Key_Escape:
            if self.current_start_pt:
                self.current_start_pt = None
                self.update()

        super().keyPressEvent(event)

    # ——————————————
    # Painting

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._pix_full is None: return

        # 1. Update Scaling Math
        # We calculate this here so it's always fresh if the window resized
        w, h = self.width(), self.height()
        pix_w, pix_h = self._pix_full.width(), self._pix_full.height()

        # Calculate how the image fits (KeepAspectRatio)
        self._scale_factor = min(w / pix_w, h / pix_h)
        self._scaled_w = int(pix_w * self._scale_factor)
        self._scaled_h = int(pix_h * self._scale_factor)
        self._offset_x = (w - self._scaled_w) / 2
        self._offset_y = (h - self._scaled_h) / 2

        # 2. Draw Background
        painter.drawPixmap(
            int(self._offset_x), int(self._offset_y),
            self._scaled_w, self._scaled_h,
            self._pix_full
        )

        # 3. Draw ROIs
        for roi in self.rois:
            painter.setPen(QPen(self.completed_color, 2))
            painter.drawRect(self._image_rect_to_widget(roi))

        # 4. Draw Preview
        if self.current_start_pt and self._mouse_pos:
            p0 = self._image_to_widget(self.current_start_pt)
            p1 = self._mouse_pos
            rect = QRect(p0.toPoint(), p1.toPoint()).normalized()
            painter.setPen(QPen(self.active_color, 1, Qt.DashLine))
            painter.drawRect(rect)

    # ——————————————
    # Coordinate Helpers

    def _widget_to_image(self, pt: QPointF) -> QPointF | None:
        if self._pix_full is None: return None
        # Subtract offset, then divide by scale to "get back" to original pixels
        x = (pt.x() - self._offset_x) / self._scale_factor
        y = (pt.y() - self._offset_y) / self._scale_factor

        # Clamp to ensure coordinates stay within the actual image pixels
        return QPointF(
            max(0, min(x, self._pix_full.width())),
            max(0, min(y, self._pix_full.height()))
        )

    def _image_to_widget(self, pt: QPointF) -> QPointF:
        return QPointF(
            pt.x() * self._scale_factor + self._offset_x,
            pt.y() * self._scale_factor + self._offset_y
        )

    def _image_rect_to_widget(self, rect: QRect) -> QRect:
        tl = self._image_to_widget(QPointF(rect.topLeft()))
        br = self._image_to_widget(QPointF(rect.bottomRight()))
        return QRect(tl.toPoint(), br.toPoint())
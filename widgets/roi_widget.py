from PySide6.QtCore import Qt, Signal, QPointF, QRect, QRectF
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen


class ROICanvas(QGraphicsView):
    """
    A zoomable canvas for selecting Regions of Interest (ROIs).
    - Allows exactly two boxes to be drawn.
    - Click once to start a box, click again to finish.
    - Drawing a third box replaces the first one (FIFO).
    - Supports Zoom (Ctrl+Scroll / +/-) and Pan (Shift+Scroll).
    - Supports Undo (Ctrl+Z) and Cancel current draw (Escape).
    - Emits a list of QRects in true image coordinates.
    """

    # Emits a list of QRects representing the boxes in image-space
    roi_updated = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Scene and View Setup ---
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setBackgroundBrush(QColor("#222222"))

        # --- State Variables ---
        self._image_item = None
        self.rois_items: list[QGraphicsRectItem] = []

        self.current_start_pt: QPointF | None = None
        self._preview_rect_item: QGraphicsRectItem | None = None

        # --- Drawing Styles ---
        self.active_pen = QPen(QColor(Qt.red), 1, Qt.DashLine)
        self.active_pen.setCosmetic(True)  # Prevents line scaling when zooming

        self.completed_pen = QPen(QColor(Qt.green), 2)
        self.completed_pen.setCosmetic(True)

    # ——————————————
    # Public API

    def set_background(self, pixmap: QPixmap):
        """Load a new image, clear existing ROIs, and reset view."""
        self._scene.clear()
        self.rois_items.clear()
        self.current_start_pt = None
        self._preview_rect_item = None

        self._image_item = self._scene.addPixmap(pixmap)
        self.reset_view()
        self.roi_updated.emit([])

    def reset_view(self):
        """Resets the view to fit the entire image within the viewport."""
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def reset_rois(self):
        """Clear all drawn boxes."""
        for item in self.rois_items:
            self._scene.removeItem(item)
        self.rois_items.clear()
        self._cancel_preview()
        self.roi_updated.emit([])

    def undo_last_roi(self):
        """Remove the most recently added box (LIFO)."""
        if self.rois_items:
            item = self.rois_items.pop()
            self._scene.removeItem(item)
            self._emit_rois()

    def _cancel_preview(self):
        """Cancels a drawing in progress."""
        if self._preview_rect_item:
            self._scene.removeItem(self._preview_rect_item)
            self._preview_rect_item = None
        self.current_start_pt = None

    def _emit_rois(self):
        """Converts scene items to standard QRects and emits them."""
        # Convert QRectF bounds to standard integer QRects
        rects = [item.rect().toRect() for item in self.rois_items]
        self.roi_updated.emit(rects)

    # ——————————————
    # Internal View Math

    def _clamp_to_image(self, pos: QPointF) -> QPointF:
        """Forces coordinates to stay strictly within image boundaries."""
        if not self._image_item:
            return pos
        rect = self._image_item.boundingRect()
        x = max(rect.left(), min(pos.x(), rect.right()))
        y = max(rect.top(), min(pos.y(), rect.bottom()))
        return QPointF(x, y)

    def _zoom(self, factor):
        """Applies a zoom factor, centered on the mouse cursor."""
        if self._image_item is None:
            return
        if factor < 1.0:
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            if h_bar.maximum() <= 0 and v_bar.maximum() <= 0:
                return
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)

    # ——————————————
    # Mouse & Key Events

    def wheelEvent(self, event):
        """Handles zooming and panning via the mouse wheel."""
        if self._image_item is None:
            return

        angle = event.angleDelta().y()

        if event.modifiers() == Qt.ControlModifier:
            if angle > 0:
                self._zoom(1.15)
            else:
                self._zoom(1 / 1.15)
        elif event.modifiers() == Qt.ShiftModifier:
            h_bar = self.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - angle)
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Resets the view on double-click."""
        if event.button() == Qt.LeftButton:
            self.reset_view()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Handles the click-to-start, click-to-finish drawing logic."""
        if event.button() != Qt.LeftButton or self._image_item is None:
            super().mousePressEvent(event)
            return

        # 1. Map viewport click to native image coordinates
        scene_pos = self.mapToScene(event.pos())
        img_pt = self._clamp_to_image(scene_pos)

        # 2. First click: Start the preview rectangle
        if self.current_start_pt is None:
            self.current_start_pt = img_pt
            self._preview_rect_item = QGraphicsRectItem(QRectF(img_pt, img_pt))
            self._preview_rect_item.setPen(self.active_pen)
            self._scene.addItem(self._preview_rect_item)

        # 3. Second click: Finalize the rectangle
        else:
            final_rect = QRectF(self.current_start_pt, img_pt).normalized()

            # Remove preview item
            self._cancel_preview()

            # Create the final green rectangle
            rect_item = QGraphicsRectItem(final_rect)
            rect_item.setPen(self.completed_pen)
            self._scene.addItem(rect_item)

            # FIFO logic: Ensure max 2 rectangles
            if len(self.rois_items) >= 2:
                oldest = self.rois_items.pop(0)
                self._scene.removeItem(oldest)

            self.rois_items.append(rect_item)
            self._emit_rois()

    def mouseMoveEvent(self, event):
        """Updates the preview dashed box as the user moves the mouse."""
        if self.current_start_pt and self._preview_rect_item:
            scene_pos = self.mapToScene(event.pos())
            img_pt = self._clamp_to_image(scene_pos)

            # .normalized() allows drawing in any direction safely
            new_rect = QRectF(self.current_start_pt, img_pt).normalized()
            self._preview_rect_item.setRect(new_rect)

        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """Handles Undo, Cancel, and Keyboard Zoom shortcuts."""
        # Undo (Ctrl+Z)
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_last_roi()
            event.accept()

        # Cancel current draw
        elif event.key() == Qt.Key_Escape:
            self._cancel_preview()
            event.accept()

        # Keyboard Zoom (+ / -)
        elif event.key() in (Qt.Key_Equal, Qt.Key_Plus):
            self._zoom(1.5)
            event.accept()
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom(1 / 1.5)
            event.accept()

        else:
            super().keyPressEvent(event)
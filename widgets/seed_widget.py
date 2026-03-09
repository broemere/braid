from PySide6.QtCore import Qt, Signal, QPoint, QRect, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import QWidget, QSizePolicy


class SeedDrawingLabel(QWidget):
    # Signal emits: (shape_type, coords_dict)
    shape_drawn = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.original_pixmap = None

        # --- State Separation Fix ---
        self.current_tool = 'rect'  # What the user has selected in the UI
        self.drawn_shape_type = None  # What is actually drawn on the image

        # Drawing State
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.has_shape = False

        # Display Geometry
        self.draw_rect = QRect()
        self.scale_factor = 1.0

    def set_pixmap(self, pixmap):
        if pixmap is None:
            self.original_pixmap = None
            self.update()
            return

        if not isinstance(pixmap, QPixmap):
            # Pass silently or log error, but don't crash
            return

        self.original_pixmap = pixmap
        self.update()

    def set_tool(self, tool_type):
        """ Updates the active tool, but DOES NOT change existing shapes. """
        self.current_tool = tool_type
        # Note: We do NOT call update() here. Changing the tool shouldn't
        # visually change anything until the user clicks to draw.

    def undo(self):
        self.has_shape = False
        self.start_point = None
        self.end_point = None
        self.drawn_shape_type = None  # Reset the drawn type
        self.update()
        self.shape_drawn.emit(None, {})

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Draw Background
        painter.fillRect(self.rect(), QColor("#222"))

        if self.original_pixmap is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for image...")
            return

        # 2. Calculate Geometry
        w_widget = self.width()
        h_widget = self.height()
        w_img = self.original_pixmap.width()
        h_img = self.original_pixmap.height()

        self.scale_factor = min(w_widget / w_img, h_widget / h_img)
        w_draw = int(w_img * self.scale_factor)
        h_draw = int(h_img * self.scale_factor)

        x_off = (w_widget - w_draw) // 2
        y_off = (h_widget - h_draw) // 2
        self.draw_rect = QRect(x_off, y_off, w_draw, h_draw)

        painter.drawPixmap(
            self.draw_rect.x(), self.draw_rect.y(),
            self.draw_rect.width(), self.draw_rect.height(),
            self.original_pixmap
        )

        # 3. Draw Seed Shape
        # FIX: We check self.drawn_shape_type, NOT self.current_tool
        if (self.is_drawing or self.has_shape) and self.start_point and self.end_point and self.drawn_shape_type:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QColor(0, 255, 0, 100))

            rect = QRect(self.start_point, self.end_point).normalized()

            if self.drawn_shape_type == 'rect':
                painter.drawRect(rect)
            elif self.drawn_shape_type == 'ellipse':
                painter.drawEllipse(rect)

    def mousePressEvent(self, event):
        if not self.original_pixmap: return

        if self.draw_rect.contains(event.pos()):
            # FIX: Lock in the tool type at the moment drawing starts
            self.drawn_shape_type = self.current_tool

            self.start_point = event.pos()
            self.end_point = event.pos()
            self.is_drawing = True
            self.has_shape = False
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            x = max(self.draw_rect.left(), min(event.pos().x(), self.draw_rect.right()))
            y = max(self.draw_rect.top(), min(event.pos().y(), self.draw_rect.bottom()))
            self.end_point = QPoint(x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            self.is_drawing = False
            self.has_shape = True
            self.update()
            self._emit_shape_data()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo()

    def _emit_shape_data(self):
        if not self.start_point or not self.end_point: return

        ui_rect = QRect(self.start_point, self.end_point).normalized()

        rel_x = ui_rect.x() - self.draw_rect.x()
        rel_y = ui_rect.y() - self.draw_rect.y()
        rel_w = ui_rect.width()
        rel_h = ui_rect.height()

        img_x = int(rel_x / self.scale_factor)
        img_y = int(rel_y / self.scale_factor)
        img_w = int(rel_w / self.scale_factor)
        img_h = int(rel_h / self.scale_factor)

        data = {}
        # FIX: Use drawn_shape_type here as well
        if self.drawn_shape_type == 'rect':
            data = {'x': img_x, 'y': img_y, 'w': img_w, 'h': img_h}
        elif self.drawn_shape_type == 'ellipse':
            radius_x = img_w // 2
            radius_y = img_h // 2
            center_x = img_x + radius_x
            center_y = img_y + radius_y
            data = {
                'center_x': center_x,
                'center_y': center_y,
                'radius_x': radius_x,
                'radius_y': radius_y
            }

        self.shape_drawn.emit(self.drawn_shape_type, data)
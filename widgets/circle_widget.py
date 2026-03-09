import hashlib
from PySide6.QtCore import QRectF
from PySide6.QtGui import Qt, QIcon, QPixmap, QPainter, QColor
from processing.data_loader import load_colors

colors = load_colors()


def make_circle_icon(color: str | QColor, diameter: int = 8) -> QIcon:
    if isinstance(color, str):
        color = QColor(color)
    px = QPixmap(diameter, diameter)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing, False)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    # draw slightly inset to avoid antialias clipping at the edges
    p.drawEllipse(QRectF(0.5, 0.5, diameter - 1, diameter - 1))
    p.end()
    return QIcon(px)


def get_color(data):
    n_colors = len(colors)
    encoded_data = data.encode('utf-8')
    sha256_hash = hashlib.sha256(encoded_data)
    hash_int = int(sha256_hash.hexdigest(), 16)
    index = hash_int % n_colors
    hex_color = colors[index]
    return hex_color

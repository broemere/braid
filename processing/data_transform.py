import numpy as np
import cv2
from PySide6.QtGui import QImage, QPixmap
import logging
log = logging.getLogger(__name__)

def numpy_to_qpixmap(numpy_array: np.ndarray) -> QPixmap:
    """
    Converts a NumPy array to a QPixmap.

    Handles both grayscale (2D) and color (3D) images.
    Assumes color images from OpenCV are in BGR format.
    """
    if numpy_array is None:
        return QPixmap()  # Return an empty pixmap if the array is null

    # If this is a slice (view), this forces a copy into a new contiguous block.
    if not numpy_array.flags['C_CONTIGUOUS']:
        numpy_array = np.ascontiguousarray(numpy_array)

    height, width = numpy_array.shape[:2]
    bytes_per_line = numpy_array.strides[0]

    # --- Determine the QImage format ---
    if numpy_array.ndim == 2:
        # Grayscale image
        q_image_format = QImage.Format_Grayscale8
    elif numpy_array.ndim == 3:
        # Color image
        if numpy_array.shape[2] == 4:
            # RGBA format
            q_image_format = QImage.Format_RGBA8888
        else:
            # Standard 3-channel color. OpenCV uses BGR, but Qt needs RGB.
            # We must convert it.
            numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
            q_image_format = QImage.Format_RGB888
    else:
        # Unsupported format
        return QPixmap()

    # --- Create QImage from the NumPy array's memory buffer ---
    q_image = QImage(numpy_array.data, width, height, bytes_per_line, q_image_format)

    # QImage might hold a reference to the numpy array. To be safe,
    # copy it before returning, so the array can be garbage collected.
    return QPixmap.fromImage(q_image.copy())


def auto_thresh(signals, images):

    threshed_images = []

    signals.message.emit("Thresholding image...")

    total = len(images)*255
    count = 0

    for i, img in enumerate(images):
        threshed_images.append([])
        for th in range(255):
            _, thresh1 = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
            threshed_images[i].append(thresh1)

            count += 1
            pct = int((count / total) * 100)
            signals.progress.emit(pct)

    return threshed_images
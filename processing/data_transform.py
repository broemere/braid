import ast
import logging
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

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

def serialize_objects(obj):
    """
    Recursively convert numpy arrays and scalars into JSON-serializable forms:
      - ndarray → dict with keys __ndarray__, dtype, shape, data (as nested lists)
      - numpy scalar → native Python type via .item()
    """
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": obj.shape
        }
    elif isinstance(obj, np.generic):
        # covers np.int32, np.float64, etc.
        return obj.item()
    elif isinstance(obj, Path):
        # Convert Path objects to a special dictionary format
        return {
            "__path__": str(obj)
        }
    elif isinstance(obj, OrderedDict):
        # CRITICAL: This must come before the standard dict check!
        # We save it as a list of pairs to guarantee order preservation in JSON
        return {
            "__OrderedDict__": [[k, serialize_objects(v)] for k, v in obj.items()]
        }
    elif isinstance(obj, set):
        return {
            "__set__": [serialize_objects(v) for v in obj]
        }
    elif isinstance(obj, dict):
        return {k: serialize_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_objects(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_objects(v) for v in obj)
    else:
        return obj


def deserialize_objects(obj):
    """
    Recursively walk obj and convert back to numpy arrays when possible:
      - If obj is a dict with "__ndarray__", build an array with the right dtype & shape.
      - If obj is a nested list of numbers (1D/2D/3D), cast it to an ndarray.
      - Otherwise, recurse into dicts & lists; leave other types untouched.
    """
    # 1) Reverse of our special-encoded ndarray
    if isinstance(obj, dict) and "__ndarray__" in obj:
        dtype_value = obj.get("dtype", None)
        try:
            dtype = np.dtype(dtype_value)
        except TypeError:
            # Legacy BRAID sessions store structured dtypes as their repr string.
            # literal_eval safely recovers the list of field descriptors.
            dtype = np.dtype(ast.literal_eval(dtype_value))

        array_data = obj["__ndarray__"]
        if dtype.names and isinstance(array_data, list):
            # JSON converts each structured record tuple into a list. NumPy needs
            # the records converted back to tuples to preserve the 1D table shape.
            array_data = [tuple(record) for record in array_data]

        arr = np.array(array_data, dtype=dtype)
        if "shape" in obj:
            arr = arr.reshape(obj["shape"])
        return arr

    # 2) Reverse of encoded Path
    if isinstance(obj, dict) and "__path__" in obj:
        return Path(obj["__path__"])

    # 3) Reverse of encoded OrderedDict
    if isinstance(obj, dict) and "__OrderedDict__" in obj:
        # Rebuild the OrderedDict from the list of [key, value] pairs
        return OrderedDict([(k, deserialize_objects(v)) for k, v in obj["__OrderedDict__"]])

    # 4) Reverse of encoded set
    if isinstance(obj, dict) and "__set__" in obj:
        # Rebuild the set from the list
        return set(deserialize_objects(v) for v in obj["__set__"])

    # 5) Try to turn pure numeric lists into arrays
    if isinstance(obj, list):
        try:
            arr = np.array(obj)
            # only accept if it really is numeric and 1–3 dimensional
            if arr.dtype.kind in ("i","u","f") and 1 <= arr.ndim <= 3:
                return arr
        except Exception:
            pass
        # otherwise, recurse into each element
        return [deserialize_objects(v) for v in obj]

    # 6) Recurse into plain dicts
    if isinstance(obj, dict):
        return {k: deserialize_objects(v) for k, v in obj.items()}

    # 7) Leave everything else alone
    return obj

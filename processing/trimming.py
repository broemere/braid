import os

import numpy as np
from tifffile import TiffFile


def select_time_range(data: np.ndarray, start_time: float, end_time: float):
    """Return the inclusive time-range selection and its source row indices."""
    if data.size == 0 or "time_s" not in (data.dtype.names or ()):
        return np.array([], dtype=data.dtype), np.array([], dtype=np.int64)

    mask = (data["time_s"] >= start_time) & (data["time_s"] <= end_time)
    source_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    return data[mask], source_indices


def map_distance_extrema_to_source(data: np.ndarray, source_indices):
    """Return local distance extrema and their corresponding source row indices."""
    indices = np.asarray(source_indices, dtype=np.int64)
    if data.size == 0:
        raise ValueError("Cannot calculate extrema for an empty active dataset.")
    if "distance" not in (data.dtype.names or ()):
        raise ValueError("Active data does not contain a distance column.")
    if len(data) != len(indices):
        raise ValueError("Active data and source-frame indices must have equal lengths.")

    max_data_index = int(np.argmax(data["distance"]))
    min_data_index = int(np.argmin(data["distance"]))
    return (
        max_data_index,
        min_data_index,
        int(indices[max_data_index]),
        int(indices[min_data_index]),
    )


def iter_selected_frames(file_path: str, frame_indices):
    """Yield source video frames in the requested order."""
    indices = np.asarray(frame_indices, dtype=np.int64)
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.tif', '.tiff']:
        with TiffFile(file_path) as tif:
            frame_count = len(tif.pages)
            for idx in indices:
                source_idx = int(idx)
                if source_idx < 0 or source_idx >= frame_count:
                    raise IndexError(
                        f"Source frame {source_idx} is outside TIFF frame range 0..{frame_count - 1}."
                    )
                yield tif.pages[source_idx].asarray()
        return

    if file_ext == '.mkv':
        # Keep the pure trimming helpers importable in lightweight test environments.
        import cv2

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Worker failed to open video file: {file_path}")

        try:
            if indices.size == 0:
                return

            is_contiguous = indices.size == 1 or np.all(np.diff(indices) == 1)
            if is_contiguous:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
                for source_idx in indices:
                    ret, frame_data = cap.read()
                    if not ret:
                        raise RuntimeError(
                            f"Video ended before source frame {int(source_idx)} could be read."
                        )
                    yield frame_data
            else:
                for source_idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_idx))
                    ret, frame_data = cap.read()
                    if not ret:
                        raise RuntimeError(f"Could not read source frame {int(source_idx)}.")
                    yield frame_data
        finally:
            cap.release()
        return

    raise ValueError(f"Unsupported geometry file format: {file_ext}")

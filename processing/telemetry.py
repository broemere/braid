import numpy as np


REQUIRED_ANALYSIS_COLUMNS = ("time_s", "distance", "force", "cycle")
OPTIONAL_FRAME_COLUMNS = ("frameIdx", "frame_index")


def build_analysis_records(data: dict) -> np.ndarray:
    """Build numeric plot/analysis records while ignoring auxiliary metadata."""
    if not data:
        raise ValueError("Telemetry data is empty.")

    missing = [key for key in REQUIRED_ANALYSIS_COLUMNS if key not in data]
    if missing:
        raise ValueError(f"Telemetry data is missing required columns: {', '.join(missing)}")

    selected_keys = list(REQUIRED_ANALYSIS_COLUMNS)
    selected_keys.extend(key for key in OPTIONAL_FRAME_COLUMNS if key in data)

    converted_columns = {}
    expected_length = None
    dtype = []

    for key in selected_keys:
        column_dtype = np.int32 if key == "cycle" else np.float64
        try:
            column = np.asarray(data[key], dtype=column_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Telemetry column '{key}' must contain numeric values.") from exc

        if column.ndim != 1:
            raise ValueError(f"Telemetry column '{key}' must be one-dimensional.")

        if expected_length is None:
            expected_length = len(column)
        elif len(column) != expected_length:
            raise ValueError("Telemetry columns must all contain the same number of samples.")

        converted_columns[key] = column
        dtype.append((key, 'i4' if key == "cycle" else 'f8'))

    if expected_length == 0:
        raise ValueError("Telemetry data does not contain any samples.")

    records = np.empty(expected_length, dtype=dtype)
    for key, column in converted_columns.items():
        records[key] = column

    return records

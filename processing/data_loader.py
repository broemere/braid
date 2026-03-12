import os
import cv2
import json
import logging
import getpass
import numpy as np
from tifffile import TiffFile
from skimage import img_as_float
from skimage.segmentation import chan_vese
from skimage.draw import rectangle, ellipse
from PySide6.QtCore import QRect

log = logging.getLogger(__name__)


def get_system_username():
    """Returns the current system user name in a cross-platform way."""
    try:
        return getpass.getuser()
    except Exception:
        # Fallbacks: Windows, Unix, etc.
        return os.environ.get('USERNAME') or os.environ.get('USER') or os.environ.get('LOGNAME') or None


def load_colors():
    with open("resources/colors.json", 'r') as f:
        colors = json.load(f)
    return colors


def frame_loader(signals, file_path, frame_indices, count=False):
    """
    Loads specific frames from a video or multi-page TIFF file, handling errors gracefully.

    This function checks the file extension to determine the loading method. For TIFF files,
    it uses the tifffile library to directly access frames by index. For all other file
    types, it uses OpenCV's VideoCapture.
    """
    loaded_frames = {}
    file_ext = os.path.splitext(file_path)[1].lower()

    # --- TIFF File Handling ---
    if file_ext in ['.tif', '.tiff']:
        try:
            with TiffFile(file_path) as tif:
                frame_count = len(tif.pages)
                signals.message.emit("Collecting TIFF image data...")
                log.info(f"Starting frame extraction for {len(frame_indices)} frames from {file_path}: {frame_indices}")

                for i, f in enumerate(frame_indices):
                    try:
                        if f >= frame_count:
                            log.warning(
                                f"Frame index {f} is out of bounds for TIFF with {frame_count} pages. Skipping.")
                            continue

                        frame = tif.pages[f].asarray()

                        # Ensure the frame is grayscale for consistent processing
                        if frame.ndim == 3:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Handles RGB or multi-channel images
                        else:
                            gray = frame.copy()  # It's already grayscale, create a copy

                        # Normalize to 8-bit uint for compatibility with downstream processes
                        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
                        gray = gray.astype(np.uint8)

                        # Embed frame count into the first row of pixels
                        if i + len(str(frame_count)) < gray.shape[1]:
                            for j, digit in enumerate(str(frame_count)):
                                gray[0, i + j] = int(digit)
                        else:
                            log.warning(f"Frame {f}: Not enough horizontal pixels to write metadata.")

                        loaded_frames[f] = gray

                        pct = int(((i + 1) / len(frame_indices)) * 100)
                        signals.progress.emit(pct)

                    except Exception as e:
                        # Log error for a single frame and continue with the next
                        log.error(f"Error processing TIFF frame at index {f}: {e}", exc_info=True)
                        signals.message.emit(f"Error on TIFF frame {f}, see log for details.")

                if count:
                    loaded_frames[frame_count] = None

                    embedded_data = {}

                    for i, pg in enumerate(tif.pages):
                        try:
                            desc = pg.tags.get("ImageDescription")
                            if not desc:
                                log.debug(f"No deviceTime in first page, stopping search.")
                                continue
                            try:
                                info = json.loads(desc.value)
                            except Exception:
                                continue
                            #t = info.get("frameIdx")

                            keys = list(info.keys())

                            for k in keys:
                                if k not in embedded_data:
                                    embedded_data[k] = []
                                embedded_data[k].append(info[k])


                            # "time_s"
                            # "frameIdx"
                            # "distance"
                            # "cycle"
                            # "force"

                            #if isinstance(t, (int, float)):
                            #frames.append(t)

                            pct = int(((i + 1) / frame_count) * 100)
                            signals.progress.emit(pct)

                        except Exception as e:
                            # Log error for a single frame and continue with the next
                            log.error(f"Error processing TIFF frame at index {i}: {e}", exc_info=True)
                            signals.message.emit(f"Error on TIFF frame {i}, see log for details.")

                    loaded_frames["data"] = embedded_data



        except Exception as e:
            err_msg = f"Failed to open or process TIFF file: {file_path}. Error: {e}"
            log.error(err_msg, exc_info=True)
            signals.message.emit(err_msg)
            raise IOError(err_msg)

    # --- Video File Handling (Original Logic) ---
    else:
        raise "Other video files not yet supported. Contact customer support for additional help."

    signals.progress.emit(100)
    signals.message.emit("Frame processing complete.")
    return loaded_frames


def _interpolate_rois_worker(roi_data: dict, pct: float) -> list[dict]:
    """Pure function to interpolate ROIs safely in a background thread."""

    def lerp_int(start, end, pct):
        return int(round(start + (end - start) * pct))

    def lerp_rect(r_min, r_max, pct):
        return QRect(
            lerp_int(r_min.x(), r_max.x(), pct),
            lerp_int(r_min.y(), r_max.y(), pct),
            lerp_int(r_min.width(), r_max.width(), pct),
            lerp_int(r_min.height(), r_max.height(), pct)
        )

    interpolated_rois = []
    for i in range(2):
        min_roi = roi_data["min"][i]
        max_roi = roi_data["max"][i]
        interp_rect = lerp_rect(min_roi["roi_rect"], max_roi["roi_rect"], pct)

        shape_type = min_roi["seed_shape_type"]
        c_min = min_roi["seed_coords"]
        c_max = max_roi["seed_coords"]

        interp_coords = {}
        if shape_type == 'rect':
            interp_coords = {
                'x': lerp_int(c_min['x'], c_max['x'], pct),
                'y': lerp_int(c_min['y'], c_max['y'], pct),
                'w': lerp_int(c_min['w'], c_max['w'], pct),
                'h': lerp_int(c_min['h'], c_max['h'], pct)
            }
        elif shape_type == 'ellipse':
            interp_coords = {
                'center_x': lerp_int(c_min['center_x'], c_max['center_x'], pct),
                'center_y': lerp_int(c_min['center_y'], c_max['center_y'], pct),
                'radius_x': lerp_int(c_min['radius_x'], c_max['radius_x'], pct),
                'radius_y': lerp_int(c_min['radius_y'], c_max['radius_y'], pct)
            }

        interpolated_rois.append({
            "roi_rect": interp_rect,
            "seed_shape_type": shape_type,
            "seed_coords": interp_coords
        })
    return interpolated_rois


def geometry_worker(signals, config: dict):
    """
    The heavy-lifting background task.
    Reads the TIFF file, applies Chan-Vese, and calculates dimensional math.
    """
    file_path = config['file_path']
    distances = config['distances']
    min_dist = config['min_dist']
    max_dist = config['max_dist']
    roi_data = config['roi_data']

    mu = config['mu']
    gamma = config['gamma']
    lambda1 = config['lambda1']

    frames_out = []
    width_masks_out, thickness_masks_out = [], []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    try:
        with TiffFile(file_path) as tif:
            total_frames = len(distances)
            signals.message.emit("Calculating Geometry over all frames...")

            for i, distance in enumerate(distances):
                # 1. Calculate percentage and clamp it mathematically
                pct = (distance - min_dist) / (max_dist - min_dist)
                pct = np.clip(pct, 0.0, 1.0)

                # 2. Get interpolated boxes and seeds
                interp_rois = _interpolate_rois_worker(roi_data, pct)

                # 3. Extract Image Data
                frame = tif.pages[i].asarray()
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame.copy()

                # if i % 10 == 0:  # Only log every 10th frame so we don't spam the console  # Debugging
                #     log.info(f"--- FRAME {i} | Type {gray.dtype} ---")
                #     log.info(f"1. Frame min: {gray.min()}")
                #     log.info(f"1. Frame max: {gray.max()}")

                cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
                gray = gray.astype(np.uint8)

                dimensions = []  # Will hold [width_px, length_px]

                # 4. Process both ROIs (0 = Width, 1 = Length)
                for roi_idx, roi in enumerate(interp_rois):
                    # Crop
                    r = roi['roi_rect']
                    crop = gray[r.y(): r.y() + r.height(), r.x(): r.x() + r.width()]

                    if crop.size == 0:
                        dimensions.append(0)
                        # Ensure we append empty masks to maintain index parity
                        empty_mask = {'mask': np.array([], dtype=bool), 'offset_x': 0, 'offset_y': 0}
                        if roi_idx == 0:
                            width_masks_out.append(empty_mask)
                        else:
                            thickness_masks_out.append(empty_mask)
                        continue

                    # Generate Seed Mask
                    mask_shape = crop.shape
                    seed_mask = np.zeros(mask_shape, dtype=bool)
                    coords = roi['seed_coords']

                    if roi['seed_shape_type'] == 'rect':
                        start = (coords['y'], coords['x'])
                        end = (coords['y'] + coords['h'], coords['x'] + coords['w'])
                        rr, cc = rectangle(start=start, end=end, shape=mask_shape)
                        seed_mask[rr, cc] = True
                    elif roi['seed_shape_type'] == 'ellipse':
                        rr, cc = ellipse(coords['center_y'], coords['center_x'],
                                         coords['radius_y'], coords['radius_x'], shape=mask_shape)
                        seed_mask[rr, cc] = True

                    # Run Chan-Vese
                    img_float = img_as_float(crop)
                    if gamma != 1.0:
                        img_float = img_float ** gamma

                    cv_result = chan_vese(img_float, mu=mu, lambda1=lambda1, lambda2=1.0,
                                          tol=1e-3, max_num_iter=100, dt=0.5,
                                          init_level_set=seed_mask, extended_output=True)  # Ensure tuple output

                    # Convert to binary mask (0 or 1)
                    final_mask = cv_result[0].astype(np.uint8)

                    # Optional Morphology cleanup (ensure mask is 0/255 for cv2)
                    cleaned_mask = cv2.morphologyEx(final_mask * 255, cv2.MORPH_OPEN, kernel)
                    binary_mask = cleaned_mask // 255

                    # if i % 10 == 0:  # Only log every 10th frame so we don't spam the console  ## Debugging
                    #     log.info(f"--- FRAME {i} | ROI {roi_idx} ---")
                    #     log.info(f"1. Crop Shape: {crop.shape}")
                    #     log.info(f"2. Seed Pixels: {np.sum(seed_mask)}")
                    #     log.info(f"3. Chan-Vese Pixels: {np.sum(final_mask)}")
                    #     log.info(f"4. Cleaned Pixels: {np.sum(binary_mask)}")

                    # --- MASK CROPPING & STORAGE ---
                    # Find coordinates of all non-zero pixels
                    y_idx, x_idx = np.nonzero(binary_mask)

                    if len(y_idx) > 0:
                        # Get bounding box of the active mask
                        min_y, max_y = y_idx.min(), y_idx.max()
                        min_x, max_x = x_idx.min(), x_idx.max()

                        # Slice the array and cast to boolean (1 byte per pixel)
                        tight_mask = binary_mask[min_y:max_y + 1, min_x:max_x + 1].astype(bool)
                        mask_data = {
                            'mask': tight_mask,
                            'offset_x': int(min_x),
                            'offset_y': int(min_y)
                        }
                    else:
                        mask_data = {
                            'mask': np.array([], dtype=bool),
                            'offset_x': 0,
                            'offset_y': 0
                        }
                        # -------------------------------

                    # Simply store the masks based on the ROI index
                    if roi_idx == 0:
                        width_masks_out.append(mask_data)
                    else:
                        thickness_masks_out.append(mask_data)

                    # Both ROIs for this frame are processed. Log the frame index.
                frames_out.append(i)

                # Update Progress
                progress_pct = int(((i + 1) / total_frames) * 100)
                signals.progress.emit(progress_pct)

            signals.message.emit("Geometry calculation complete.")

            # Return only the frames and the raw segmentation masks
            return {
                #'idx': frames_out,
                'first_masks': width_masks_out,
                'second_masks': thickness_masks_out
            }

    except Exception as e:
        signals.message.emit(f"Error calculating geometry: {str(e)}")
        raise e
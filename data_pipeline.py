import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
#from processing.data_transform import zero_data, smooth_data, label_image, create_visual_from_labels, convert_numpy, restore_numpy, n_closest_numbers
from processing.data_loader import frame_loader, geometry_worker
from processing.data_transform import auto_thresh
from config import APP_VERSION
from widgets.error_bus import user_error
import cv2
from processing.data_transform import numpy_to_qpixmap

import numpy as np
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.draw import disk, ellipse, rectangle
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


log = logging.getLogger(__name__)

KERNEL_SIZE = 7
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

class DataPipeline(QObject):
    # --- SIGNALS ---
    data_available = Signal(dict)

    # Plot inputs
    plot_selection_changed = Signal(str)
    cycle_selection_changed = Signal(str)

    trimmed_data_available = Signal(object)  # 'object' is safest for passing numpy arrays
    trim_time_changed = Signal(float)

    # Scale inputs
    known_length_changed = Signal(float)
    pixel_length_changed = Signal(float)
    scale_is_manual_changed = Signal(bool)
    manual_conversion_factor_changed = Signal(float)
    conversion_factor_changed = Signal(float)

    # Frames
    left_image_changed = Signal(np.ndarray)
    roi_min_image_loaded = Signal(np.ndarray)
    roi_max_image_loaded = Signal(np.ndarray)

    rois_ready = Signal(list)
    cropped_images_ready = Signal(list)
    seed_masks_ready = Signal(dict)
    threshed_images_ready = Signal(list)

    dimension_images_ready = Signal(np.ndarray, np.ndarray)

    geometry_available = Signal(dict)
    mechanics_available = Signal(dict)

    relaxation_available = Signal(dict)

    exported_file = None


    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_manager = None
        self.VERSION = APP_VERSION
        self.author = None
        self.video_formats = [".tiff", ".tif", ".avi", ".mkv"]

        self.platform = 'mac' if sys.platform == 'darwin' else 'win'
        self.ctrl_key = "Cmd" if self.platform == "mac" else "Ctrl"

        self.video = None
        self.frame_count = 0
        self.start = 0
        self.frame_count = 0

        # PLOT TAB
        self.plot_selection: str = "Time vs. Force"
        self.cycle_selection: str = "All Cycles"
        self.trim_time: float = 0.0
        self.data_trimmed = None

        # SCALE TAB
        self.known_length = 0.0
        self.pixel_length = 0.0
        self.scale_is_manual = False
        self.manual_conversion_factor = 0.0
        self.conversion_factor = 0.0  # The final, authoritative value

        self.left_image: np.ndarray | None = None

        self.data = None
        self.frame_data = {}
        self.roi_data = {
            'min': [],
            'max': []
        }

        self.mu = 0
        self.lambda1 = 0
        self.gamma = 0

        self.xy_roi_idx = 0
        self.thickness_roi_idx = 1

    @Slot(float, object)
    def set_trimmed_data(self, trim_time: float, trimmed_data: np.ndarray):
        """Receives trimmed data from the PlotTab and broadcasts it to the rest of the app."""
        self.trim_time = trim_time
        self.data_trimmed = trimmed_data

        log.info(f"Data trimmed at {self.trim_time}s. Broadcasting new data.")

        self._recalculate_roi_indices()

        # Emit signals so other tabs can update
        self.trim_time_changed.emit(self.trim_time)
        self.trimmed_data_available.emit(self.data_trimmed)


    ### region HEADER

    def on_author_changed(self, new_author):
        self.author = new_author
        log.info(f"User: {self.author}")

    def load_video_file(self, file_path: str, index=None):
        self.video = file_path
        if index is None:
            index = self.start
        self.task_manager.queue_task(
            frame_loader,  # function
            self.video,  # file_path
            [index],  # frame_indices
            True,  # return frame_count
            on_result=self.initial_frame_loaded # Optional: a method in DataPipeline to handle the result
        )

    def initial_frame_loaded(self, result: dict):
        """
        Callback function executed when the frame_loader task is complete.
        'result' is the dictionary of NumPy arrays returned by frame_loader.
        """
        log.info(f"Received {len(result)-2} loaded frames from worker.")
        if not result:
            log.warning("Frame loader returned no frames.")
            return
        it = iter(result)
        first_frame_index = next(it)
        self.left_image = result[first_frame_index]
        if self.left_image is None:
            log.error("Did not receive left frame")
        self.left_image_changed.emit(self.left_image)
        self.frame_count = next(it)
        log.info(f"Frame count found: {self.frame_count}")
        log.info(f"Data keys: {result['data'].keys()}")
        log.info(f"Data found: {result['data']}")
        self.data = result["data"]
        self.max_distance_index = np.argmax(self.data["distance"])
        self.min_distance_index = np.argmin(self.data["distance"])
        self.data_available.emit(self.data)
        self.load_frames([self.max_distance_index, self.min_distance_index])

    def load_frames(self, index):
        self.task_manager.queue_task(
            frame_loader,  # function
            self.video,  # file_path
            index,  # frame_indices
            False,  # return frame_count
            on_result=self.frame_loaded # Optional: a method in DataPipeline to handle the result
        )

    def frame_loaded(self, result):
        for k, v in result.items():
            self.frame_data[k] = v
            if k == self.max_distance_index:
                print("loaded max roi frame")
                self.roi_max_image_loaded.emit(v)
            if k == self.min_distance_index:
                print('loaded min roi frame')
                self.roi_min_image_loaded.emit(v)

    # endregion

    ### region PLOT TAB

    @Slot(str)
    def set_plot_selection(self, selection_text: str):
        """Sets the active plot selection (e.g., 'Time vs. Force')"""
        if self.plot_selection != selection_text:
            self.plot_selection = selection_text
            self.plot_selection_changed.emit(self.plot_selection)

    @Slot(str)
    def set_cycle_selection(self, selection_text: str):
        """Sets the active cycle filter (e.g., 'All Cycles')"""
        if self.cycle_selection != selection_text:
            self.cycle_selection = selection_text
            self.cycle_selection_changed.emit(self.cycle_selection)

    # endregion

    ### region SCALE TAB

    def _recalculate_conversion_factor(self):
        """Central calculation. Called whenever an input changes."""
        new_factor = 0.0
        if self.scale_is_manual:
            new_factor = self.manual_conversion_factor
        elif self.known_length > 0 and self.pixel_length > 0:
            new_factor = self.pixel_length / self.known_length

        # Use the main setter to update the value and emit the signal
        self.set_conversion_factor(new_factor, force_update=True)

    def set_known_length(self, length: float):
        if self.known_length != length:
            self.known_length = length
            self.known_length_changed.emit(self.known_length)
            self._recalculate_conversion_factor()

    def set_pixel_length(self, length: float):
        if self.pixel_length != length:
            self.pixel_length = length
            self.pixel_length_changed.emit(self.pixel_length)
            self._recalculate_conversion_factor()

    def set_scale_is_manual(self, is_manual: bool):
        if self.scale_is_manual != is_manual:
            self.scale_is_manual = is_manual
            self.scale_is_manual_changed.emit(self.scale_is_manual)
            self._recalculate_conversion_factor()

    def set_manual_conversion_factor(self, factor: float):
        """This is called when the user types in the manual spinbox."""
        if self.manual_conversion_factor != factor:
            self.manual_conversion_factor = factor
            # Only recalculate if we are currently in manual mode
            self.manual_conversion_factor_changed.emit(self.manual_conversion_factor)
            if self.scale_is_manual:
                self._recalculate_conversion_factor()

    def set_conversion_factor(self, factor: float, force_update=False):
        """
        This is the final setter for the authoritative value.
        It's called by the recalculate method or can be set directly.
        """
        if self.conversion_factor != factor or force_update:
            self.conversion_factor = factor
            log.info(f"Conversion factor updated: {self.conversion_factor}")
            #self.data_version += 1
            self.conversion_factor_changed.emit(self.conversion_factor)

    # endregion

    ### region ROI TAB

    def _recalculate_roi_indices(self):
        """Calculates min/max distance indices from the active (trimmed) dataset."""
        if self.data_trimmed is None or self.data_trimmed.size == 0:
            return

        # np.argmax/argmin return the integer row index.
        # Since we only trim the tail, these row indices still map correctly
        # to the original frame indices.
        new_max_idx = int(np.argmax(self.data_trimmed["distance"]))
        new_min_idx = int(np.argmin(self.data_trimmed["distance"]))

        frames_to_load = []

        # Check if the max index got trimmed off
        if new_max_idx != self.max_distance_index:
            log.info(f"Max distance index changed from {self.max_distance_index} to {new_max_idx}")
            self.max_distance_index = new_max_idx
            frames_to_load.append(self.max_distance_index)

        # Check if the min index got trimmed off
        if new_min_idx != self.min_distance_index:
            log.info(f"Min distance index changed from {self.min_distance_index} to {new_min_idx}")
            self.min_distance_index = new_min_idx
            frames_to_load.append(self.min_distance_index)

        # If either index changed, send them to the worker to load the new frames.
        # Once loaded, your existing `frame_loaded` callback will catch them
        # and emit the image signals to update the UI automatically.
        if frames_to_load:
            log.info(f"Loading new ROI frames: {frames_to_load}")
            self.load_frames(frames_to_load)

    def _generate_crops_for_target(self, target):
        """Helper: Slices the arrays and stores them in the roi_data dictionary."""
        if not self.frame_data:
            return

        if target == 'min':
            image = self.frame_data.get(self.min_distance_index)  # Get full-size image
        else:
            image = self.frame_data.get(self.max_distance_index)

        if image is None:
            return

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply crops
        for roi_dict in self.roi_data[target]:
            rect = roi_dict["roi_rect"]
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            if w > 0 and h > 0:
                crop = gray[y:y + h, x:x + w]
                if crop.size > 0:
                    roi_dict["crop_img"] = crop

    def receive_roi_data(self, roi_data, target):
        """
        Receives QRects from ROITab, initializes the nested structure,
        generates crops, and emits QPixmaps for the SeedTab.
        """
        print(f"{len(roi_data)} boxes for {target}: {roi_data}")

        # 1. Reset and populate the list of dictionaries for this target
        self.roi_data[target] = []
        for rect in roi_data:
            self.roi_data[target].append({
                "roi_rect": rect,
                "crop_img": None,
                "seed_shape_type": None,
                "seed_coords": None,
                "seed_mask": None
            })


        self.correlate_rois()

        # 2. Generate crops and store them directly inside the nested dict
        self._generate_crops_for_target(target)

        # 3. Gather all valid crops in order (Min first, then Max) to send to UI
        crops_pixmaps = []
        for t in ['min', 'max']:
            for roi_dict in self.roi_data.get(t, []):
                if roi_dict["crop_img"] is not None:
                    crops_pixmaps.append(numpy_to_qpixmap(roi_dict["crop_img"]))

        # 4. Emit the Pixmaps to the Seed Tab
        if crops_pixmaps:
            self.cropped_images_ready.emit(crops_pixmaps)

    # endregion

    ### region SEED TAB

    def receive_seed_shape(self, index, shape_type, data):
        """
        Called by UI when user finishes drawing. Maps the UI index back
        to the nested dictionary and stores the seed data/mask.
        """
        # 1. Map the flat UI index (0, 1, 2, 3) to target ('min' or 'max') and sub-index
        target = None
        sub_index = 0
        min_len = len(self.roi_data.get('min', []))
        max_len = len(self.roi_data.get('max', []))

        if index < min_len:
            target = 'min'
            sub_index = index
        elif index < min_len + max_len:
            target = 'max'
            sub_index = index - min_len
        else:
            return  # Index out of bounds (user clicked an empty canvas)

        # 2. Grab the specific dictionary for this ROI
        roi_dict = self.roi_data[target][sub_index]
        crop_img = roi_dict.get("crop_img")

        if crop_img is None:
            return

        # 3. Handle clearing the shape (Undo/Reset)
        if shape_type is None or not data:
            roi_dict["seed_shape_type"] = None
            roi_dict["seed_coords"] = None
            roi_dict["seed_mask"] = None
            print(f"Mask cleared for image {index} ({target} ROI {sub_index})")
            return

        # 4. Generate the Mask
        image_shape = crop_img.shape
        mask = np.zeros(image_shape, dtype=bool)

        if shape_type == 'rect':
            start = (data['y'], data['x'])
            end = (data['y'] + data['h'], data['x'] + data['w'])
            rr, cc = rectangle(start=start, end=end, shape=image_shape)
            mask[rr, cc] = True

        elif shape_type == 'ellipse':
            rr, cc = ellipse(data['center_y'], data['center_x'],
                             data['radius_y'], data['radius_x'], shape=image_shape)
            mask[rr, cc] = True

        # 5. Save everything into the unified dictionary
        roi_dict["seed_shape_type"] = shape_type
        roi_dict["seed_coords"] = data
        roi_dict["seed_mask"] = mask
        print(f"Mask created for image {index} ({target} ROI {sub_index}) with shape {shape_type}")

        # 6. Check if we have all required masks to emit
        # (Extracts non-None masks from the nested structure)
        valid_masks = [
            item["seed_mask"]
            for t in ['min', 'max']
            for item in self.roi_data.get(t, [])
            if item.get("seed_mask") is not None
        ]

        if len(valid_masks) == 4:
            self.seed_masks_ready.emit(valid_masks)

    # endregion

    ### region THRESHOLD TAB

    def run_chan_vese(self, img, mu, gamma, lambda1, seed):
        image = img_as_float(img)
        if gamma != 1.0:
            image = image ** gamma

        cv_result = chan_vese(image,
                              mu=mu,
                              lambda1=lambda1,
                              lambda2=1.0,
                              tol=1e-3,
                              max_num_iter=100,
                              dt=0.5,
                              init_level_set=seed,
                              extended_output=True)

        final_mask = cv_result[0]
        # final_energies = cv_result[2]  # To check convergence
        raw_mask = final_mask.astype(np.uint8) * 255
        cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, KERNEL)

        return cleaned_mask

    def apply_threshold(self, smoothness_val: int, shadow_recovery_val: int):
        """
        Applies Chan-Vese segmentation to all valid crops using their paired seed masks.
        """
        processed_pixmaps = []

        # 1. Gather all valid ROI dictionaries that have both a crop and a mask
        valid_rois = []
        for target in ['min', 'max']:
            for roi_dict in self.roi_data.get(target, []):
                crop = roi_dict.get("crop_img")
                mask = roi_dict.get("seed_mask")

                if crop is not None and mask is not None:
                    valid_rois.append(roi_dict)

        # Ensure we have all 4 before doing heavy computation
        if len(valid_rois) != 4:
            print("Waiting for all 4 seed masks to be drawn...")
            return

        # 2. Process each valid ROI
        for roi_dict in valid_rois:
            img = roi_dict["crop_img"]
            seed_mask = roi_dict["seed_mask"]

            # Convert to float for Chan-Vese (skimage handles the 0.0-1.0 mapping)

            mu_norm = (smoothness_val - 1) / 99.0
            mu = 0.001 + (mu_norm ** 2) * (0.2 - 0.001)

            # 2. Calculate Coupled Shadow Parameters (Linear scale)
            # shadow_recovery_val is 0 to 100
            shadow_norm = shadow_recovery_val / 100.0

            # Map 0->100 to Gamma 1.0->0.5
            gamma_val = 1.0 - (shadow_norm * 0.5)

            # Map 0->100 to Lambda1 1.0->0.2
            lambda1_val = 1.0 - (shadow_norm * 0.8)

            cleaned_mask = self.run_chan_vese(img, mu, gamma_val, lambda1_val, seed_mask)

            # --- OVERLAY LOGIC ---
            # 1. Ensure the original crop is 3-channel (BGR) so we can apply color
            if len(img.shape) == 2:
                base_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                base_bgr = img.copy()

            # 2. Define the overlay color and transparency level
            overlay_color = (0, 255, 0)  # BGR format: Green (Change to 0,0,255 for Red)
            alpha = 0.4  # Transparency (0.0 is invisible, 1.0 is a solid block of color)

            # 3. Create a copy of the base image and paint the mask area solid green
            color_layer = base_bgr.copy()
            color_layer[cleaned_mask == 255] = overlay_color

            # 4. Blend the solid color layer with the original image
            blended_output = cv2.addWeighted(color_layer, alpha, base_bgr, 1 - alpha, 0)

            # 5. Convert to QPixmap (your helper already handles BGR -> RGB conversion)
            pix = numpy_to_qpixmap(blended_output)
            processed_pixmaps.append(pix)

            roi_dict["cv_img"] = cleaned_mask

        self.mu = mu
        self.lambda1 = lambda1_val
        self.gamma_val = gamma_val

        # 5. Emit back to the UI
        self.threshed_images_ready.emit(processed_pixmaps)

    # endregion

    ### region GEOMETRY IMAGES

    def request_dimension_images(self):
        """
        Called by the GeometryTab when it opens or when dimensions are swapped.
        Fetches the 'min' state crops based on current index tracking and emits them.
        """
        min_rois = self.roi_data.get('min', [])
        if len(min_rois) >= 2:
            w_img = min_rois[self.xy_roi_idx].get('cv_img')
            t_img = min_rois[self.thickness_roi_idx].get('cv_img')

            if w_img is not None and t_img is not None:
                self.dimension_images_ready.emit(w_img, t_img)

    def swap_dimensions(self):
        """
        Flips the state mapping for Width and Thickness, then updates the UI.
        """
        # Swap the tracking indices
        self.xy_roi_idx, self.thickness_roi_idx = self.thickness_roi_idx, self.xy_roi_idx
        print(f"Swapped Dimensions! Width is now ROI {self.xy_roi_idx}, Thickness is ROI {self.thickness_roi_idx}")

        # Emit the new images to immediately update the visual reference in the UI
        self.request_dimension_images()
        if hasattr(self, 'first_segments'):
            self.calculate_dimensions()

        # NOTE: If we already have computed segmentation masks stored,
        # we will eventually call our math recalculation function right here!

    def correlate_rois(self):
        """
        Pairs the 'min' ROIs with the correct 'max' ROIs based on spatial overlap.
        Reorders the 'max' list in self.roi_data to perfectly match the 'min' list.
        """
        min_data = self.roi_data.get('min', [])
        max_data = self.roi_data.get('max', [])

        # We need exactly 2 ROIs in both states to perform a correlation
        if len(min_data) != 2 or len(max_data) != 2:
            print("Cannot correlate: Need exactly 2 ROIs in both min and max states.")
            return

        m0_rect = min_data[0]["roi_rect"]
        m1_rect = min_data[1]["roi_rect"]

        x0_rect = max_data[0]["roi_rect"]
        x1_rect = max_data[1]["roi_rect"]

        def get_overlap_area(rect1, rect2):
            """Helper to calculate the overlapping area of two QRects."""
            intersection = rect1.intersected(rect2)
            if intersection.isValid():
                return intersection.width() * intersection.height()
            return 0

        # --- Test Configuration A ---
        # Min 0 pairs with Max 0
        # Min 1 pairs with Max 1
        score_a = get_overlap_area(m0_rect, x0_rect) + get_overlap_area(m1_rect, x1_rect)

        # --- Test Configuration B ---
        # Min 0 pairs with Max 1
        # Min 1 pairs with Max 0
        score_b = get_overlap_area(m0_rect, x1_rect) + get_overlap_area(m1_rect, x0_rect)

        # --- Apply the Best Match ---
        if score_a >= score_b:
            print("Correlation: Min 0 -> Max 0, Min 1 -> Max 1")
            # They are already in the correct order, do nothing.
        else:
            print("Correlation: Min 0 -> Max 1, Min 1 -> Max 0")
            # Swap the max list so the indices match the min list
            self.roi_data['max'] = [max_data[1], max_data[0]]

    def get_interpolated_data(self, pct: float) -> list[dict]:
        """
        Calculates the interpolated ROI bounding boxes and seed shapes between min and max states.
        Returns a list of dictionaries structured identically to the base roi_data items.
        """
        if not self.roi_data.get("min") or not self.roi_data.get("max"):
            return []

        def lerp_int(start: float, end: float, pct: float) -> int:
            """Standard linear interpolation, correctly rounded to the nearest integer pixel."""
            return int(round(start + (end - start) * pct))

        def lerp_rect(r_min: QRect, r_max: QRect, pct: float) -> QRect:
            """Interpolates all four dimensions of a QRect."""
            x = lerp_int(r_min.x(), r_max.x(), pct)
            y = lerp_int(r_min.y(), r_max.y(), pct)
            w = lerp_int(r_min.width(), r_max.width(), pct)
            h = lerp_int(r_min.height(), r_max.height(), pct)
            return QRect(x, y, w, h)

        def lerp_seed(min_dict: dict, max_dict: dict, pct: float) -> tuple:
            """Interpolates the seed coordinates if the shape types match."""
            shape_type = min_dict.get("seed_shape_type")

            # Enforce shape matching (cannot interpolate a rect into an ellipse)
            if not shape_type or shape_type != max_dict.get("seed_shape_type"):
                return None, None

            c_min = min_dict.get("seed_coords")
            c_max = max_dict.get("seed_coords")

            if not c_min or not c_max:
                return None, None

            interp_coords = {}

            if shape_type == 'rect':
                interp_coords['x'] = lerp_int(c_min['x'], c_max['x'], pct)
                interp_coords['y'] = lerp_int(c_min['y'], c_max['y'], pct)
                interp_coords['w'] = lerp_int(c_min['w'], c_max['w'], pct)
                interp_coords['h'] = lerp_int(c_min['h'], c_max['h'], pct)

            elif shape_type == 'ellipse':
                interp_coords['center_x'] = lerp_int(c_min['center_x'], c_max['center_x'], pct)
                interp_coords['center_y'] = lerp_int(c_min['center_y'], c_max['center_y'], pct)
                interp_coords['radius_x'] = lerp_int(c_min['radius_x'], c_max['radius_x'], pct)
                interp_coords['radius_y'] = lerp_int(c_min['radius_y'], c_max['radius_y'], pct)

            return shape_type, interp_coords

        interpolated_rois = []

        # Iterate through the paired ROIs (assuming correlate_rois() was already run)
        for i in range(2):
            min_roi = self.roi_data["min"][i]
            max_roi = self.roi_data["max"][i]

            # 1. Interpolate the global bounding box
            interp_rect = lerp_rect(min_roi["roi_rect"], max_roi["roi_rect"], pct)

            # 2. Interpolate the local seed shape
            shape_type, interp_coords = lerp_seed(min_roi, max_roi, pct)

            # 3. Package it into the standard dictionary format
            interpolated_rois.append({
                "roi_rect": interp_rect,
                "seed_shape_type": shape_type,
                "seed_coords": interp_coords,
                "crop_img": None,  # To be filled when the specific frame is loaded
                "seed_mask": None  # To be generated from the coords
            })

        return interpolated_rois

    def get_geometry(self):
        """
        Packages current state data and kicks off the background calculation task.
        """
        if not self.video or self.data is None or "distance" not in self.data:
            print("Cannot calculate geometry: Data or video not loaded.")
            return

        if len(self.roi_data.get('min', [])) != 2 or len(self.roi_data.get('max', [])) != 2:
            print("Cannot calculate geometry: ROIs not fully established.")
            return

        distances = self.data_trimmed["distance"]

        # Package a "snapshot" so the thread doesn't read live UI variables
        snapshot_config = {
            'file_path': self.video,
            'distances': distances,
            'min_dist': distances[self.min_distance_index],
            'max_dist': distances[self.max_distance_index],
            'roi_data': self.roi_data,  # QRects and dicts are safe to pass by reference if we only read them
            'conversion_factor': self.conversion_factor,
            'mu': getattr(self, 'mu', 0.05),  # Fallbacks in case apply_threshold wasn't run
            'gamma': getattr(self, 'gamma_val', 1.0),
            'lambda1': getattr(self, 'lambda1', 1.0)
        }

        # Queue the heavy-lifting task
        self.task_manager.queue_task(
            geometry_worker,  # The pure worker function
            snapshot_config,  # The configuration dict
            on_result=self._on_geometry_computed
        )

    def _on_geometry_computed(self, result: dict):
        """
        Callback triggered when the background task finishes successfully.
        """
        print("Geometry computed successfully. Calculating dimensions...")
        self.first_segments = result["first_masks"]
        self.second_segments = result["second_masks"]
        self.calculate_dimensions()


    # endregion

    def calculate_dimensions(self):
        """
        Calculates X, Y, and Z dimensions from the stored segmentation masks.
        Calculates the Y-Z cross-sectional area (normal to the X stretching direction).
        Emits the final data payload to the GeometryTab.
        """
        if not hasattr(self, 'first_segments') or not hasattr(self, 'second_segments'):
            print("No segmentation data available to calculate dimensions.")
            return

        print("Calculating XYZ dimensions from stored masks...")

        frames = []
        raw_x = []
        raw_y = []
        raw_z = []

        # 1. Map the ROIs to the XY image and the Z image
        # Assuming xy_roi_idx tracks the first image (XY Orientation)
        if self.xy_roi_idx == 0:
            xy_masks_list = self.first_segments
            z_masks_list = self.second_segments
        else:
            xy_masks_list = self.second_segments
            z_masks_list = self.first_segments

        num_frames = len(xy_masks_list)
        cf = self.conversion_factor if self.conversion_factor > 0 else 1.0

        for i in range(num_frames):
            frames.append(i)

            # --- XY Image Processing ---
            xy_mask = xy_masks_list[i]['mask']
            if xy_mask.size > 0:
                # X Measure: Horizontal sum (across columns -> axis=1)
                x_sums = np.sum(xy_mask, axis=1)
                valid_x = x_sums[x_sums > 0]
                raw_x.append(np.mean(valid_x) if len(valid_x) > 0 else 0)

                # Y Measure: Vertical sum (across rows -> axis=0)
                y_sums = np.sum(xy_mask, axis=0)
                valid_y = y_sums[y_sums > 0]
                raw_y.append(np.mean(valid_y) if len(valid_y) > 0 else 0)
            else:
                raw_x.append(0)
                raw_y.append(0)

            # --- Z Image Processing ---
            z_mask = z_masks_list[i]['mask']
            if z_mask.size > 0:
                # Z Measure: Vertical sum (across rows -> axis=0)
                z_sums = np.sum(z_mask, axis=0)
                valid_z = z_sums[z_sums > 0]
                raw_z.append(np.mean(valid_z) if len(valid_z) > 0 else 0)
            else:
                raw_z.append(0)

        # 2. Convert to real-world units (mm)
        x_mm = np.array(raw_x) / cf
        y_mm = np.array(raw_y) / cf
        z_mm = np.array(raw_z) / cf

        # 3. Calculate Load-Bearing Cross-Sectional Area (Y-Z Plane)
        raw_areas = y_mm * z_mm

        # 4. Setup Savitzky-Golay Smoothing
        calc_window = int(len(frames) * 0.05)
        window_length = max(5, calc_window if calc_window % 2 != 0 else calc_window + 1)

        # 5. Apply Smoothing to all arrays
        if window_length > 3 and len(frames) >= window_length:
            from scipy.signal import savgol_filter
            # Explicitly setting mode='nearest' to prevent polynomial boundary wiggles
            smooth_x = savgol_filter(x_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_y = savgol_filter(y_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_z = savgol_filter(z_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_area = savgol_filter(raw_areas, window_length=window_length, polyorder=3, mode='nearest')
        else:
            smooth_x, smooth_y, smooth_z, smooth_area = x_mm, y_mm, z_mm, raw_areas

        # 6. Package and Emit
        result = {
            'frames': frames,
            'dim_x': smooth_x.tolist(),
            'dim_y': smooth_y.tolist(),
            'dim_z': smooth_z.tolist(),
            'area': smooth_area.tolist()
        }

        self.geometry_available.emit(result)
        print("XYZ Dimensions calculated and emitted to UI.")

        # Save the payload so the mechanics function can use it
        self.geometry_data = result
        self.calculate_mechanics()
        self.calculate_relaxation()

    # endregion

    def calculate_mechanics(self):
        """
        Derives True Stress, Strain, Poisson's Ratios, and Energy Dissipation
        using strict X, Y, Z coordinate mapping.
        """
        print("Calculating biomechanics...")
        if getattr(self, 'geometry_data', None) is None or self.data is None:
            print("Cannot calculate mechanics: Missing geometry or machine data.")
            return

        # 1. Gather Arrays
        geom = self.geometry_data
        dim_x_opt = np.array(geom['dim_x'])  # Optical transverse loading direction
        dim_y = np.array(geom['dim_y'])  # Optical longitudinal unloaded direction
        dim_z = np.array(geom['dim_z'])  # Optical transverse unloaded direction
        area = np.array(geom['area'])  # Cross-section normal to X (Y * Z)
        force_mN = self.data_trimmed['force']
        if "2026-02-05" in str(self.video):  # Quirk for backwards load cell data
            force_mN = -force_mN
        machine_x = self.data_trimmed['distance']  # Mechanical loading direction
        cycle_flags = self.data_trimmed['cycle']
        time_s = self.data_trimmed['time_s']

        # Failsafe: Ensure arrays are perfectly aligned in length
        min_len = min(len(dim_x_opt), len(force_mN))
        if min_len == 0:
            return

        dim_x_opt = dim_x_opt[:min_len]
        dim_y = dim_y[:min_len]
        dim_z = dim_z[:min_len]
        area = area[:min_len]

        force_mN = force_mN[:min_len]
        machine_x = machine_x[:min_len]
        cycle_flags = cycle_flags[:min_len]
        time_s = time_s[:min_len]

        # 2. Establish Rest State (t=0 Baselines)
        X0_opt = np.mean(dim_x_opt[:3]) if np.mean(dim_x_opt[:3]) != 0 else 1e-6
        Y0 = np.mean(dim_y[:3]) if np.mean(dim_y[:3]) != 0 else 1e-6
        Z0 = np.mean(dim_z[:3]) if np.mean(dim_z[:3]) != 0 else 1e-6
        A0 = np.mean(area[:3]) if np.mean(area[:3]) != 0 else 1e-6
        X0_mech = np.mean(machine_x[:3]) if np.mean(machine_x[:3]) != 0 else 1e-6

        # 3. Continuous Array Math

        # Stress calculations (kPa = mN / mm^2)
        true_stress_kpa = force_mN / area
        eng_stress_kpa = force_mN / A0

        # Stretch Ratios (Lambda)
        stretch_x_opt = dim_x_opt / X0_opt
        stretch_y = dim_y / Y0
        stretch_z = dim_z / Z0
        stretch_x_mech = machine_x / X0_mech

        # Logarithmic (True) Strain (epsilon = ln(lambda))
        true_strain_x_opt = np.log(stretch_x_opt)
        true_strain_y = np.log(stretch_y)
        true_strain_z = np.log(stretch_z)
        true_strain_x_mech = np.log(stretch_x_mech)

        # --- UPDATED POISSON'S RATIO LOGIC ---
        # Widen the deadband to 2% strain to avoid the divide-by-zero asymptote at cycle boundaries.
        # Use np.nan so the plot leaves a clean gap when resting, rather than snapping to 0.5.
        strain_threshold = 0.02

        poissons_ratio_zx = np.where(np.abs(true_strain_x_opt) > strain_threshold,
                                     -true_strain_z / true_strain_x_opt,
                                     np.nan)

        poissons_ratio_yx = np.where(np.abs(true_strain_x_opt) > strain_threshold,
                                     -true_strain_y / true_strain_x_opt,
                                     np.nan)

        # 4. Parse Cycles & Calculate Hysteresis Energy
        unique_cycles = np.unique(cycle_flags)
        cycle_parsing = {}
        energy_dissipated = []

        for c in unique_cycles:
            mask = (cycle_flags == c)
            idx = np.where(mask)[0]

            if len(idx) < 3:
                continue

            # We use the mechanical strain for the integral to prevent pixel noise
            # from causing artificial energy loops
            c_strain = true_strain_x_opt[idx]
            c_stress = true_stress_kpa[idx]
            c_dist = machine_x[idx]

            # Find the exact peak to split the cycle
            peak_relative_idx = np.argmax(c_dist)
            peak_global_idx = idx[peak_relative_idx]

            # --- Explicit Energy Integration ---
            # 1. Isolate loading and unloading arrays
            load_strain = c_strain[:peak_relative_idx + 1]
            load_stress = c_stress[:peak_relative_idx + 1]

            unload_strain = c_strain[peak_relative_idx:]
            unload_stress = c_stress[peak_relative_idx:]

            # 2. Calculate Work In (Area under loading curve)
            w_in = np.trapz(load_stress, load_strain)

            # 3. Calculate Work Out (Area under unloading curve)
            # np.trapz returns a negative value because strain is decreasing, so we use abs()
            w_out = abs(np.trapz(unload_stress, unload_strain))

            # 4. Dissipated Energy = Work In - Work Out
            dissipated = w_in - w_out

            # 5. Safeguard: Tissue cannot generate energy.
            # Any negative value is pure sensor lag/noise in the fully elastic regime.
            dissipated_mJ_mm3 = max(0.0, dissipated)

            energy_dissipated.append(dissipated_mJ_mm3)

            # Store the indices so the UI can easily color-code Loading vs. Unloading curves
            cycle_parsing[int(c)] = {
                'full_idx': idx.tolist(),
                'peak_idx': int(peak_global_idx),
                'loading_idx': idx[:peak_relative_idx + 1].tolist(),
                'unloading_idx': idx[peak_relative_idx:].tolist()
            }

        # 5. Package and Emit Payload
        mechanics_payload = {
            'time_s': time_s.tolist(),
            'true_stress_kpa': true_stress_kpa.tolist(),
            'eng_stress_kpa': eng_stress_kpa.tolist(),

            # Export all strain variants
            'strain_x_opt': true_strain_x_opt.tolist(),
            'strain_x_mech': true_strain_x_mech.tolist(),
            'strain_y': true_strain_y.tolist(),
            'strain_z': true_strain_z.tolist(),

            # Export all stretch variants
            'stretch_x_opt': stretch_x_opt.tolist(),
            'stretch_x_mech': stretch_x_mech.tolist(),
            'stretch_y': stretch_y.tolist(),
            'stretch_z': stretch_z.tolist(),

            'poissons_ratio_zx': poissons_ratio_zx.tolist(),
            'poissons_ratio_yx': poissons_ratio_yx.tolist(),
            'energy_dissipated': energy_dissipated,
            'cycle_parsing': cycle_parsing
        }

        self.mechanics_payload = mechanics_payload
        self.mechanics_available.emit(mechanics_payload)
        print("Biomechanics calculated and emitted!")

    def calculate_relaxation(self):
        """
        Parses a stress relaxation hold, isolating the active motor ramp using mechanical strain,
        and calculating moduli using the true optical tissue strain.
        """
        print("Calculating stress relaxation metrics...")
        if getattr(self, 'mechanics_payload', None) is None:
            return

        from scipy.optimize import curve_fit

        data = self.mechanics_payload
        geom = self.geometry_data
        time_s = np.array(data['time_s'])
        stress = np.array(data['true_stress_kpa'])
        dim_z = np.array(geom['dim_z'])

        # Explicitly separate the two strain arrays for their specific jobs
        strain_mech = np.array(data['strain_x_mech'])
        strain_opt = np.array(data['strain_x_opt'])

        # 1. Isolate the Active Loading Ramp (Jaws Moving)
        # -> STRICTLY using the smooth mechanical strain as the trigger
        strain_rate = np.diff(strain_mech)

        # Use a small threshold (e.g., 5% of max speed) to ignore baseline noise
        movement_threshold = np.max(strain_rate) * 0.05
        moving_indices = np.where(strain_rate > movement_threshold)[0]

        if len(moving_indices) == 0:
            print("No jaw movement detected in data.")
            return

        # The exact frames the motor started and stopped
        ramp_start_idx = moving_indices[0]
        hold_start_idx = moving_indices[-1] + 1

        # 2. Find the True Peak Stress (Top of the compliance hump)
        peak_relative_idx = np.argmax(stress[hold_start_idx:])
        peak_stress_idx = hold_start_idx + peak_relative_idx
        peak_stress = stress[peak_stress_idx]

        # 3. Calculate Instantaneous Modulus (E_inst)
        # -> STRICTLY using the optical strain for the math
        baseline_stress = stress[ramp_start_idx]
        baseline_strain = strain_opt[ramp_start_idx]

        delta_stress = stress[hold_start_idx] - baseline_stress
        delta_strain = strain_opt[hold_start_idx] - baseline_strain

        if delta_strain > 0:
            e_inst = delta_stress / delta_strain
        else:
            e_inst = 0.0

        # 4. Prepare Hold Data for SLS Fitting
        hold_time_raw = time_s[peak_stress_idx:]
        hold_time_shifted = hold_time_raw - hold_time_raw[0]
        hold_stress = stress[peak_stress_idx:]
        final_recorded_stress = hold_stress[-1]

        def two_tau_model(t, sigma_inf, sigma_1, tau_1, sigma_2, tau_2):
            return sigma_inf + sigma_1 * np.exp(-t / tau_1) + sigma_2 * np.exp(-t / tau_2)

        total_viscous_drop = peak_stress - final_recorded_stress

        # Guesses: [Asymptote, Fast Drop, Fast Tau, Slow Drop, Slow Tau]
        p0 = [final_recorded_stress, total_viscous_drop * 0.5, 1.0, total_viscous_drop * 0.5, 50.0]
        bounds = (0, np.inf)

        try:
            popt, _ = curve_fit(two_tau_model, hold_time_shifted, hold_stress, p0=p0, bounds=bounds, maxfev=5000)

            # Unpack the raw optimized parameters
            sigma_inf_raw, s1, t1, s2, t2 = popt

            # Sort them so tau_1 is ALWAYS the "fast" phase, and tau_2 is the "slow" phase
            if t1 > t2:
                t1, t2 = t2, t1
                s1, s2 = s2, s1

            sigma_inf, sigma_1, tau_1, sigma_2, tau_2 = sigma_inf_raw, s1, t1, s2, t2

            # Generate the fitted curve array for the UI
            fitted_stress = two_tau_model(hold_time_shifted, sigma_inf, sigma_1, tau_1, sigma_2, tau_2)

        except Exception as e:
            print(f"2-Tau Curve fit failed: {e}")
            sigma_inf = final_recorded_stress
            sigma_1, tau_1 = (total_viscous_drop * 0.5), 1.0
            sigma_2, tau_2 = (total_viscous_drop * 0.5), 50.0
            fitted_stress = hold_stress

        # 6. Final Derived Metrics
        relaxed_delta_stress = sigma_inf - baseline_stress

        # -> STRICTLY using the optical strain for the math
        if delta_strain > 0:
            e_inf = relaxed_delta_stress / delta_strain
        else:
            e_inf = 0.0

        percent_relaxation = ((peak_stress - sigma_inf) / peak_stress) * 100 if peak_stress > 0 else 0.0

        # 7. Package and Emit
        relax_payload = {
            'time_s': time_s.tolist(),
            'stress_kpa': stress.tolist(),
            'dim_z': dim_z.tolist(),
            'hold_start_idx': int(hold_start_idx),
            'peak_stress_idx': int(peak_stress_idx),
            'hold_time_raw': hold_time_raw.tolist(),
            'fitted_stress': fitted_stress.tolist(),
            'metrics': {
                'peak_stress': float(peak_stress),
                'sigma_inf': float(sigma_inf),
                'sigma_1': float(sigma_1),
                'tau_1': float(tau_1),
                'sigma_2': float(sigma_2),
                'tau_2': float(tau_2),
                'e_inst': float(e_inst),
                'e_inf': float(e_inf),
                'percent_relax': float(percent_relaxation)
            }
        }

        self.relaxation_available.emit(relax_payload)
        print("Relaxation calculated and emitted!")

    ### region EXPORT TAB

    def generate_report(self):
        """Prepares the trimmed data and calculated mechanics for CSV export."""
        print("Gathering data for export...")

        # 2. Safely handle array lengths
        num_rows = len(self.data_trimmed)

        if num_rows == 0:
            print("Cannot export: Data arrays are empty.")
            return

        # 3. Define the translation map for your headers
        header_mapping = {
            "time_s": "Time (s)",
            "force": "Force (mN)",
            "distance": "Distance (mm)",
            "cycle": "Cycle Number"
        }

        report_data = {}

        # 4. Grab columns from data_trimmed and apply mapped headers
        for raw_col in self.data_trimmed.dtype.names:
            # Use the mapped name if it exists, otherwise fallback to title-casing the raw name
            friendly_name = header_mapping.get(raw_col, raw_col.replace("_", " ").title())
            report_data[friendly_name] = self.data_trimmed[raw_col][:num_rows].tolist()

        # 5. Add the calculated mechanics columns with readable headers
        report_data["X Dimension (mm)"] = self.geometry_data["x_dim"][:num_rows].tolist()
        report_data["Y Dimension (mm)"] = self.geometry_data["y_dim"][:num_rows].tolist()
        report_data["Z Dimension (mm)"] = self.geometry_data["z_dim"][:num_rows].tolist()
        report_data["YZ Area (mm^2)"] = self.geometry_data["area"][:num_rows].tolist()
        report_data["True Stress (kPa)"] = self.mechanics_payload["true_stress"][:num_rows].tolist()
        report_data["Stretch X Tissue"] = self.mechanics_payload["stretch_x_opt"][:num_rows].tolist()
        report_data["Stretch X Jaws"] = self.mechanics_payload["stretch_x_mech"][:num_rows].tolist()

        if hasattr(self, 'relaxation_available') and getattr(self, 'relaxation_payload', None) is not None:
            rel_data = self.relaxation_payload
            peak_idx = rel_data['peak_stress_idx']
            fit_array = rel_data['fitted_stress']
            m = rel_data['metrics']

            # A. Pad the fitted stress array with NaNs so it aligns perfectly with time
            padded_fit = np.full(num_rows, np.nan)
            available_slots = num_rows - peak_idx

            if available_slots > 0:
                fit_len = min(len(fit_array), available_slots)
                padded_fit[peak_idx: peak_idx + fit_len] = fit_array[:fit_len]

            report_data["Fitted Stress 2-Tau (kPa)"] = padded_fit.tolist()

            # B. Append scalar metrics as columns (Value in Row 1, blanks below)
            scalar_metrics = {
                "Peak Stress (kPa)": m['peak_stress'],
                "Step Modulus E_step (kPa)": m['e_inst'],
                "Relaxed Modulus E_inf (kPa)": m['e_inf'],
                "Total Relaxation (%)": m['percent_relax'],
                "Eq Stress Model (kPa)": m['sigma_inf'],
                "Fast Visc Drop (kPa)": m['sigma_1'],
                "Fast Tau (s)": m['tau_1'],
                "Slow Visc Drop (kPa)": m['sigma_2'],
                "Slow Tau (s)": m['tau_2']
            }

            for col_name, val in scalar_metrics.items():
                col_data = np.full(num_rows, np.nan)
                if num_rows > 0:
                    col_data[0] = val  # Drop the metric into the very first row
                report_data[col_name] = col_data.tolist()


        # 6. Pass to the CSV writer
        self.write_csv_report(report_data, num_rows)


    def write_csv_report(self, report_data: dict, num_rows: int):
        """Handles the logic of writing the final report to a CSV file."""
        filename = os.path.splitext(os.path.basename(self.video))[0].removesuffix("_video")
        folder = os.path.dirname(self.video)
        filepath = Path(f"{folder}/{filename}_results.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Handle existing files by incrementing a counter
        i = 1
        base_stem = filepath.stem
        while filepath.exists():
            i += 1
            filepath = filepath.with_name(f"{base_stem}_{i}.csv")

        header = ",".join(report_data.keys())
        with open(filepath, 'w', newline='') as f:
            f.write(header + '\n')
            for i in range(num_rows):
                row_values = [col[i] for col in report_data.values()]
                row_values = [np.nan if not np.isfinite(val) else val for val in row_values]
                formatted_row = [('' if np.isnan(val) else f'{val:.10g}') for val in row_values]
                f.write(','.join(formatted_row) + '\n')
        print(f"Report successfully written to {filepath}")
        self.exported_file = filepath

    # endregion